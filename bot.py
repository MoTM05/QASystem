import logging
import os
import asyncio
from typing import Optional

from telegram import Update, File as TelegramFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import \
    StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter


BOT_TOKEN = "YOUR_TOKEN_HERE"
FAISS_INDEX_PATH = "faiss_AiDoc2"
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
TEMP_UPLOAD_DIR = "temp_uploads_bot"


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


qa_system: Optional[RetrievalQA] = None
embeddings_model: Optional[HuggingFaceEmbeddings] = None


def get_embeddings():
    global embeddings_model
    if embeddings_model is None:
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {EMBEDDING_MODEL_NAME}")
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings_model


def initialize_llm():
    logger.info(f"Loading Llama-2 model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file {MODEL_PATH} not found!")
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please download it.")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.2,
        max_tokens=512,
        top_p=1,
        callback_manager=callback_manager,
        n_ctx=4096,
        verbose=False,
        n_gpu_layers=0
    )
    return llm


def check_faiss_index_exists(path: str) -> bool:
    """Проверяет наличие файлов index.faiss и index.pkl в указанной директории."""
    index_file = os.path.join(path, "index.faiss")
    pkl_file = os.path.join(path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)


def initialize_qa_chain(force_rebuild_chain_only=False):
    global qa_system

    current_embeddings = get_embeddings()

    if not check_faiss_index_exists(FAISS_INDEX_PATH):
        logger.warning(
            f"FAISS index files not found in '{FAISS_INDEX_PATH}'. "
            "QA system will not be available until a document is uploaded and index is created."
        )
        qa_system = None
        return False

    if qa_system is not None and not force_rebuild_chain_only:
        logger.info("QA system already initialized and using existing index.")
        return True

    try:
        logger.info(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")

        loop = asyncio.get_event_loop()
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            current_embeddings,
            allow_dangerous_deserialization=True
        )

        llm = initialize_llm()

        qa_system = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        logger.info("QA system initialized successfully with FAISS index.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize QA system: {str(e)}", exc_info=True)
        qa_system = None
        return False


def extract_text_from_file(file_path: str) -> str:
    """Извлечь текст из файла на основе его расширения"""
    try:
        if file_path.lower().endswith('.pdf'):
            from pdfminer.high_level import extract_text as pdf_extract_text
            logger.info(f"Extracting text from PDF: {file_path}")
            return pdf_extract_text(file_path)
        elif file_path.lower().endswith('.md'):
            import markdown
            from bs4 import BeautifulSoup
            logger.info(f"Extracting text from Markdown: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                html = markdown.markdown(f.read())
                return BeautifulSoup(html, "html.parser").get_text()
        elif file_path.lower().endswith('.txt'):
            logger.info(f"Extracting text from TXT: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        logger.warning(f"Unsupported file format for extraction: {file_path}")
        raise ValueError("Unsupported file format for text extraction.")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я бот для ответов на вопросы по вашим документам.",
    )
    await help_command(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет сообщение с описанием команд."""
    help_text = (
        "Я могу отвечать на вопросы на основе загруженных вами документов.\n\n"
        "<b>Доступные команды:</b>\n"
        "/start - Начало работы\n"
        "/help - Показать это сообщение\n"
        "/status - Показать статус QA системы и базы знаний\n\n"
        "<b>Как использовать:</b>\n"
        "1. Если у меня уже есть база знаний (проверьте командой /status), вы можете сразу задавать вопросы.\n"
        "2. Чтобы добавить новые знания, отправьте мне документ (PDF, TXT, MD). "
        "Я обработаю его и <b>дополню</b> существующую базу знаний.\n"
        "3. После обработки документа, задайте мне вопрос текстом.\n"
    )
    await update.message.reply_html(help_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статус QA системы."""
    status_msg_parts = []
    if qa_system:
        status_msg_parts.append("✅ QA система инициализирована и готова к работе.")
        if check_faiss_index_exists(FAISS_INDEX_PATH):
            status_msg_parts.append(f"📖 База знаний в '{FAISS_INDEX_PATH}' доступна и используется.")
        else:
            status_msg_parts.append(
                f"⚠️ База знаний в '{FAISS_INDEX_PATH}' НЕ найдена, хотя QA система активна (ошибка конфигурации?).")
    else:
        status_msg_parts.append("❌ QA система НЕ инициализирована.")
        if check_faiss_index_exists(FAISS_INDEX_PATH):
            status_msg_parts.append(
                f"ℹ️ Обнаружена база знаний в '{FAISS_INDEX_PATH}'. Попытка инициализации QA системы...")
            if initialize_qa_chain():
                status_msg_parts.append("✅ QA система успешно инициализирована!")
            else:
                status_msg_parts.append("❌ Не удалось инициализировать QA систему. Проверьте логи.")
        else:
            status_msg_parts.append(
                "ℹ️ База знаний не найдена. Пожалуйста, загрузите документ для создания базы знаний и активации системы.")

    await update.message.reply_text("\n".join(status_msg_parts))


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает загруженный документ и добавляет его в FAISS индекс."""
    if not update.message.document:
        await update.message.reply_text("Пожалуйста, отправьте документ.")
        return

    doc = update.message.document
    file_name = doc.file_name

    if not file_name.lower().endswith(('.pdf', '.md', '.txt')):
        await update.message.reply_text("Неподдерживаемый формат файла. Пожалуйста, загрузите PDF, MD или TXT.")
        return

    processing_message = await update.message.reply_text(f"Получен файл: {file_name}. Начинаю обработку...")

    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)

    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{doc.file_id}-{file_name}")

    try:
        tg_file: TelegramFile = await context.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(temp_file_path)
        logger.info(f"Файл сохранен: {temp_file_path}")

        await processing_message.edit_text(f"Файл: {file_name}\nИзвлекаю текст из документа...")
        content = extract_text_from_file(temp_file_path)

        if not content.strip():
            await processing_message.edit_text(f"Файл: {file_name}\nНе удалось извлечь текст или документ пуст.")
            return

        await processing_message.edit_text(f"Файл: {file_name}\nРазбиваю текст на части...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len
        )
        docs_chunks = text_splitter.split_text(content)

        if not docs_chunks:
            await processing_message.edit_text(
                f"Файл: {file_name}\nТекст не удалось разбить на части (слишком короткий?).")
            return

        await processing_message.edit_text(
            f"Файл: {file_name}\nОбновляю векторную базу знаний... Это может занять некоторое время."
        )

        current_embeddings = get_embeddings()
        db: Optional[FAISS] = None
        loop = asyncio.get_event_loop()

        if check_faiss_index_exists(FAISS_INDEX_PATH):
            try:
                logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH} to add new documents.")
                db = await loop.run_in_executor(
                    None,
                    FAISS.load_local,
                    FAISS_INDEX_PATH,
                    current_embeddings,
                    True
                )
                logger.info("Adding new texts to the existing FAISS index...")

                await loop.run_in_executor(None, db.add_texts, docs_chunks, current_embeddings.embed_documents)
                logger.info("New documents added to existing FAISS index.")
            except Exception as e:
                logger.error(f"Failed to load or update existing FAISS index: {e}. Creating a new one.", exc_info=True)
                await processing_message.edit_text(
                    f"Файл: {file_name}\nОшибка при загрузке/обновлении существующей базы. Создаю новую..."
                )
                db = await loop.run_in_executor(None, FAISS.from_texts, docs_chunks, current_embeddings)
        else:
            logger.info(f"FAISS index not found at {FAISS_INDEX_PATH}. Creating a new one.")
            db = await loop.run_in_executor(None, FAISS.from_texts, docs_chunks, current_embeddings)

        if db:
            if not os.path.exists(FAISS_INDEX_PATH):
                os.makedirs(FAISS_INDEX_PATH)
            logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
            await loop.run_in_executor(None, db.save_local, FAISS_INDEX_PATH)
            logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

            if initialize_qa_chain(force_rebuild_chain_only=True):
                await processing_message.edit_text(
                    f"Документ '{file_name}' успешно обработан и добавлен в базу знаний!\n"
                    "Теперь вы можете задавать по нему вопросы."
                )
            else:
                await processing_message.edit_text(
                    f"Файл: {file_name}\nДокумент обработан, но не удалось переинициализировать QA систему. Проверьте логи."
                )
        else:
            await processing_message.edit_text(f"Файл: {file_name}\nНе удалось создать или обновить базу данных FAISS.")

    except FileNotFoundError as e:
        logger.error(f"Ошибка FileNotFoundError (вероятно, модель LLM): {str(e)}")
        await processing_message.edit_text(
            f"Ошибка: {str(e)}. Убедитесь, что модель LLaMA скачана и путь к ней ({MODEL_PATH}) указан верно.")
    except Exception as e:
        logger.error(f"Ошибка при обработке документа '{file_name}': {str(e)}", exc_info=True)
        await processing_message.edit_text(f"Произошла ошибка при обработке документа '{file_name}': {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Временный файл удален: {temp_file_path}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовое сообщение как вопрос к QA системе."""
    global qa_system
    if not update.message or not update.message.text:
        return

    question = update.message.text
    logger.info(f"Получен вопрос от {update.effective_user.name}: {question}")

    if qa_system is None:
        if not initialize_qa_chain():
            await update.message.reply_text(
                "Система QA не готова. Пожалуйста, сначала загрузите документ, "
                "или проверьте /status, если база знаний должна быть доступна."
            )
            return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(None, qa_system.invoke, {"query": question})

        answer = result.get('result', "Не удалось получить ответ из QA системы.")
        source_documents = result.get('source_documents')

        logger.info(f"Ответ: {answer}")
        await update.message.reply_text(answer)

        if source_documents:
            sources_text_parts = ["\n\n<b>Источники (наиболее релевантные фрагменты):</b>"]
            for i, doc_source in enumerate(source_documents):
                content_preview = doc_source.page_content.replace('<', '<').replace('>', '>')
                source_info = f"{i + 1}. 📄 ...{content_preview[:150]}..."
                sources_text_parts.append(source_info)

            sources_full_text = "\n".join(sources_text_parts)
            if len(sources_full_text) > 4096:
                sources_full_text = sources_full_text[:4090] + "..."
            await update.message.reply_html(sources_full_text)

    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {str(e)}", exc_info=True)
        await update.message.reply_text(f"Произошла ошибка при обработке вашего вопроса: {str(e)}")


def main() -> None:
    """Запуск бота."""
    if not BOT_TOKEN or BOT_TOKEN == "TELEGRAM_BOT_TOKEN":
        logger.critical("Не указан токен бота (BOT_TOKEN). Бот не может быть запущен.")
        return

    get_embeddings()
    if initialize_qa_chain():
        logger.info("QA система успешно инициализирована при запуске с существующим индексом.")
    else:
        logger.warning("QA система не была инициализирована при запуске. "
                       "Проверьте наличие FAISS индекса или загрузите документы для его создания.")

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запускается...")
    application.run_polling()


if __name__ == "__main__":
    main()