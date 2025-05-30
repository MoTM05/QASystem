# 📚💬 Система Q&A с документами на основе LLM (Llama 2) и RAG

Добро пожаловать в проект системы вопросов и ответов (Q&A), которая позволяет задавать вопросы к вашим документам (PDF, TXT, MD) и получать ответы на основе мощной языковой модели Llama 2, 
используя архитектуру Retrieval-Augmented Generation (RAG). Проект предоставляет два интерфейса: Telegram-бот и RESTful API на FastAPI.

## ✨ Особенности

*   **Мультимодальные интерфейсы:** Взаимодействуйте с системой через удобный Telegram-бот или интегрируйте её в свои приложения с помощью RESTful API.
*   **Обработка документов:** Загружайте документы в форматах PDF, TXT и Markdown для извлечения информации.
*   **Retrieval-Augmented Generation (RAG):** Система использует векторную базу данных FAISS для поиска наиболее релевантных фрагментов текста из ваших документов, а затем передает их языковой модели Llama 2 для генерации точных и контекстуально релевантных ответов.
*   **Локальная LLM:** Использование локально развернутой модели Llama 2 (через `llama-cpp-python`) обеспечивает конфиденциальность данных и не требует сторонних API-ключей.
*   **Масштабируемость:** Архитектура позволяет легко добавлять новые документы, дополняя существующую базу знаний.

## 🛠️ Технологии

*   **Python 3.12**
*   **Telegram Bot API** (через `python-telegram-bot`)
*   **FastAPI:** Для создания высокопроизводительного API.
*   **LangChain:** Фреймворк для разработки приложений на основе LLM.
*   **FAISS:** Библиотека для эффективного поиска по векторным представлениям.
*   **HuggingFace Embeddings:** Для создания векторных представлений текста.
*   **llama-cpp-python:** Python-биндинги для `llama.cpp` для запуска моделей Llama 2.
*   **pdfminer.six, markdown, beautifulsoup4:** Для извлечения текста из документов.

## 🚀 Быстрый старт

Следуйте этим инструкциям, чтобы запустить проект локально.
### 📦 Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/MoTM05/QASystem.git
    cd QASystem
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv venv
    # Для Windows:
    .\venv\Scripts\activate
    # Для macOS/Linux:
    source venv/bin/activate
    ```

3.  **Установите зависимости Python:**
    ```bash
    pip install -r requirements.txt
    ```
    *Примечание: Если возникнут проблемы с `llama-cpp-python`, возможно, потребуется установить инструменты сборки C++ (например, Build Tools для Visual Studio на Windows или `build-essential` на Linux).*

    ### ⬇️ Загрузка больших моделей

Из-за большого размера модель Llama 2 не хранится в репозитории. Вам необходимо скачать её вручную:

1.  **Скачайте модель Llama 2 GGUF:**
    Перейдите по ссылке на Hugging Face Hub, например:
    [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf)
    Скачайте файл `llama-2-7b-chat.Q4_K_M.gguf`.

2.  **Поместите модель в правильную директорию:**
    Создайте папку `models` в корне вашего проекта, если ее нет:
    ```bash
    mkdir models
    ```
    Переместите скачанный файл `llama-2-7b-chat.Q4_K_M.gguf` в эту папку `models`.

    Ваша структура должна выглядеть так:
    ```
    folder/
    ├── models/
    │   └── llama-2-7b-chat.Q4_K_M.gguf
    └── ...
    ```
### 🏃 Запуск

Вы можете запустить оба приложения одновременно или по отдельности.

#### Запуск Telegram-бота

В новом терминале, находясь в корне проекта (и активировав виртуальное окружение):
```bash
python bot.py
```
#### Запуск API

В новом терминале, находясь в корне проекта (и активировав виртуальное окружение):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API будет доступно по адресу http://127.0.0.1:8000. Вы можете перейти по адресу http://127.0.0.1:8000/docs для доступа к интерактивной документации Swagger UI.
