# Text Summarizer using T5 Transformer

An end-to-end NLP application that generates concise summaries from long text and conversations using a fine-tuned T5-small Transformer model. The application combines Hugging Face Transformers, PyTorch, FastAPI, and a simple web interface.

## Features

- Automatic text and dialogue summarization.
- Fine-tuning of a pretrained T5-small model on the SAMSum dataset.
- Text preprocessing and tokenization using the T5 tokenizer.
- Sequence generation using beam search.
- FastAPI backend for model inference.
- Interactive web interface built with HTML, CSS, and JavaScript.
- CPU and CUDA device selection when supported.

## Tech Stack

- **Language:** Python
- **NLP / Deep Learning:** Hugging Face Transformers, PyTorch
- **Model:** T5-small
- **Dataset:** SAMSum
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Data Processing:** Pandas, Regular Expressions

## Project Structure

```text
text-summarizer/
├── app.py
├── index.html
├── text_summarizer.ipynb
├── saved_summary_model/
│   ├── config.json
│   ├── model weights
│   ├── tokenizer files
│   └── other model configuration files
├── samsum-train.csv
├── samsum-validation.csv
├── requirements.txt
└── README.md
```

The model directory and dataset files must be supplied separately if they are not included in the repository. The notebook contains the training workflow.

## Model Training

The notebook follows these steps:

1. Load the SAMSum training and validation datasets.
2. Sample 4,000 training examples and 500 validation examples.
3. Clean and normalize the dialogue and summary text.
4. Tokenize inputs and target summaries.
5. Fine-tune the pretrained T5-small model using the Hugging Face Trainer API.
6. Save the fine-tuned model and tokenizer to `saved_summary_model/`.

The notebook configuration uses six training epochs and a batch size of eight per device.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ashutosh101026/Text-Summarizer
cd text-summarizer
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install fastapi uvicorn transformers torch sentencepiece jinja2 pandas
```

Place the trained model files in `saved_summary_model/` and ensure that the dataset CSV files are available before running the training notebook.

## Run the Application

Start the FastAPI server from the project root:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

Enter text in the input box and click **Summarize** to generate a summary.

The API endpoint is:

```text
POST /summarize/
```

Example request:

```json
{
  "dialogue": "Enter the text or conversation you want to summarize."
}
```

Example response structure:

```json
{
  "summary": "The generated summary will appear here."
}
```

## Future Improvements

- Evaluate the model using ROUGE metrics on a held-out test set.
- Add input length validation and more robust error handling.
- Improve the interface with copy-summary and clear-text options.
- Deploy the application for public access.
- Explore model optimization to reduce inference latency.

## Learning Outcomes

This project demonstrates practical experience with Transformer-based NLP, model fine-tuning, text preprocessing, sequence generation, REST API development, and frontend-backend integration.
