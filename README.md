# Korea Compliance Report - Question Answering

This repository contains code and tools for extracting information from a PDF document (Korea’s Mutual Evaluation Report) and answering specific questions using various models. The answers are extracted based on the provided context in the report and saved to a CSV file for analysis.

## Overview

The goal of this project is to convert a PDF report into text, read a set of questions from a CSV file, and then answer each question based on the context of the document using different AI models.

### Steps

1. **Convert PDF to Text**  
   The PDF is processed and converted into text using the `PyMuPDF` (fitz) library.
   
2. **Load Questions**  
   The questions are read from a CSV file, which are used to query the document.

3. **Answer Questions Using Models**  
   Two different approaches were used for question answering:
   - **Solution 1**: Huggingface `deepset/roberta-base-squad2` model.
   - **Solution 2**: Huggingface `meta-llama/Llama-3.2-1B` model.
   - **Solution3**: Huggingface `Intel/dynamic_tinybert` model.
   
4. **Store Answers**  
   The answers to each question are stored in a CSV file with the format:  
   `question_id;question;answer`

## Requirements

- Python 3.x
- PyMuPDF (fitz)
- Transformers
- Pandas
- Torch

## Usage

1. **Convert PDF to Text**  
   Use the `convert_pdf_to_text()` function to extract text from the PDF.

2. **Load Questions**  
   Use the `load_questions()` function to read questions from a CSV file.

3. **Answer Questions**  
   For each question, pass it along with the extracted text context to the chosen model using the Huggingface `pipeline`.

4. **Save Results**  
   Answers are written to a CSV file.

## Possible Improvements

- Experiment with other advanced models for better accuracy.
- Convert the PDF to a FAISS vector database for improved search and answer extraction.

## Useful Ressources

https://huggingface.co/deepset/roberta-base-squad2

https://huggingface.co/meta-llama/Llama-3.2-1B


https://huggingface.co/Intel/dynamic_tinybert
