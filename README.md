# Korea Compliance Report - Question Answering  

This repository contains code and tools for extracting information from the **Korea Mutual Evaluation Report** (PDF) and answering specific questions using different AI models. The extracted answers are saved in a CSV file for further analysis.  

## Project Overview  

The goal of this project is to:  
✅ Convert a **PDF report** into text  
✅ Read a set of **questions** from a CSV file  
✅ Answer each question using **AI models**  
✅ Store results in a structured **CSV file**  

## Workflow  

1. **Convert PDF to Text**  
   - The PDF is processed and converted into text using `PyMuPDF` (fitz).  

2. **Load Questions**  
   - The questions are read from a CSV file and formatted for processing.  

3. **Answer Questions Using AI Models**  
   - **Solution 1:** Uses Hugging Face’s `Intel/dynamic_tinybert` & `deepset/roberta-base-squad2` models for direct question-answering.  
   - **Solution 2 (RAG-based):** Implements **Retrieval-Augmented Generation (RAG)** using:  
     - `HuggingFaceEmbeddings (sentence-transformers/all-mpnet-base-v2)` for vector representation  
     - `LLaMA 3 (llama3-8b-8192)` as the language model  
     - `LangChain` for document retrieval and question answering  

     🔹 **Why RAG?**  
     - RAG is **more efficient** than traditional models because it **indexes and stores knowledge** in a structured way.  
     - Instead of processing the entire document every time, it retrieves **only relevant context**, reducing computation and improving accuracy.  

4. **Store Answers**  
   - Answers are saved in a **CSV file** with the format:  
     ```
     question_id;question;answer
     ```  

## Setup & Installation  

### 1️⃣ Install Dependencies  
Run the following command to install required Python libraries:  
```bash
pip install -r requirements.txt
```  

### 2️⃣ Initialize Environment  
Run the initialization script:  
```bash
./init.sh
```  

### 3️⃣ Set API Keys (If Required)  
Ensure API keys for **LangSmith** and **GROQ** are set:  
```bash
export LANGSMITH_API_KEY="your_api_key"
export GROQ_API_KEY="your_api_key"
```  

## 📂 Project Structure  

```
📁 docs/  
   ├── Korea-Follow-Up-Report-2024.pdf  
   ├── korea_aml_questions_all.csv  
📁 results/  
   ├── solution1_huggingface_intel/answers.csv  
   ├── solution1_huggingface_roberta/answers.csv  
   ├── solution2_RAG/answers.csv  
📜 solution1_huggingface_intel.py  
📜 solution2_RAG.py  
📜 requirements.txt  
📜 README.md  
```  

## Models Used  

| Model | Description | Source |
|-------|------------|--------|
| **Intel/dynamic_tinybert** | Optimized for question answering | [Intel TinyBERT](https://huggingface.co/Intel/dynamic_tinybert) |
| **deepset/roberta-base-squad2** | Fine-tuned for SQuAD 2.0 QA tasks | [RoBERTa SQuAD2](https://huggingface.co/deepset/roberta-base-squad2) |
| **sentence-transformers/all-mpnet-base-v2** | Embedding model for semantic search | [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| **Meta LLaMA 3 (llama3-8b-8192)** | Large language model for RAG-based QA | [LLaMA 3](https://huggingface.co/meta-llama/Llama-3.2-1B) |

## Why Use RAG?  

Retrieval-Augmented Generation (RAG) **enhances efficiency** in document-based question answering:  
✅ **Better Storage & Indexing** – Converts documents into a structured vector database for efficient lookup.  
✅ **Faster Response Time** – Retrieves only **relevant** information, reducing the need to process the entire document.  
✅ **Higher Accuracy** – Provides context-aware answers, avoiding hallucinations common in generative models.  

## Additional Resources  

- 🔗 [LangChain Documentation](https://python.langchain.com/)  
- 🔗 [Retrieval-Augmented Generation (RAG)](https://huggingface.co/blog/rag)  
- 🔗 [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  

---

**Contributions Welcome!**  Feel free to open issues & pull requests!  

