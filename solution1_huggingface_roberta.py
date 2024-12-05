from transformers import pipeline
import fitz
import pandas as pd
import os


def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def load_questions(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', delimiter=';')
    questions = df['question'].tolist() 
    return questions, df



model_name = "deepset/roberta-base-squad2"
text= convert_pdf_to_text("./docs/Korea-Follow-Up-Report-2024.pdf")
# print(text)
questions, df = load_questions("./docs/korea_aml_questions_all.csv")
# print(questions)

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)

output_file = "./results/solution1_huggingface_roberta/answers.csv"

# csv for results
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("question_id;question;answer\n")

for idx, q in enumerate(questions, start=1):
    QA_input = {
        'question': q,
        'context': text
    }
    res = nlp(QA_input)
    
    with open(output_file, "a") as f:
        f.write(f"{idx};{q};{res['answer']}\n")
