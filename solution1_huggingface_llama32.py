from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import fitz
import pandas as pd
import os
import torch

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

text= convert_pdf_to_text("./docs/Korea-Follow-Up-Report-2024.pdf")
questions, df = load_questions("./docs/korea_aml_questions_all.csv")

model_id= "meta-llama/Llama-3.2-1B"
pipe= pipeline("text-generation", model=model_id, torch_dtype=torch.float16, device_map="auto")

output_file = "./results/solution1_huggingface_llama32/aa.csv"

# csv for results
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("question_id;question;answer\n")

for idx, q in enumerate(questions, start=1):
    QA_input = f"'question': {q} , context': {text}"
    print("***********")
    print(QA_input)
    with open("test.txt", "a") as f:
        f.write(QA_input)



    res =  pipe(QA_input, max_length=9000, do_sample=True, temperature=0.2, top_k=50, top_p=0.9)

    print(res)
    
    with open(output_file, "a") as f:
        f.write(f"{idx};{q};{res}\n")


# NB: for testing I ran the command:
# torchrun --nproc_per_node=1 solution1_huggingface_llama32.py 