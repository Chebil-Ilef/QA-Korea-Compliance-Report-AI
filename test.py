import torch
from transformers import pipeline


torch.cuda.empty_cache()

model_id= "meta-llama/Llama-3.2-1B"


pipe= pipeline("text-generation", model=model_id, torch_dtype=torch.float16, device_map="auto")

with open("test.txt", "r") as f:
    QA_input = f.read()

QA_input= "'question': 'What is the purpose of the report?', 'context': 'The Financial Action Task Force (FATF) is an intergovernmental organization founded in 1989 on the initiative of the G7 to develop policies to combat money laundering. In 2001 its mandate expanded"
result = pipe(QA_input, max_new_tokens=1000, do_sample=True, temperature=0.2, top_k=50, top_p=0.9)


print(result[0])