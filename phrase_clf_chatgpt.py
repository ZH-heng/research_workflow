import json
from openai import OpenAI
from pathlib import Path
import multiprocessing

client = OpenAI(base_url="xxx", api_key="xxx")
def chat(prompt):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "system", "content": "You are an NLP expert."},
                    {"role": "user", "content": prompt}],
        temperature = 0
    )
    return dict(response.choices[0].message)

def worker(path):
    # Each process handles one file, processing 10 sentences at a time, and saves the results to the "res" folder.
    path = Path(path)
    save_path = f"./res/{path.name}"
    if Path(save_path).exists():
        with open(save_path, encoding="utf8") as f:
            lines = f.read().strip().split("\n")
        start_idx = int(lines[-1].split("\t")[0].split("-")[1])
    else:
        start_idx = 0
    with open(path, encoding="utf8") as f:
        phrases = f.read().split("\n")
    for i in range(start_idx, len(phrases), 10):
        phrases_10 = [f"{_} //" for _ in phrases[i:i+10]]
        prompt = f"""The research workflow for solving scientific problems is divided into three stages: Data Preparation, Data Processing, and Data Analysis.
First, try to understand the distinctions among these three stages based on your existing knowledge and the following examples:
datasets and preprocessing // data_preparation
template generation // data_processing
case study // data_analysis
corpus and annotation // data_preparation
model training // data_processing
correlation analysis // data_analysis
corpus statistics // data_preparation
data augmentation // data_ processing
effect of word selection // data_analysis
data split // data_preparation
span matching // data_processing
comparison with baselines // data_analysis
data collection // data_preparation
paragraph retrieval // data_processing
human evaluation // data_analysis

Then, answer the stages described by the following phrases:
{"\n".join(phrases_10)}

Please follow my input order and respond in the following format, without any additional explanation.
original phrase // label"""
        resp = chat(prompt)
        data = f"{i}-{i+10}\t{json.dumps(resp)}"
        with open(save_path, "a", encoding="utf8") as f:
            f.write(data+"\n")
        print(f"process: {path.stem}  complete: {i+10}/{len(phrases)}")

def main():
    # The research workflow phrases are saved into 5 txt files, and processed using 5 processes.
    files = Path("./files").glob("*.txt")
    with multiprocessing.Pool(processes=5) as pool:
        pool.map(worker, files)

if __name__ == "__main__":
    main()
