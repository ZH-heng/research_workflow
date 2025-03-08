# Automatic Generation of Research Workflow from Academic Papers: A Full-text Mining Approach

## Overview
Dataset and source code for paper "Automatic Generation of Research Workflow from Academic Papers: A Full-text Mining Approach".

## Directory structure
<pre>
research_workflow                         Root directory
├── code                                  Code folder
│   ├── PU-Learning.ipynb                 Source code for PU Learning to obtain reliable negative samples
│   ├── paragraph_clf.py                  Source code for research workflow paragraph identification based on pre-trained models
│   ├── workflow_phrase_gen.py            Source code for workflow phrase generation using pre-trained models
│   ├── workflow_phrase_gen_prompt.py     Source code for workflow phrase generation using pre-trained models with prompt learning
│   ├── phrase_clf_chatgpt.py             Source code for workflow phrase classification using ChatGPT
├── data                                  Dataset folder
│   ├── workflow_phrases.txt              197,304 research workflow phrases
│   └── paper_phrase.parquet              The workflows in three research stage of NLP papers
├── Annotation_Guidelines.docx            Annotation Guidelines for Research Workflow
└── README.md
</pre>

## Dataset Discription
  - <code>./data/workflow_phrases.txt</code> After research workflow paragraph identification, workflow phrase generation, and workflow phrase classification, a total of 227,814 workflow phrases were obtained. Subsequently, based on lemmatization, the number of phrases was reduced to 197,304.
  - <code>./data/paper_phrase.parquet</code> Organize research workflow phrases of each paper according to three stages: data preparation, data processing, and data analysis. Within each stage, the order of workflow phrases should correspond to their position within the respective paragraphs of the paper.

## Quick Start
 - <b>PU Learning</b>
   - <code>./code/PU-Learning.ipynb</code> Execute the program step by step, utilize the Spy technique in PU Learning to obtain reliable negative samples.
 - <b>Research workflow paragraph identification</b>
   - <code>python ./code/paragraph_clf.py</code> Run paragraph_clf.py to classify research workflow paragraphs based on pre-trained models, including SciBERT, BERT and RoBERT.
 - <b>Research workflow phrase generation</b>
   - <code>python ./code/workflow_phrase_gen.py</code> Run workflow_phrase_gen.py to generate research workflow phrases based on pre-trained models, including BART, MVP, PEGASUS, T5, and Flan-T5.
   - <code>python ./code/workflow_phrase_gen_prompt.py</code> To further enhance the effectiveness of workflow phrase generation, we integrate prompt learning into Flan-T5.
 - <b>Research workflow phrase classification</b>
   - <code>python ./code/phrase_clf_chatgpt.py</code> Run phrase_clf_chatgpt.py to categorize workflow phrases into data preparation, data processing, and data analysis, using ChatGPT with few-shot learning.

## Dependency packages
- pytorch=2.0.1
- transformers=4.28.1
- pandas=2.0.0
- tqdm=4.65.0
- loguru=0.7.0
- fasttext=0.9.2
- numpy=1.24.1

