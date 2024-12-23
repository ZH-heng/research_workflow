# Automatic Generation of Research Workflow in Natural Language Processing: A Full-text Mining Approach

## Overview
Dataset and source code for paper "Automatic Generation of Research Workflow in Natural Language Processing: A Full-text Mining Approach".

## Directory structure
<pre>
research_workflow                              Root directory
├── PU-Learning.ipynb                          Source code for PU Learning to obtain reliable negative samples
├── paragraph_clf.py                           Source code for research workflow paragraph identification based on pre-trained models
├── workflow_phrase_gen.py                     Source code for workflow phrase generation using pre-trained models
├── workflow_phrase_gen_prompt.py              Source code for workflow phrase generation using pre-trained models with prompt learning
├── phrase_clf_chatgpt.py                      Source code for workflow phrase classification using ChatGPT
├── data                                       Dataset folder
│   ├── workflow_phrases.txt                   197,304 research workflow phrases
│   └── paper_phrase.parquet                   The workflows in three research stage of  NLP papers
└── README.md
</pre>