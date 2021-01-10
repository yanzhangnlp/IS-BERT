"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
import sys
import os
import torch
import numpy as np

script_folder_path = os.path.dirname(os.path.realpath(__file__))

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = '../training/nli/output/training_nli_bert-base-uncased-2021-01-10_14-44-13'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)

sts_corpus = "../datasets/stsbenchmark/" 
target_eval_files = set(['sts','sts12', 'sts13', 'sts14', 'sts15', 'sts16', 'sick-r']) 

evaluators = []         #evaluators has a list of different evaluator classes we call periodically
sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, sts_corpus))
for target in target_eval_files:
	output_filename_eval = os.path.join(script_folder_path,sts_corpus + target + "-test.csv")
	evaluators.append(EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), name=target))

evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores))
model.evaluate(evaluator)
