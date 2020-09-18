# Masked Pun Detector Trained with BERT

This is a repository of a simple pun detector trained on the BERT[[1]](#1) architecture.  

## Data
For fine-tuning the pretrained BERT embeddings, I use the Pun of the Day 
dataset[[2]](#2) specifically curated for humor recognition by Weller and Seppi (2019)
[[3]](#3). The dataset cosists of 2459 one-liner puns and 2403 non-pun sentences.
I split the dataset in a training-development ratio of 5:1. Additionally, I create 
an augmented fine-tuning dataset by making <em>n</em> copies of each training example, where <em>n</em> is the number of tokens in each example, 
and the <em>i</em>-th token of the <em>i</em>-th copy is replaced by a special token \<MASK\>.
To evaluate the finetuned model, I use the heterographic pun detection dataset of SemEval
2017[[4]](#4).

The data can be viewed in the `data` directory. The `base` and `masked` subdirectories
correspond to the training data with and without \<MASK\>. Both subdirectories contain
`.tsv` files of train, development and test data with an additional set of toy 
examples in `trial.tsv`. All `.tsv` files follow the `ID | Pun | Label` column layout.

## Model
The standard BERT recipe involves pretraining and fine-tuning. For the pun 
detection task, I use the pretrained uncased BERT base embeddings, and fine-tune 
the embeddings on the Pun of the Day dataset[[2]](#2).

### Fine-tuning
I compare two fine-tuning methods in my experiment. The first model is finetuned 
on the unmodified SemEval 2017 heterographic dataset. The second model is finetuned 
on the same dataset but with \<MASK\>. I get the inspiration from the masked finetuning
from the fact that humans are able to laugh at puns with slight paraphrasing, or even
when they do not hear every word in the pun clearly. Masked finetuning is a simple
way to simulate such variation in puns. In the future, I would like to apply more
types of variations in the fine-tuning process.

The fine-tuning code is based on the BERT repository[[5]](#5). To fine-tune the 
baseline model, run 
```shell
python run_classifier.py \
  --task_name=pun \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/base \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/pun_output/
```

The code to fine-tune the masked model is:
```shell
python run_classifier.py \
  --task_name=pun \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/masked \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/pun_output/
```

To test the models, use the following code:
```shell
python run_classifier.py \
  --task_name=pun \
  --do_predict=true \
  --data_dir=$DATA_DIR/base \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/pun_output/
```

## Results
On the 1780 testing examples, baseline fine-tuning achieves an accuracy of 0.82,
whereas masked fine-tuning achieves an accuracy of 0.86. This shows the masked 
dataset is effective in fine-tuning. However, further work is needed 
to investigate how much improvement on accuracy is due to masked fine-tuning as 
opposed the the addition of duplicate training instances.

Full prediction results can be viewed in `results` directory. Each subdirectory 
presents results from the two models. The raw results are 
in `test_results.tsv`. By running the code in `compare_results.py`, I get the 
the organized results `full_results.tsv` and the wrong predictions `wrong_pred.tsv`.

## Distributing the Repository
This repository is for non-commercial use only. If you want to contribute to the 
repository, please contact qiweis2015@gmail.com.

If you use the model or results in this repository in your work, please cite the 
link to this repository along with the author's name: Qiwei Shao.

## References
<a id="1">[1]</a> 
Devlin et al (2018). 
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
<em>arXiv preprint arXiv:1810.04805</em>

<a id="2">[2]</a> 
Yang et al (2015).
Humor recognition and humor anchor extraction.
In <em>Proceedings of the 2015 Conference on Empirical Methods in Natural Language 
Processing,</em> pages 2367–2376.

<a id="3">[3]</a>
Weller and Seppi (2019).
Humor detection: A transformer gets the last laugh.
In <em>Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint
Conference on Natural Language Processing (EMNLPIJCNLP)</em>, pages 3612–3616.

<a id="4">[4]</a>
Miller, Hempelmann and Gurevych (2017).
SemEval-2017 Task 7: Detection and Interpretation of English Puns.
In <em>Proceedings of the 11th International Workshop
on Semantic Evaluation (SemEval-2017)</em>, pages
58–68, Vancouver, Canada. Association for Computational
Linguistics.

<a id="5">[5]</a>
https://github.com/google-research/bert
