# sentence_ordering
code for sentence ordering

### Requirements
python 3.6.5, pytorch 1.8.0

Please download pytorch_model.bin for bart-large from Huggingface/transformers (https://github.com/huggingface/transformers), and put it into facebook/bart_large/.

### Train and evaluate
change the --data_dir and --fea_data_dir in run_berson_bart.sh.
```
bash run_berson_bart.sh
```

### Test
```
bash run_test.sh
```
