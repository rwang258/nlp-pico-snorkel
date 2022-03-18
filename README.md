# nlp-pico-snorkel

**PIO_processing** contains code used by Bojian to generate the input datasets.

**archive** and **archive2** contains old code.

**lf_datasets** and **nltk_data** contains datasets used in the labeling functions with Snorkel.

**error.ipynb** contains the code used to generate the Snorkel error files.

**error** contains the error files. 
- **combined** has the final error files. Tokens that are gold labeled as intervention are bolded, tokens that are missed (tokens gold labeled as intervention, but were predicted as not intervention or abstained on) are italicized and bolded. Tokens that are gold-labeled as not intervention but predicted by Snorkel as intervention are crossed out (mislabeled).

- **gold**, **missed*, and **wrong** contain specific types of error files. Some background: Snorkel predicts 3 labels. A label of 1 means intervention, 0 means not intervention, -1 means abstain.

  - The folder 'gold' has all tokens labeled as intervention, highlighted. The folder 'missed' has all tokens that are missed highlighted (meaning tokens that are labeled as 1, but predicted as 0 or -1). The folder 'wrong' has mislabeled tokens highlighted (meaning tokens labeled as 0 but predicted as 1, and tokens labeled as 1 but predicted as 0).

**df.ipynb** and **df.py** contains the code used to generate the input to Snorkel and BERT.

**snorkel_pico.ipynb** and **snorkel_pico.py** is using Snorkel on the PICO dataset.

**bert.ipynb** is the baseline BERT supervised learning model on the same intervention labeling task with the same dataset.

Not here, but will be linked with Box soon:
**pico_datasets** contains the original abstracts in the PICO paper.
**df_orig.pickle**, **df_train.pickle**, **df_test.pickle** contains the pickel objects used to store the input to Snorkel and BERT.
**first_sentence_file.pickle** contains the first sentence of every abstract.

Temporary files:
**bert_gold.txt**, **bert_input.txt**, **bert_preds.txt** are the inputs, gold labels, and predictions for a sample abstract from BERT. 
