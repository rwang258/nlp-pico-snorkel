from glob import glob
import os, sys
import json
import pickle
from transformers import AutoTokenizer

TOP = './ebm_nlp_1_00/'

# DATA_PKL = 'PIO_data.pkl'
# DATA_PKL = 'PIO_data_abstracts.pkl'
DATA_PKL = 'PIO_data_PMID_abstracts.json'

print('Reading documents...')

docs_list = []

# doc_fnames = glob('%s/documents/*.tokens' %(TOP))
doc_fnames = glob('%s/documents/*.text' %(TOP))

count = 0
for i, fname in enumerate(doc_fnames):
  pmid_doc = os.path.basename(fname).split('.')[0]
  abstracts = open(fname).read()
  current_data = [pmid_doc, abstracts]
  docs_list.append(current_data)

  if (i//100 != (i-1)//100):
    sys.stdout.write('\r\tprocessed %04d / %04d docs\n' %(i, len(doc_fnames)))
    sys.stdout.flush()

with open(DATA_PKL, 'w') as fout:
  print('\nWriting data to %s' %DATA_PKL)
  json.dump(docs_list, fout)

