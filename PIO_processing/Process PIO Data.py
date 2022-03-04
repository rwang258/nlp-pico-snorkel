from glob import glob
import os, sys
import pickle
from transformers import AutoTokenizer

TOP = './ebm_nlp_1_00/'

DATA_PKL = 'PIO_data.pkl'

print('Reading documents...')

tokens_list = []
abstract_list = []
label_list = []
pmid_list = []
data = {}

doc_fnames = glob('%s/documents/*.tokens' %(TOP))
label_fnames = glob('%s/annotations/aggregated/starting_spans/*/*_AGGREGATED.ann' %(TOP))

count = 0
for i, fname in enumerate(doc_fnames):
  pmid_doc = os.path.basename(fname).split('.')[0]
  pmid_label_intervention = os.path.basename(label_fnames[i]).split("_")[0]
  pmid_label_outcome = os.path.basename(label_fnames[i+4993]).split("_")[0]
  pmid_label_participants = os.path.basename(label_fnames[i+4993*2]).split("_")[0]
  if pmid_doc == pmid_label_intervention == pmid_label_outcome == pmid_label_participants:
    count = count + 1  # to check whether the labels and document are aligned, if the count equals 4993, it will work

    intervention_label = open(label_fnames[i]).read().split(',')
    # new_intervention_label = ["Intervention" if label == "1" else label for label in intervention_label]
    outcome_label = open(label_fnames[i+4993]).read().split(',')
    # new_outcome_label = ["Outcome" if label == "1" else label for label in outcome_label]
    participants_label = open(label_fnames[i+4993*2]).read().split(',')
    # new_participants_label = ["Participants" if label == "1" else label for label in participants_label]
    current_label = ["Int" if i=="1" else "Out" if o=="1" else "Pop" if p=="1" else "0" for
    i,o,p in zip(intervention_label,outcome_label,participants_label)]
    label_list.append(current_label)

    tokens = open(fname).read().split()
    abstract = open(fname.replace('tokens', 'text')).read()

    tokens_list.append(tokens)
    abstract_list.append(abstract)
    pmid_list.append(pmid_doc)
  # else:
  #   print("{} {} {} {} {}".format(i,pmid_doc,pmid_label_intervention,pmid_label_outcome,pmid_label_participants))

  if (i//100 != (i-1)//100):
    sys.stdout.write('\r\tprocessed %04d / %04d docs\n' %(i, len(doc_fnames)))
    sys.stdout.flush()

  #  class_name = label_fnames[i].split("\\")[1]  # e.g., label_fnames[0].split("\\") is
  # ['./ebm_nlp_1_00//annotations/aggregated/starting_spans',
  # 'interventions',
  # '10036953_AGGREGATED.ann']

print(count)

  # tokens = open(fname).read().split()
  # docs_dict[pmid] = {}
  # docs_dict[pmid]['tokens'] = tokens
  # docs_list.append(tokens)
  #
  #
  # if (i//100 != (i-1)//100):
  #   sys.stdout.write('\r\tprocessed %04d / %04d docs' %(i, len(doc_fnames)))
  #   sys.stdout.flush()

data["token"]=tokens_list
data['abstract']=abstract_list
data["label"]=label_list
data['pmid']=pmid_list
with open(DATA_PKL, 'wb') as fout:
  print('\nWriting data to %s' %DATA_PKL)
  pickle.dump(data, fout)

