from glob import glob
import os, sys
import pickle
import json

DATA_PKL = 'PIO_data_test.pkl'
DATA_JSON = 'PIO_data_PMID_abstracts_test.json'
print('Reading documents...')

data_list = []

doc_fnames = glob('documents/*.txt')
label_fnames_list = []
label_fnames_list.append(glob('annotations/individual/starting_spans/*/test/gold/*_00000.ann'))
label_fnames_list.append(glob('annotations/individual/starting_spans/*/test/gold/*_00002.ann'))
label_fnames_list.append(glob('annotations/individual/starting_spans/*/test/gold/*_00003.ann'))

for label_fnames in label_fnames_list:  # create test dataset for each professional (3 in total)
    count = 0
    docs_list = []
    for i, fname in enumerate(label_fnames):
        if i == 191:  # only go through the first 191 items (the left are went through in advance)
            break
        pmid_label_intervention = os.path.basename(label_fnames[i]).split("_")[0]
        pmid_label_outcome = os.path.basename(label_fnames[i + 191]).split("_")[0]
        pmid_label_participants = os.path.basename(label_fnames[i + 191 * 2]).split("_")[0]
        if pmid_label_intervention == pmid_label_outcome == pmid_label_participants:
            count = count + 1  # to check whether the labels and document are aligned, if the count equals 191, it will work
            abstract = open("documents/{}.txt".format(pmid_label_intervention)).read()
            current_data = [pmid_label_intervention, abstract]
            docs_list.append(current_data)
    print(count)
    data_list.append(docs_list)

with open(DATA_JSON, 'w') as fout:
  print('\nWriting data to %s' %DATA_JSON)
  json.dump(data_list, fout)