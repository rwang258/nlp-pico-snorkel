from glob import glob
import os, sys
import pickle
import json

TOP = './ebm_nlp_1_00/'
DATA_PKL = 'PIO_data_test.pkl'
DATA_JSON = 'PIO_data_test.json'
print('Reading documents...')

data_list = []

doc_fnames = glob('%s/documents/*.tokens' %(TOP))
label_fnames_list = []
label_fnames_list.append(glob('%s/annotations/individual/starting_spans/*/test/gold/*_00000.ann' %(TOP)))
label_fnames_list.append(glob('%s/annotations/individual/starting_spans/*/test/gold/*_00002.ann' %(TOP)))
label_fnames_list.append(glob('%s/annotations/individual/starting_spans/*/test/gold/*_00003.ann' %(TOP)))

for label_fnames in label_fnames_list:  # create test dataset for each professional (3 in total)
    count = 0
    data = {}
    docs_list = []
    label_list = []
    for i, fname in enumerate(label_fnames):
        # pmid_doc = os.path.basename(fname).split('.')[0]
        if i == 191:  # only go through the first 191 items (the left are went through in advance)
            break
        pmid_label_intervention = os.path.basename(label_fnames[i]).split("_")[0]
        pmid_label_outcome = os.path.basename(label_fnames[i + 191]).split("_")[0]
        pmid_label_participants = os.path.basename(label_fnames[i + 191 * 2]).split("_")[0]
        if pmid_label_intervention == pmid_label_outcome == pmid_label_participants:
            count = count + 1  # to check whether the labels and document are aligned, if the count equals 191, it will work

            intervention_label = open(label_fnames[i]).read().split(',')
            # new_intervention_label = ["Intervention" if label == "1" else label for label in intervention_label]
            outcome_label = open(label_fnames[i + 191]).read().split(',')
            # new_outcome_label = ["Outcome" if label == "1" else label for label in outcome_label]
            participants_label = open(label_fnames[i + 191 * 2]).read().split(',')
            # new_participants_label = ["Participants" if label == "1" else label for label in participants_label]
            current_label = ["Int" if i == "1" else "Out" if o == "1" else "Pop" if p == "1" else "0" for
                             i, o, p in zip(intervention_label, outcome_label, participants_label)]
            label_list.append(current_label)
            tokens = open("./ebm_nlp_1_00/documents/{}.tokens".format(pmid_label_intervention)).read().split()
            docs_list.append(tokens)
    print(count)
    data["doc"]=docs_list
    data["label"]=label_list
    data_list.append(data)
with open(DATA_JSON, 'w') as fout:
  print('\nWriting data to %s' %DATA_JSON)
  json.dump(data_list, fout)