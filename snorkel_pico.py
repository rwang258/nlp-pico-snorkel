# Importations.
import pandas as pd
import numpy as np
import snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.analysis import get_label_buckets
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re
import string # For punctuation.
import os
import random
import pickle


with open('first_sentence_file.pickle', 'rb') as handle:
    first_sentence_file = pickle.load(handle)
    
# Read in suffixes.
# https://druginfo.nlm.nih.gov/drugportal/jsp/drugportal/DrugNameGenericStems.jsp
df_drugs = pd.read_csv("suffixes/drug_suffixes.txt", header = None)
df_surgery = pd.read_csv("suffixes/surgical_suffixes.txt", header = None)
df_psych = pd.read_csv("suffixes/psychotherapy_keywords.txt", header = None)

df_psych[0] = df_psych[0].str.lower()
df_surgery[0] = df_surgery[0].str.lower()
df_drugs[0] = df_drugs[0].str.lower()



# Read in FDA data.

df_purple = pd.read_csv("fda_approved_drugs/products_purplebook.csv")
df_orange = pd.read_csv("fda_approved_drugs/products_orangebook.txt", 
                        sep = "~")
df_drugs_at_fda = pd.read_csv("fda_approved_drugs/products_drugs_at_fda.txt", 
                              sep = "\t", 
                              error_bad_lines = False)



# Filter drug suffixes with three characters or fewer.
drug_suffixes = list(df_drugs[0])
drug_suffixes = [x.lower() for x in drug_suffixes if len(x) > 3]




# Concatenate FDA drug data.
set_proprietary = list(df_drugs_at_fda["DrugName"]) + list(df_purple["Proprietary Name"]) + list(df_orange["Trade_Name"])
set_proper = list(df_drugs_at_fda["ActiveIngredient"]) + list(df_purple["Proper Name"]) + list(df_orange["Ingredient"])

# Remove floats and integers.
set_proprietary = [item.lower() for item in set_proprietary if not isinstance(item, float)]
set_proprietary = [item for item in set_proprietary if not isinstance(item, int)]
set_proper = [item.lower() for item in set_proper if not isinstance(item, float)]
set_proper = [item for item in set_proper if not isinstance(item, int)]

# Cast as sets to remove duplicates.
set_proprietary = set(set_proprietary)
set_proper = set(set_proper)
set_fda = set.union(set_proprietary, set_proper)



# read the pickle file
with open('df_orig.pickle', 'rb') as handle:
    df_orig = pickle.load(handle)

# Remove None type labels. BERT does the same.
df_orig = df_orig.dropna()
df_orig = df_orig.reset_index(drop = True)

# Train-test split (80% / 20%, stratified by gold label value).
X_train, X_test, y_train, y_test = train_test_split(df_orig["token"], 
                                                    df_orig["gold"], 
                                                    test_size = 0.1, 
                                                    random_state = 42)
df_train = df_orig.iloc[X_train.index].reset_index(drop = True)
df_test = df_orig.iloc[X_test.index].reset_index(drop = True)



# Label macros.
ABSTAIN = -1
NOT_I = 0
I = 1

# Data for labeling functions.
generic_interventions = ["therap", "treatment", "intervention",
                         "placebo", "dose", "control", "vaccin"]
nltk.download("stopwords")
stop_words = stopwords.words("english")
print("Total stop words =", len(stop_words))




# Labeling function for tokens, if token is present in first sentence (title)
@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def in_title(x, first_sentence_file):
    return I if x.token.lower() in first_sentence_file[x.file] else ABSTAIN
# abstain or not_i?

# Labeling function for tokens, if token is present in first sentence (title)
@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def in_title2(x, first_sentence_file):
    return I if x.token.lower() in first_sentence_file[x.file] else ABSTAIN
# abstain or not_i?

# Labeling function for tokens, if token is present in first sentence (title)
@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def in_title3(x, first_sentence_file):
    return I if x.token.lower() in first_sentence_file[x.file] else ABSTAIN
# abstain or not_i?

# Labeling function for tokens, if token is present in first sentence (title)
@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def in_title4(x, first_sentence_file):
    return I if x.token.lower() in first_sentence_file[x.file] else ABSTAIN
# abstain or not_i?


# Labeling function for tokens, if token is present in first sentence (title)
@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def not_in_title(x, first_sentence_file):
    return NOT_I if x.token.lower() not in first_sentence_file[x.file] else ABSTAIN
# abstain or not_i?


@labeling_function(resources=dict(first_sentence_file=first_sentence_file))
def surround_in_title(x, first_sentence_file):
    if((x.token_index>0) and (x.token_index<len(x.abstract)-1)):
        if (x.abstract[x.token_index-1].lower() in first_sentence_file[x.file]) and (x.abstract[x.token_index+1].lower() in first_sentence_file[x.file]):
            return I
    
    return ABSTAIN
# abstain or not_i?


# Labeling function for tokens that contain drug suffixes.
@labeling_function()
def contains_drug_suffix(x):
    return I if (any(suffix.lower() in x.token.lower() for suffix in drug_suffixes)) else ABSTAIN
    
# Labeling function for tokens that contain surgical suffixes.
@labeling_function()
def contains_surgical_suffix(x):
    return I if (any(suffix.lower() in x.token.lower() for suffix in df_surgery[0])) else ABSTAIN

# Labeling function for tokens that contain psychological / psychotherapeutic keywords.
@labeling_function()
def contains_psych_term(x):
    return I if (any(suffix.lower() in x.token.lower() for suffix in df_psych[0])) else ABSTAIN

# Labeling function for tokens that contain generic intervention keywords.
@labeling_function()
def is_generic(x):
    return I if (any(term.lower() in x.token.lower() for term in generic_interventions)) else ABSTAIN

# Labeling function for stop words.
@labeling_function()
def is_stop_word(x):
    return NOT_I if x.token.lower() in stop_words else ABSTAIN

# Labeling function for tokens that are punctuation.
@labeling_function()
def is_punctuation(x):
    return NOT_I if x.token.lower() in string.punctuation else ABSTAIN


# Labeling function for FDA approved drugs.
@labeling_function()
def contains_fda_drug(x):
    if (len(x.token) <= 5):
        return ABSTAIN

    return I if (any(x.token.lower() in drug.lower() for drug in set_fda)) else ABSTAIN


# checks if the preceding token is 'of' or 'with' (effect of... I, treat with... I)
@labeling_function()
def has_prev_word_as(x):
    words = ['of', 'with', 'receive', 'and']
    if ((x.token_index > 0) and (x.abstract[x.token_index-1].lower() in words)):
        return I 

    else:
        return ABSTAIN
    
# checks if the next token is 'group' or 'groups'
@labeling_function()
def has_next_word_as(x):
    words = ['group', 'groups']
    if ((x.token_index < len(x.abstract)-1) and (x.abstract[x.token_index+1].lower() in words)):
        return NOT_I

    else:
        return ABSTAIN
    
# Labeling function which labels a token as NOT_I if it is in the last 50% of the file tokens.
@labeling_function()
def has_high_idx(x):
    percent = x.token_index / x.file_len
    if percent > 0.50:
        return NOT_I
    else:
        return ABSTAIN
    
    
    
# Labeling function for tokens, sees if left span of token within sentence contains keyword
@labeling_function()
def left_span_contains(x):
    
    i = 0
    while(x.abstract[i] != x.token):
        i+=1
        
    count = 0
    while(i >= 0 and count < 10):
        if((x.abstract[i] == 'determine') or (x.abstract[i] == 'assess')):
            return I
        i-=1
        count+=1
        
    return ABSTAIN
# look into spouse tutorial left spans, and using 'resources' in LFs


# checks if the preceding token is VBD, VBN (e.g. was administered)
@labeling_function()
def right_span_vb_pos(x):
    if (x.token_index < len(x.abstract) - 2) and (x.pos_abstract[x.token_index+1] == 'VBD') and (x.pos_abstract[x.token_index+2] == 'VBN'):
        return I 

    else:
        return ABSTAIN
    
    
# checks if the preceding token is VBD, VBN (e.g. was administered)
@labeling_function()
def left_span_vb_pos(x):
    if (x.token_index > 0) and ('V' in x.pos_abstract[x.token_index-1]):
        return I

    else:
        return ABSTAIN
    


    

    

    
# Apply LFs to dataframe.
lfs = [
       #contains_psych_term, # accuracy = 0.131380
       # is_punctuation,
       #has_prev_word_as,
    
       # has_next_word_as_drug, low accuracy and coverage
    
       # left_span_contains,
       # right_span_vb_pos,
       # left_span_vb_pos,
    
       #negative LFs
       is_stop_word,
       has_next_word_as,
       has_high_idx,
    
       #positive LFs
       is_generic,
       contains_drug_suffix,
       contains_surgical_suffix,
       contains_fda_drug,
    
#        in_title,
#         in_title2,
#         in_title3,
#         in_title4,
       # not_in_title,
#        surround_in_title,

      ]

print('got to here')

applier = PandasLFApplier(lfs = lfs)
L_train = applier.apply(df = df_train)
L_test = applier.apply(df = df_test)
# L_dev = applier.apply(df = df_dev)



# Define Y_train, Y_test.
Y_train = df_train["gold"].to_numpy(dtype = int)
Y_test = df_test["gold"].to_numpy(dtype = int)
# Y_dev = df_dev["Gold"].to_numpy(dtype = int)



# Summarive coverage, conflicts, empirical accurcacy of LFs.
LFAnalysis(L_train, lfs).lf_summary(Y_train)
# LFAnalysis(L_dev, lfs).lf_summary(Y_dev)



LFAnalysis(L_test, lfs).lf_summary(Y_test)



# Majority vote model.
majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L = L_train)





# Label model.
label_model = LabelModel(cardinality = 2, verbose = True)
label_model.fit(L_train = L_train, n_epochs = 500, log_freq = 100, seed = 123)





# Compute model performance metrics.
majority_scores = majority_model.score(L = L_test, Y = Y_test, 
                                       tie_break_policy = "random",
                                       metrics = ["f1", "accuracy", "precision", 
                                                  "recall", "roc_auc", "coverage"])
label_scores = label_model.score(L = L_test, Y = Y_test, 
                                 tie_break_policy = "random",
                                 metrics = ["f1", "accuracy", "precision", 
                                            "recall", "roc_auc", "coverage"])




# Compare model performance metrics.
majority_f1 = majority_scores.get("f1")
majority_acc = majority_scores.get("accuracy")
majority_prec = majority_scores.get("precision")
majority_rec = majority_scores.get("recall")
majority_roc = majority_scores.get("roc_auc")
majority_cov = majority_scores.get("coverage")
print(f"{'Majority Model F1:':<25} {majority_f1 * 100:.1f}%")
print(f"{'Majority Model Accuracy:':<25} {majority_acc * 100:.1f}%")
print(f"{'Majority Model Precision:':<25} {majority_prec * 100:.1f}%")
print(f"{'Majority Model Recall:':<25} {majority_rec * 100:.1f}%")
print(f"{'Majority Model AUC ROC:':<25} {majority_roc * 100:.1f}%")
print(f"{'Majority Model Coverage:':<25} {majority_cov * 100:.1f}%")
print("++++++++++++++++++++++++")

label_f1 = label_scores.get("f1")
label_acc = label_scores.get("accuracy")
label_prec = label_scores.get("precision")
label_rec = label_scores.get("recall")
label_roc = label_scores.get("roc_auc")
label_cov = label_scores.get("coverage")
print(f"{'Label Model F1:':<25} {label_f1 * 100:.1f}%")
print(f"{'Label Model Accuracy:':<25} {label_acc * 100:.1f}%")
print(f"{'Label Model Precision:':<25} {label_prec * 100:.1f}%")
print(f"{'Label Model Recall:':<25} {label_rec * 100:.1f}%")
print(f"{'Label Model AUC ROC:':<25} {label_roc * 100:.1f}%")
print(f"{'Label Model Coverage:':<25} {label_cov * 100:.1f}%")





# View "dummy" accuracy if predicting majority class every time.
print("Accuracy if predicting majority class", 
      df_test["gold"].value_counts(normalize = True).max())
