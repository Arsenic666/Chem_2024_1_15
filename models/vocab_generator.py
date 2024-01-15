import re
import numpy as np
import pandas as pd
from utils.rxn import *
from utils.molecule import *
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# size of word vector
size = 100
train_set = list()
vocab_dict = dict()

# BH_HTE
BH_HTE_df = pd.read_excel("../data/BH_HTE/BH_HTE_data.xlsx")
BH_text = list()
for i in range(BH_HTE_df.shape[0]):
    base = smi_tokenizer(BH_HTE_df.loc[i]["base_smiles"])
    ligand = smi_tokenizer(BH_HTE_df.loc[i]["ligand_smiles"])
    aryl_halide = smi_tokenizer(BH_HTE_df.loc[i]["aryl_halide_smiles"])
    additive = smi_tokenizer(BH_HTE_df.loc[i]["additive_smiles"])
    product = smi_tokenizer(BH_HTE_df.loc[i]["product_smiles"])

    text = smi_tokenizer("CC1=CC=C(N)C=C1") + ["."] + aryl_halide + ["."] + additive + ["."] + base + ["."] + ligand + [">>"] + product
    train_set.append(text)
    BH_text.append(" ".join(text))

BH_text = pd.DataFrame(BH_text, columns=["rxnfp"])
BH_HTE_df = pd.concat([BH_HTE_df, BH_text], axis=1)
BH_HTE_df.to_excel("../data/BH_HTE/BH_HTE_rxnfp.xlsx")

# Heck
Heck_df = pd.read_excel("../data/Heck/Heck processed data.xlsx")
Heck_rxn_list = df_to_rxn_list(Heck_df)
Heck_text = list()
for rxn in Heck_rxn_list:
    text = list()
    for reactant in rxn.reactants:
        text = text + smi_tokenizer(reactant) + ["."]

    for reagent in rxn.reagents:
        text = text + smi_tokenizer(reagent) + ["."]

    for cat in rxn.cats:
        text = text + smi_tokenizer(cat) + ["."]

    for sol in rxn.solvents:
        text = text + smi_tokenizer(sol) + ["."]

    text = text + [">>"]
    for j in range(len(rxn.products)):
        if j == len(rxn.products) - 1:
            text = text + smi_tokenizer(rxn.products[j])
        else:
            text = text + smi_tokenizer(rxn.products[j]) + ["."]

    train_set.append(text)
    Heck_text.append(" ".join(text))

Heck_text = pd.DataFrame(Heck_text, columns=["rxnfp"])
Heck_df = pd.concat([Heck_df, Heck_text], axis=1)
Heck_df.to_excel("../data/Heck/Heck_rxnfp.xlsx")


# Suzuki_HTE
Suzuki_HTE_df = pd.read_excel("../data/Suzuki_HTE/Suzuki_HTE_data.xlsx")
Suzuki_text = list()

# vocab renew
for i in range(Suzuki_HTE_df.shape[0]):
    reactant1 = Suzuki_HTE_df.loc[i]["Reactant_1_Name"]
    reactant2 = Suzuki_HTE_df.loc[i]["Reactant_2_Name"]
    cat = Suzuki_HTE_df.loc[i]["Catalyst_1_Short_Hand"]
    ligand = Suzuki_HTE_df.loc[i]["Ligand_Short_Hand"]
    base = Suzuki_HTE_df.loc[i]["Reagent_1_Short_Hand"]
    sol = Suzuki_HTE_df.loc[i]["Solvent_1_Short_Hand"]

    if pd.isnull(ligand) and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(sol)
    if pd.isnull(ligand) and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol)
    if pd.isnull(ligand) == False and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(sol)
    if pd.isnull(ligand) == False and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol)

    train_set.append(text)
    Suzuki_text.append(" ".join(text))

Suzuki_text = pd.DataFrame(Suzuki_text, columns=["rxnfp"])
Suzuki_HTE_df = pd.concat([Suzuki_HTE_df, Suzuki_text], axis=1)
Suzuki_HTE_df.to_excel("../data/Suzuki_HTE/Suzuki_HTE_rxnfp.xlsx")

# generate vocab file
print("The length of train set is: %d" % len(train_set))
word_id = dict()
word_vec = list()
model = Word2Vec(train_set, vector_size=size, window=20, min_count=5, epochs=20, sg=1)
for i, w in enumerate(model.wv.index_to_key):
    vocab_dict[w] = model.wv[w]
    # record for evaluation
    word_id[w] = i
    word_vec.append(model.wv[w])
vocab_dict_to_txt(vocab_dict)
print("There are %d atoms in the dict" % len(vocab_dict))

# Evaluation
X_reduced = PCA(n_components=2).fit_transform(np.array(word_vec))
q_w = ["Cl", "F", "Br", "I", "[Pd]", "[Pd++]", "[Pd+2]"]
plt.figure(dpi=120)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color="black")
for w in q_w:
    if w in word_id.keys():
        xy = X_reduced[word_id[w], :]
        plt.scatter(xy[0], xy[1], color="r")
        plt.text(xy[0], xy[1], w, color="b")

# plt.figure(dpi=120)
# for w in word_id.keys():
#     xy = X_reduced[word_id[w], :]
#     plt.scatter(xy[0], xy[1], color="r")
#     plt.text(xy[0], xy[1], w, color="b")

plt.tight_layout()
plt.show()