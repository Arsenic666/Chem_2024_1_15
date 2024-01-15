from utils.rxn import *
from utils.dataset_analysis import *
import matplotlib.pyplot as plt

# 1.import data
data = pd.read_excel("../../data/Heck/Heck processed data.xlsx")

# 2.convert df into rxn_class list
rxn_list = df_to_rxn_list(data)

# 3.divide into intra & inter
rxn_intra = list() # intra
rxn_inter = list() # inter
for rxn in rxn_list:
    if len(rxn.reactants) == 1:
        rxn_intra.append(rxn)
    if len(rxn.reactants) == 2:
        rxn_inter.append(rxn)

# 4.count rxn types
# intra
cnt_intra = 1
for i in range(1, len(rxn_intra)):
    if rxn_intra[i].index > rxn_intra[i - 1].index:
        cnt_intra += 1
# inter
cnt_inter = 1
for i in range(1, len(rxn_inter)):
    if rxn_inter[i].index > rxn_inter[i - 1].index:
        cnt_inter += 1
print("The reaction types of intramolecular reaction is %d" % cnt_intra)
print("The reaction types of intermolecular reaction is %d" % cnt_inter)

# 5.analysis of yield distribution
yield_intra = [0] * 20
yield_inter = [0] * 20
yield_total = [0] * 20
x_label = np.linspace(2.5, 97.5, 20)
# yield number
for rxn in rxn_intra:
    for i in range(1, len(x_label)):
        if int(rxn.rxn_yield) > x_label[i-1] - 2.5 and int(rxn.rxn_yield) <= x_label[i] + 2.5:
            yield_intra[i] += 1
for rxn in rxn_inter:
    for i in range(1, len(x_label)):
        if int(rxn.rxn_yield) > x_label[i-1] and int(rxn.rxn_yield) <= x_label[i]:
            yield_inter[i] += 1
for rxn in rxn_list:
    for i in range(1, len(x_label)):
        if int(rxn.rxn_yield) > x_label[i-1] and int(rxn.rxn_yield) <= x_label[i]:
            yield_total[i] += 1
# figure of yield distribution
fig, ax = plt.subplots(ncols=3, dpi=120, figsize=(10, 5))
ax[0].bar(x_label, yield_intra, width=5, color=[114/255, 188/255, 213/255], edgecolor="k")
ax[1].bar(x_label, yield_inter, width=5, color=[255/255, 208/255, 111/255], edgecolor="k")
ax[2].bar(x_label, yield_total, width=5, color=[231/255, 98/255, 84/255], edgecolor="k")
# beautify
fig.suptitle("Yield Distribution", fontsize=16)
ax[0].legend(["Intramolecular"], loc="upper left", prop={'size': 11})
ax[1].legend(["Intermolecular"], loc="upper left", prop={'size': 11})
ax[2].legend(["Total"], loc="upper left", prop={'size': 11})
for i in range(3):
    ax[i].set_xlabel("Yield", fontsize=11)
    ax[i].set_ylabel("Count", fontsize=11)
    ax[i].set_xticks([0, 25, 50, 75, 100])
    ax[i].grid(True, axis="y", alpha=0.5, linestyle="--")
plt.tight_layout()
plt.savefig("../../figures/Yield Distribution.png")
plt.show()

# 6.mds analysis
# get mol
mol_intra = [Chem.MolFromSmiles(rxn.products[0]) for rxn in rxn_intra]
mol_inter = [Chem.MolFromSmiles(rxn.products[0]) for rxn in rxn_inter]
# get morgan fp
fp_r = 2
morgan_intra = list()
for mol in mol_intra:
    morgan_intra.append(AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, 1024))
morgan_inter = list()
for mol in mol_inter:
    morgan_inter.append(AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, 1024))
# mds matrix
mds_total = morgan_mds(morgan_intra + morgan_inter, n=2)
# Figure
plt.figure(dpi=120)
# inter
plt.scatter(mds_total[len(morgan_intra):, 0], mds_total[len(morgan_intra):, 1], color=[117/255, 157/255, 219/255], marker=".", s=50)
# intra
plt.scatter(mds_total[0:len(morgan_intra), 0], mds_total[0:len(morgan_intra), 1], color=[236/255, 164/255, 124/255], marker=".", s=50)
# beautify
plt.title("MDS Analysis",fontsize=16)
plt.xlabel("MDS1", fontsize=11)
plt.ylabel("MDS2", fontsize=11)
plt.grid(True, alpha=0.5, linestyle="--")
plt.legend(["Intermolecular", "Intramolecular"], loc="best")
plt.tight_layout()
plt.savefig("../../figures/MDS Analysis.png")
plt.show()

# 7.analysis of database quality
# Attention: reagents & catalysts are all called catalysts
top_n = 15
# intra
intra_cat, intra_sol = top_coverage(rxn_intra, n=top_n)
# inter
inter_cat, inter_sol = top_coverage(rxn_inter, n=top_n)
# total
total_cat, total_sol = top_coverage(rxn_list, n=top_n)
# Figure
x_label = np.linspace(1, top_n, top_n)
fig, ax = plt.subplots(ncols=3, dpi=120, figsize=(10, 5))
ax[0].plot(x_label, intra_cat, color=[236/255, 164/255, 124/255], marker="^")
ax[0].plot(x_label, intra_sol, color=[117/255, 157/255, 219/255], marker="s")
ax[1].plot(x_label, inter_cat, color=[236/255, 164/255, 124/255], marker="^")
ax[1].plot(x_label, inter_sol, color=[117/255, 157/255, 219/255], marker="s")
ax[2].plot(x_label, total_cat, color=[236/255, 164/255, 124/255], marker="^")
ax[2].plot(x_label, total_sol, color=[117/255, 157/255, 219/255], marker="s")
# beautify
fig.suptitle("Reagents Diversity", fontsize=16)
ax[0].set_title("Intramolecular", fontsize=11)
ax[1].set_title("Intermolecular", fontsize=11)
ax[2].set_title("Total", fontsize=11)
for i in range(3):
    ax[i].legend(["catalysts", "solvents"], loc="upper left", prop={'size': 8})
    ax[i].set_xlabel("Top N", fontsize=11)
    ax[i].set_ylabel("Reaction Coverage", fontsize=11)
    ax[i].grid(True, alpha=0.5, linestyle="--")
plt.tight_layout()
plt.savefig("../../figures/Reagents Diversity.png")
plt.show()

# 8.generate excel tables respectively
df_intra = rxn_list_to_df(rxn_intra)
df_intra.to_excel("../../data/Heck/Intramolecular data.xlsx", sheet_name="intra data")
df_inter = rxn_list_to_df(rxn_inter)
df_inter.to_excel("../../data/Heck/Intermolecular data.xlsx", sheet_name="inter data")