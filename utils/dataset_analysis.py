import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def morgan_mds(morgan_list, n):
    """
    :param morgan_list: morgan fp list
    :param n: Dimension after dimensionality reduction
    :return: mds result
    """
    # 1.build dist matrix
    num = len(morgan_list)
    D = np.zeros([num, num])
    for i in range(num):
        for j in range(num):
            D[i][j] = 1 - DataStructs.TanimotoSimilarity(morgan_list[i], morgan_list[j])
    # 2.centralization dist matrix
    I = np.eye(num)
    i = np.ones([num, num])
    H = I - (1 / num) * i
    # 3.calculate inner product matrix
    B = -1/2 * np.dot(np.dot(H, D**2), H)
    # 4.eigenvalue decomposition
    Be, Bv = np.linalg.eigh(B)  # eigenvalues of Be matrix B, normalized eigenvectors of Bv
    simple_vector = Bv[:, np.argsort(-Be)[:n]]
    simple_val = Be[np.argsort(-Be)[:n]]
    Z = simple_vector * simple_val ** 0.5
    return Z

def top_coverage(rxn_list, n):
    """
    :param rxn_list: rxn_class list
    :param n: top n in number
    :return: 3 dicts, shows the coverage rate of top n substances
    """
    rxn_cat_dict = dict()
    rxn_sol_dict = dict()
    num_cat = 0
    num_sol = 0
    for rxn in rxn_list:
        # reagents
        for reag in rxn.reagents:
            num_cat += 1
            if reag not in rxn_cat_dict:
                rxn_cat_dict["%s" % reag] = 1
            else:
                rxn_cat_dict["%s" % reag] += 1
        # catalysts
        for cat in rxn.cats:
            num_cat += 1
            if cat not in rxn_cat_dict:
                rxn_cat_dict["%s" % cat] = 1
            else:
                rxn_cat_dict["%s" % cat] += 1
        # solvents
        for sol in rxn.solvents:
            num_sol += 1
            if sol not in rxn_sol_dict:
                rxn_sol_dict["%s" % sol] = 1
            else:
                rxn_sol_dict["%s" % sol] += 1
    # number of top n
    rxn_cat_dict = list(zip(rxn_cat_dict.values(), rxn_cat_dict.keys()))
    rxn_cat_dict = sorted(rxn_cat_dict, reverse=True)
    rxn_sol_dict = list(zip(rxn_sol_dict.values(), rxn_sol_dict.keys()))
    rxn_sol_dict = sorted(rxn_sol_dict, reverse=True)
    # return coverage rate list
    rxn_cat_list = list()
    rxn_sol_list = list()
    for i in range(n):
        if i == 0:
            rxn_cat_list.append(rxn_cat_dict[0:n][i][0] / num_cat)
            rxn_sol_list.append(rxn_sol_dict[0:n][i][0] / num_sol)
        else:
            rxn_cat_list.append(rxn_cat_list[i-1] + rxn_cat_dict[0:n][i][0] / num_cat)
            rxn_sol_list.append(rxn_sol_list[i-1] + rxn_sol_dict[0:n][i][0] / num_sol)

    return rxn_cat_list, rxn_sol_list

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
def DBSCAN_filter(data0, eps, min_samples):
    """
    :param data0: mds data
    :param eps: min radius
    :param min_samples: min sample around a point
    :return: normal points
    """
    scaler = StandardScaler()
    scaler.fit(data0)
    data = scaler.transform(data0)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(data)
    labels = model.labels_

    return labels

# test
if __name__ == "__main__":
    smis = ["CCCCOC(=C)C1=C(C)C2=C(N=C(NC3=CC=C(C=N3)N3CCN(CC3)C(=O)OC(C)(C)C)N=C2)N(C2CCCC2)C1=O", "CCOC(=O)\C=C\C1=CC=CC(=C1)C(O)=O", "CC(C)C1=COC2=C1C=CC=C2"]
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols]
    similarity = DataStructs.TanimotoSimilarity(fps[0], fps[1])
    print(similarity)
    print(morgan_mds(fps, 2).shape)
    print(DBSCAN_filter(morgan_mds(fps, 2), eps=0.01, min_samples=1))