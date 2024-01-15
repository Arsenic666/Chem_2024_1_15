"""
This Module contains some tools to convert the formation of molecules
    A: iupac to smi
    B: smi to fp
    C: smi to graph
    D: smi normalization
"""
import re
# A. iupac to smi
from time import sleep
from lxml import etree
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import quote

def iupac_to_smi_cactus(name):
    if name == "/":
        return None
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        smi = urlopen(url).read().decode('utf8')
        return smi
    except:
        return None

def iupac_to_smi_pubchem(name, bro):
    try:
        # open pubchem
        bro.get("https://pubchem.ncbi.nlm.nih.gov/#query=%s" % name)
        sleep(8)
        # Get current webpage information
        soup = BeautifulSoup(bro.page_source, "html.parser")
        feature_res = soup.find_all("div", id="featured-results")
        html_parser = etree.HTML(str(feature_res[0]))
        # smi for obtaining the best matching result
        label = html_parser.xpath('./descendant::*/text()')
        num = len(label)
        # If there are no search results
        if num == 0:
            return None
        for i in range(num):
            if label[i] == "Isomeric SMILES: ":
                smi = label[i + 1]
                break
            # If there is no smi
            if i == num - 1:
                return None
        return smi
    except:
        return None

class molecule():
    """
    Class of molecules
    """
    def __init__(self):
        self.name = str()
        self.smi = str()
    def record(self, name, smi):
        self.name = name
        self.smi = smi

class Mol_Manager():
    """
    Molecular name management library
    """
    def __init__(self, bro):
        self.no_smi = list() # Unable to find smi
        self.yes_smi = list() # Able to find smi
        self.bro = bro

    def get_smi(self, name):
        # in no_smi
        pd_no = False
        for mol in self.no_smi:
            if name == mol.name:
                pd_no = True
                return mol.smi

        # in yes_smi
        pd_yes = False
        for mol in self.yes_smi:
            if name == mol.name:
                pd_yes = True
                return mol.smi

        # Neither of the two
        if pd_no == False and pd_yes == False:
            # get smi
            smi = iupac_to_smi_cactus(name)
            if smi == None: # If cactus does not work, use PubChem
                smi = iupac_to_smi_pubchem(name, bro=self.bro)

            # put it into list
            mol = molecule()
            mol.record(name, smi)
            if smi == None:
                self.no_smi.append(mol)
            else:
                self.yes_smi.append(mol)
            return smi

# B.smi to fp
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smis_to_MACCSfp(smis):
    """
    This function is to generate MACCS-keys FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(AllChem.GetMACCSKeysFingerprint(mol)) for mol in mols]
    bits = len(fps[0])
    return fps, bits

from rdkit.Avalon import pyAvalonTools
def smis_to_AVfp(smis):
    """
        This function is to generate Avalon FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(pyAvalonTools.GetAvalonFP(mol)) for mol in mols]
    bits = len(fps[0])
    return fps, bits

def smis_to_APfp (smis):
    """
    This function is to generate Atom-Pairs FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)) for mol in mols]
    bits = len(fps[0])
    return fps, bits

def smis_to_TTfp(smis):
    """
    This function is to generate Topological-Torsions FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)) for mol in mols]
    bits = len(fps[0])
    return fps, bits

def smis_to_MGfp(smis, radius, nBits):
    """
    This function is to generate Morgan-Circular FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)) for mol in mols]
    return fps

def smis_to_RDKfp(smis):
    """
    This function is to generate RDkit FP
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = [np.array(Chem.RDKFingerprint(mol)) for mol in mols]
    bits = len(fps[0])
    return fps, bits

# C.smi to graph
import torch
from torch_geometric.data import Data

def smi_to_graph(smi):
    """
    :param smi:
    :return:
        :num: number of atoms
        :feature: feature of nodes, include AtomicNumber & FormalCharge
        :edge_index: edge information
    """
    # Get num
    mol = Chem.MolFromSmiles(smi)
    num = mol.GetNumAtoms()
    # Get label
    feature = list()
    atoms = mol.GetAtoms()
    for atom in atoms:
        # 7 features: Atomic Num, Formal Charge, Number of connected H, Explicit Valence, bonds the atom involved
        atom_feature = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetExplicitValence(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing())
        ]
        feature.append(atom_feature)
    # Get edge_index
    us = list()
    vs = list()
    bonds = mol.GetBonds()
    for bond in bonds:
        u = bond.GetBeginAtom().GetIdx()
        v = bond.GetEndAtom().GetIdx()
        us.append(u)
        vs.append(v)
        us.append(v)
        vs.append(u)
    edge_index = [us, vs]

    return num, feature, edge_index

def smis_to_graph(smis):
    temp_num = 0
    g_feature = list()
    g_us = list()
    g_vs = list()
    for smi in smis:
        num, feature, edge_index = smi_to_graph(smi)
        for atom in feature:
            g_feature.append(atom)
        for u in edge_index[0]:
            g_us.append(u + temp_num)
        for v in edge_index[1]:
            g_vs.append(v + temp_num)
        temp_num += num
    g_edge_index = [g_us, g_vs]

    x = torch.tensor(g_feature, dtype=torch.long)
    edge_index = torch.tensor(g_edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data

# D:SMILES Normalization
from rdkit import Chem
from rdkit.Chem import MolStandardize

def Mol_Clean(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = MolStandardize.normalize.Normalizer().normalize(mol)
            mol = MolStandardize.fragment.LargestFragmentChooser().choose(mol)
            mol = MolStandardize.charge.Uncharger().uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
    except:
        return False


# E:SMILES Tokenizer
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule
    """
    import re
    pattern = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

# F:Format conversion: vocab_list & vocab_dict
def vocab_dict_to_txt(dict):
    with open("../utils/vocab.txt", mode="w", encoding="utf-8") as txt:
        for key in dict.keys():
            txt.write("%s\t%s\t\t" % (key, dict[key]))
    txt.close()

def vocab_txt_to_dict(file):
    vocab_dict = dict()
    txt = open(file, mode="r", encoding="utf-8")
    vocab = txt.read().replace("\n", "").split("\t\t")
    vocab = vocab[:-1] # remove the last one, which is ""

    for pair in vocab:
        pair = pair.split("\t")
        vec = pair[1][1:-1].replace("  ", " ").split(" ") # delete "[", "]" and split by " "
        while "" in vec:
            vec.remove("") # remove null element
        vocab_dict[pair[0]] = np.array(vec).astype(float)

    return vocab_dict


# test
if __name__ == "__main__":
    # from numpy import random
    # size = 2
    # vocab_dict = dict()
    # smi = smi_tokenizer("CC(C)(C)OC(=O)[N-]C1=C[N+](=NO1)C1=CC=C(I)C=C1.[Pd++]")
    # for atom in smi:
    #     if atom not in vocab_dict.keys():
    #         vocab_dict[atom] = random.random(size=(1, size))
    # vocab_dict_to_txt(vocab_dict)
    file = "./vocab.txt"
    vocab_txt_to_dict(file)