"""
This Module contains some tools to convert the formation of molecules
    A: RXN class
    B: dataframe to list
    C: list to dataframe
    D: Heck rxn checker

"""
import re
import pandas as pd
import torch


class RXN():
    # rxn的自身参数
    def __init__(self):
        self.index = str() # 反应索引

        self.reactants = list() # 反应物
        self.products = list() # 产物

        self.reagents = list() # 试剂
        self.cats = list() # 催化剂
        self.solvents = list() # 溶剂
        self.temp = str() # 温度
        self.time = str() # 时间

        self.rxn_yield = str() # 产率

        self.ref = str() # 参考文献
        self.rxn_id = str() # Reaxys ID

    # rxn信息获取
    def get_info(self, html_parser, i):
        """
        :param html_parser: html 解析器
        :param i: 该反应类型中的第i条条件
        """
        # 反应索引
        self.index = html_parser.xpath("//span[@class='rx-element-index']/text()")[0]

        rxn_index = html_parser.xpath("//div[@class='rx-reactions-table__conditions__steps']")[i]
        # 若不存在stages-row
        rxn_index = rxn_index.xpath('./div[@class="stages-row"]')
        if len(rxn_index) == 0:
            return
        else:
            rxn_index = rxn_index[0]
        # 试剂
        rxn_r = rxn_index.xpath('./span[@class="stage-reagents"]')
        if len(rxn_r) == 0:
            pass
        else:
            for rea_html in rxn_r:
                rea = str(rea_html.xpath("string()"))
                if len(re.findall(".+[^;\xa0]", rea)) == 0:
                    self.reagents.append(rea)
                else:
                    rea = re.findall(".+[^;\xa0]", rea)[0] # html文本中有奇怪字符
                    self.reagents.append(rea)
        # 催化剂
        rxn_c = rxn_index.xpath("./span[@class='stage-catalyst']")
        if len(rxn_c) == 0:
            pass
        else:
            for cat_html in rxn_c:
                cat = str(cat_html.xpath("string()"))
                if len(re.findall(".+[^;\xa0]", cat)) == 0:
                    self.cats.append(cat)
                else:
                    cat = re.findall(".+[^;\xa0]", cat)[0] # html文本中有奇怪字符
                    self.cats.append(cat)
        # 溶剂
        rxn_s = rxn_index.xpath("./span[@class='stage-solvents']")
        if len(rxn_s) == 0:
            pass
        else:
            for sol_html in rxn_s:
                sol = str(sol_html.xpath("string()"))
                if len(re.findall(".+[^;\xa0]", sol)) == 0:
                    self.solvents.append(sol)
                else:
                    sol = re.findall(".+[^;\xa0]", sol)[0] # html文本中有奇怪字符
                    self.solvents.append(sol)
        # 时间 & 温度
        rxn_cond = rxn_index.xpath("string(./span[@class='conditions'])")
        temp = re.findall("(?<=at\s)-?[0-9]+\.[0-9]*|-?[0-9]+(?=℃)", rxn_cond)
        time = re.findall("(?<=for\s)-?[0-9]+\.[0-9]*|-?[0-9]+(?=h)", rxn_cond)
        if len(temp) == 0:
            self.temp = "/"
        else:
            self.temp = temp[0]

        if len(time) == 0:
            self.time = "/"
        else:
            self.time = time[0]

        # 产率
        rxn_y = html_parser.xpath("//td[@class='rx-reactions-table__yield display-table-cell']")[i]
        if len(rxn_y) == 0:
            self.rxn_yield = "/"
        else:
            self.rxn_yield = rxn_y.xpath("string()")

        # 参考文献
        rxn_ref = html_parser.xpath("//div[@class='citation clear']")[i]
        if len(rxn_ref) == 0:
            self.ref = "/"
        else:
            self.ref = rxn_ref.xpath("string()")

    def show_info(self):
        print(self.index, self.reactants, self.products, self.reagents, self.cats, self.solvents, self.temp, self.time, self.rxn_yield, self.rxn_id, self.ref)

# 将pandas的dataframe转为rxn的list
def df_to_rxn_list(df):
    """
    :param df: pandas的dataframe
    :return: 所有反应的rxn_class，以list类型返回
    """

    rxn_list = list()  # 反应的list
    data_size = df.shape[0]  # 获取反应数量

    for num in range(data_size):
        rxn = RXN() # 实例化反应的类
        # 获取 index
        rxn.index = df.loc[num]["rxn_index"]
        # 获取 temperature /C
        rxn.temp = df.loc[num]["temperature /C"]
        # 获取 time /h
        rxn.time = df.loc[num]["time /h"]
        # 获取 Yield
        rxn.rxn_yield = df.loc[num]["Yield"]
        # 获得Reaction ID
        rxn.rxn_id = df.loc[num]["Reaction ID"]
        # 获得参考文献
        rxn.ref = df.loc[num]["Reference"]

        # 获取 reactant
        for col in df.columns:
            if "reactants" in col:
                if df.loc[num][col] != "/": # 去除 /
                    rxn.reactants.append(df.loc[num][col])
        # 获取 product
        for col in df.columns:
            if "products" in col:
                if df.loc[num][col] != "/":  # 去除 /
                    rxn.products.append(df.loc[num][col])
        # 获取 reagent
        for col in df.columns:
            if "reagents" in col:
                if df.loc[num][col] != "/":  # 去除 /
                    rxn.reagents.append(df.loc[num][col])
        # 获取 catalyst
        for col in df.columns:
            if "catalysts" in col:
                if df.loc[num][col] != "/":  # 去除 /
                    rxn.cats.append(df.loc[num][col])
        # 获取 solvents
        for col in df.columns:
            if "solvents" in col:
                if df.loc[num][col] != "/":  # 去除 /
                    rxn.solvents.append(df.loc[num][col])

        rxn_list.append(rxn) # 放入list

    return rxn_list

# 将rxn_list转为df
def rxn_list_to_df(rxn_list):
# 遍历rxn_list, 找到reactants, products...等参数的最大数量,以确定dataframe的规格
    # 设置最大个数
    num_reactants = 0
    num_products = 0
    num_reagents = 0
    num_cats = 0
    num_sol = 0

    for rxn in rxn_list:
        num_reactants = max(len(rxn.reactants), num_reactants)
        num_products = max(len(rxn.products), num_products)
        num_reagents = max(len(rxn.reagents), num_reagents)
        num_cats = max(len(rxn.cats), num_cats)
        num_sol = max(len(rxn.solvents), num_sol)

    # 设置dataframe的索引columns
    cols = list()
    # 反应索引
    cols.append("rxn_index")
    # 反应物
    for i in range(num_reactants):
        cols.append("reactants %d" % (i + 1))
    # 产物
    for i in range(num_products):
        cols.append("products %d" % (i + 1))
    # 试剂
    for i in range(num_reagents):
        cols.append("reagents %d" % (i + 1))
    # 催化剂
    for i in range(num_cats):
        cols.append("catalysts %d" % (i + 1))
    # 溶剂
    for i in range(num_sol):
        cols.append("solvents %d" % (i + 1))
    # 温度
    cols.append("temperature /C")
    # 时间
    cols.append("time /h")
    # 产率
    cols.append("Yield")
    # Reaction ID
    cols.append("Reaction ID")
    # Citation
    cols.append("Reference")

    # 将数据导入dataframe
    data = list()
    for rxn in rxn_list:
        meta_data = list()
        # 索引
        meta_data.append(rxn.index)
        # 反应物
        for reactant in rxn.reactants:
            meta_data.append(reactant)
        while len(rxn.reactants) < num_reactants:
            meta_data.append("/")
            rxn.reactants.append("/")
        # 产物
        for product in rxn.products:
            meta_data.append(product)
        while len(rxn.products) < num_products:
            meta_data.append("/")
            rxn.products.append("/")
        # 试剂
        for reagent in rxn.reagents:
            meta_data.append(reagent)
        while len(rxn.reagents) < num_reagents:
            meta_data.append("/")
            rxn.reagents.append("/")
        # 催化剂
        for cat in rxn.cats:
            meta_data.append(cat)
        while len(rxn.cats) < num_cats:
            meta_data.append("/")
            rxn.cats.append("/")
        # 溶剂
        for sol in rxn.solvents:
            meta_data.append(sol)
        while len(rxn.solvents) < num_sol:
            meta_data.append("/")
            rxn.solvents.append("/")
        # 温度
        meta_data.append(rxn.temp)
        # 时间
        meta_data.append(rxn.time)
        # 产率
        meta_data.append(rxn.rxn_yield)
        # Reaction ID
        meta_data.append(rxn.rxn_id)
        # Reference
        meta_data.append(rxn.ref)

        # 将元数据放入数据中
        data.append(meta_data)

    # 生成dataframe
    df = pd.DataFrame(data, columns=cols)
    return df

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smi_checker(rxn):
    pd = True
    try:
        for reactant in rxn.reactants:
            if Chem.MolFromSmiles(reactant) == None:
                pd = False
                break
        for product in rxn.products:
            if Chem.MolFromSmiles(product) == None:
                pd = False
                break
        for reagent in rxn.reagents:
            if Chem.MolFromSmiles(reagent) == None:
                pd = False
                break
        for cat in rxn.cats:
            if Chem.MolFromSmiles(cat) == None:
                pd = False
                break
        for sol in rxn.solvents:
            if Chem.MolFromSmiles(sol) == None:
                pd = False
                break
    except:
        pd = False

    return pd

def Heck_filter1(rxn):

    # Convert into mol
    try:
        reactants_mol = [Chem.AddHs(Chem.MolFromSmiles(i)) for i in rxn.reactants]
        product_mol = Chem.AddHs(Chem.MolFromSmiles(rxn.products[0]))

        # check the change of the number of the atom
        num_reactants = np.array([i.GetNumAtoms() for i in reactants_mol]).sum()
        num_product = product_mol.GetNumAtoms()
        if num_reactants - num_product != 2: # HX
            if num_reactants - num_product != 9: # TfOH
                if num_reactants - num_product != 19: # TsOH
                    if num_reactants - num_product != 8: # AcOH
                        return False

        # In the reactants, there must be a halogen atom & a C=C group
        LG = ["F", "Cl", "Br", "I", "OS(=O)(=O)C(F)(F)F", "CC1=CC=C(C=C1)S(=O)(=O)O", "CC(=O)O"]
        patt_DoubleBond = Chem.MolFromSmarts('C=C')

        # Intramolecular rxn
        if len(rxn.reactants) == 1:
            pd = False

            # check C=C
            if reactants_mol[0].HasSubstructMatch(patt_DoubleBond) == False:
                return False

            # check LG
            for lg in LG:
                patt_lg = Chem.MolFromSmarts(lg)
                if reactants_mol[0].HasSubstructMatch(patt_lg):
                    pd = True
            return pd

        # Intermolecular rxn
        if len(rxn.reactants) == 2:
            pd1_lg = False
            pd2_lg = False
            # C=C
            pd1_db = reactants_mol[0].HasSubstructMatch(patt_DoubleBond)
            pd2_db = reactants_mol[1].HasSubstructMatch(patt_DoubleBond)

            # X
            for lg in LG:
                patt_lg = Chem.MolFromSmarts(lg)
                if reactants_mol[0].HasSubstructMatch(patt_lg):
                    pd1_lg = True
                if reactants_mol[1].HasSubstructMatch(patt_lg):
                    pd2_lg = True

            if int(pd1_lg) + int(pd2_lg) == 0 or int(pd1_db) + int(pd2_db) == 0:
                return False

        return True

    except:
        return False

def Heck_filter2(rxn):
    # Check whether there is Pd in the catalyst
    pd = False
    try:
        for reagent in rxn.reagents:
            if "Pd" in str(reagent):
                pd = True
        for cat in rxn.cats:
            if "Pd" in str(cat):
                pd = True

    except:
        pd = False

    return pd

from utils.molecule import *
from keras.preprocessing import text
from keras.utils import pad_sequences

def rxnfp_to_tensor(rxnfp, maxlen_, victor_size, file):
    vocab_dict = vocab_txt_to_dict(file) # get vocab_dict

    tokenizer = text.Tokenizer(num_words=100, lower=False, filters="　")
    tokenizer.fit_on_texts(rxnfp)
    smile_ = pad_sequences(tokenizer.texts_to_sequences(rxnfp), maxlen=maxlen_)
    word_index = tokenizer.word_index

    count = 0
    embedding_matrix = np.zeros((100, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector = vocab_dict[word] if word in vocab_dict else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    return torch.tensor(embedding_matrix, dtype=torch.float32)

if __name__ == "__main__":
    rxnfp = "[CLS] C N ( C ) P ( N ( C ) C ) ( N ( C ) C ) = N P ( N ( C ) C ) ( N ( C ) C ) = N C C . C C ( C ) C 1 = C C ( C ( C ) C ) = C C ( C ( C ) C ) = C 1 C 2 = C ( P ( C 3 C C C C C 3 ) C 4 C C C C C 4 ) C = C C = C 2 . F C ( F ) ( F ) c 1 c c c ( Cl ) c c 1 . o 1 n c c c 1 c 2 c c c c c 2 >> C c 1 c c c ( N c 2 c c c ( C ( F ) ( F ) F ) c c 2 ) c c 1 [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
    file = "vocab.txt"
    vec = rxnfp_to_tensor(rxnfp, 100, 100, file)
    print(vec, vec.shape)