import torch
from utils.rxn import *
from utils.molecule import *
from torch_geometric.loader import DataLoader
from models.MMYield import *
import time

# 1. import data
data = pd.read_excel("../../../data/Heck/Heck_rxnfp.xlsx")
data = data.sample(frac=1).reset_index(drop=True)
vocab_path = "../../../utils/vocab.txt"

# 2. build dataset & dataloader

# find max_len for rxnfp
max_len = -1
for batch in range(data.shape[0]):
    max_len = max(max_len, len(str(data.loc[batch]["rxnfp"]).split(" ")))

rxn_dataset = list()

rxn_list = df_to_rxn_list(data)
for batch in range(data.shape[0]):
    meta = list()
    rxn = rxn_list[batch]
    # rea
    meta.append(smis_to_graph(rxn.reactants + rxn.products))
    # add
    meta.append(smis_to_graph(rxn.reagents + rxn.cats + rxn.solvents))
    # rxnfp
    rxnfp_vec = rxnfp_to_tensor(rxnfp=data.loc[batch]["rxnfp"], maxlen_=max_len, victor_size=100, file=vocab_path)
    meta.append(rxnfp_vec)
    # exp
    meta.append(torch.tensor([rxn.temp, rxn.time], dtype=torch.float32))

    # yield
    meta.append(rxn.rxn_yield / 100)

    rxn_dataset.append(meta)

# split of train & test set
ratio = 0.7
batch_size = 768
batch = len(rxn_dataset)
train_set = rxn_dataset[0: int(ratio * batch)]
test_set = rxn_dataset[int(ratio * batch) + 1:]
# data_loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

test_RMSE = list()
test_R2 = list()
train_R2 = list()
train_RMSE = list()
pred = list()
true = list()

# 3. training of the model
# params
t = 150
lr = 5e-4
num_feature = 7
smi2vec_inputsize=100

# model
model = MMYield(
    GNN_type="GAT_module",
    GNN_params={"node_feature_num": num_feature,
                "channels": [50, 100],
                "heads": 3},
    RNN_type="smi2vec_module",
    RNN_params={"input_size":smi2vec_inputsize,
                "hidden_size":100,
                "num_layers":1,
                "output_size":100},
    exp_num=2
)
criterion = nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Training
print("Start training")

start = time.time() # record 1

for epoch in range(t):
    # Training
    st = time.time() # record 2
    global_loss = torch.tensor([0.])

    for data in train_loader:
        x = data[:-1]
        y = torch.unsqueeze(data[-1], dim=1)
        loss = criterion(model.forward(x), y)
        opti.zero_grad()
        loss.backward()
        opti.step()
        global_loss += loss.item()
    print("Training Time = %d, loss: %f" % (epoch + 1, global_loss / batch_size))

    # processing report
    print("\t Process : %.2f" %((epoch + 1) / t * 100), "%")
    remain_time = (time.time() - start) * (1 / ((epoch + 1) / t) - 1) / 3600
    print("\t Time required for one round: %.2f s" % (time.time() - st))
    print("\t Remainder: %.2f h" % remain_time)

    # record of loss during training
    # performance in train set
    with torch.no_grad():
        pred = list()
        true = list()
        for data in train_loader:
            x = data[:-1]
            tr = list(torch.unsqueeze(data[-1], dim=1).detach().numpy())
            pr = list(model.forward(x).detach().numpy())
            pred += pr
            true += tr
        train_RMSE.append(RMSE(np.array(pred), np.array(true)))
        train_R2.append(R2(np.array(pred), np.array(true)))

    # performance in test set
    with torch.no_grad():
        pred = list()
        true = list()
        for data in test_loader:
            x = data[:-1]
            tr = list(torch.unsqueeze(data[-1], dim=1).detach().numpy())
            pr = list(model.forward(x).detach().numpy())
            pred += pr
            true += tr
        test_RMSE.append(RMSE(np.array(pred), np.array(true)))
        test_R2.append(R2(np.array(pred), np.array(true)))

# 4.Evaluation
# Performance in train set
print("R2 of train set is:%.3f" % train_R2[-1])
print("RMSE of train set is: %.3f" % train_RMSE[-1])

# Performance in test set
print("R2 of test set is:%.3f" % test_R2[-1])
print("RMSE of test set is: %.3f" % test_RMSE[-1])

# 5.Figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=2, dpi=120, figsize=(10, 5))

# Training Fig
steps = np.linspace(1, t, t)
ax[0].plot(steps, train_R2, color=[236/255, 164/255, 124/255])
ax[0].plot(steps, test_R2, color=[117/255, 157/255, 219/255])

# Test set performance
ax[1].scatter(np.array(pred) * 100, np.array(true) * 100, alpha=0.7, marker=".")
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
ax[1].plot(x, y, linestyle="--", color="r")

# Beautify
ax[0].legend(["R2 value in train set", "R2 value in test set"], loc="upper left", prop={'size': 8})
ax[0].set_xlabel("Epoch", fontsize=10)
ax[0].set_ylabel("R2 value", fontsize=10)
ax[0].set_title("The R2 value of train & test set during training", fontsize=13)
ax[1].set_xlabel("Predicted Yield", fontsize=10)
ax[1].set_ylabel("Observed Yield", fontsize=10)
ax[1].set_title("Test set performance", fontsize=13)
fig.suptitle("MMYield for Heck dataset", fontsize=16)

plt.tight_layout()
plt.show()

# # # 6.save the model
# # # pd = input("save the model & Fig or not, input y/n")
# # # if pd == "y":
# # #     plt.savefig("../../../../figures/Exp1 intra graph module.svg")
# # #     torch.save(best_model, "")