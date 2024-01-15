from models.MMYield_module import *

class MMYield(nn.Module):
    def __init__(self, GNN_type, GNN_params, RNN_type, RNN_params, exp_num):
        super(MMYield, self).__init__()
        """
        GNN_type: types of GNN, used for reactants, products, catalysts, solvents
        GNN_params: dict type, params for GNN
        RNN_type: types of RNN, used for rxnfp
        RNN_params: dict type, params for RNN
        exp_num: numbers of experimental data
        """

        # # set learnable parameter
        # self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # initialize
        # self.w1.data.fill_(0.5)
        # self.w2.data.fill_(0.5)

        # input module
        self.rea, self.rea_output = module_selector(type=GNN_type, params=GNN_params)
        self.add, self.add_output = module_selector(type=GNN_type, params=GNN_params)
        self.rxn, self.rxn_output = module_selector(type=RNN_type, params=RNN_params)
        self.exp_num = exp_num

        # self.fc0 = nn.Linear(self.rea_output + self.add_output, self.rxn_output)

        # self.fc1 = nn.Linear(self.rxn_output+ self.exp_num, 500)
        self.fc1 = nn.Linear(self.rxn_output + self.exp_num + self.rea_output + self.add_output, 500)
        self.norm1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 1)


    def forward(self, x):
        rea_data, add_data, rxn_vec, exp_data = x
        gnn = torch.cat([self.rea(rea_data), self.add(add_data)], dim=1)
        # gnn = self.fc0(gnn)
        rnn = self.rxn(rxn_vec)
        x = torch.cat([gnn, rnn], dim=1)
        # x = self.w1 * gnn + self.w2 * rnn
        if self.exp_num != 0:
            x = torch.cat([x, exp_data], dim=1)
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x