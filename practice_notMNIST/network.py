from torch import nn, optim
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_features=28*28,
        hidden_size=256,
        output_features=10,
        init_weight='he',
        init_bias='zero'
    ):
        super(Model, self).__init__() # super(Model, self) <- 파이썬3에서는 더이상 필요없는 argument
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.linear1 = nn.Linear(input_features, hidden_size) # input
        self.linear2 = nn.Linear(hidden_size, output_features) # sigle hidden layer
        self.init_params()

    def init_params(self):

        init_weight_method = {
            'he': nn.init.kaiming_normal_,
            'xavier': nn.init.xavier_normal_
        }
        assert (
            self.init_weight in init_weight_method.keys()
        ), f"Select the weight initialization method in {list(init_weight_method.keys())}"

        init_bias_method = {'zero': nn.init.zeros_, 'uniform': nn.init.uniform_}
        assert (
            self.init_bias in init_bias_method.keys()
        ), f"Select the bias initialization method in {list(init_bias_method.keys())}"

        for param_name, param in self.named_parameters():
            if 'weight' in param_name:
                init_weight_method[self.init_weight](param)
            elif 'bias' in param_name:
                init_bias_method[self.init_bias](param)

    def forward(self, X):
        X = F.relu((self.linear1(X))) # 첫번째 레이어에 대한 forward
        X = self.linear2(X) # 두번째 레이어에 대한 forward
        return X