from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()

        # Implicit input layer (# of node = 8).

        # First hidden layer.
        self.hidden = nn.Linear(state_dim, hidden_dim)

        # Output layer.
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)


# if __name__ == '__main__':
#     state_dim = 8
#     action_dim = 2
#     net = DQN(state_dim, action_dim)
#
#     device = 'cpu'
#     net = net.to(device)
#
#     state = [2, 0, 0, 0, 0, 0, 0, 1]
#     state = torch.tensor(state, dtype=torch.float, device=device)
#     print(state)
#
#     output = net(state)
#     print(output)
