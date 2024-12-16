import torch
import torch.nn as nn
import torch.optim as optim

# Define your neural network model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=5)

    def forward(self, x):
        return self.fc(x)

# Create an instance of your model
model = MyModel()

# Define the mask to choose which neurons to update
# For example, we want to update the first 3 neurons in the layer:
mask = torch.ones_like(model.fc.weight)
mask[:, :3] = 0  # Set the first 3 neurons to 0 to keep them fixed

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward and backward pass
inputs = torch.randn(1, 10)
outputs = model(inputs)
target = torch.randn(1, 5)
loss = criterion(outputs, target)

# for k,v in model.named_parameters():
#     if 'bias' not in k:
#         v.requires_grad = False
#     else:
#         v.requires_grad = True

# Backpropagation
loss.backward()

# Apply the mask to the gradients
model.fc.weight.grad *= mask

for k, v in model.named_parameters():
    print(k)
    print(v)
    print(v.grad)
    print(v.grad.shape)
    print('-----------------------')


# Update the model parameters using the optimizer
optimizer.step()









#
# import torch
# import torch.nn as nn
#
# class NN_Network(nn.Module):
#     def __init__(self,in_dim,hid,out_dim):
#         super(NN_Network, self).__init__()
#
#         # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=torch%20nn%20linear#torch.nn.Linear
#         self.linear1 = nn.Linear(in_dim,hid)
#         self.linear2 = nn.Linear(hid,out_dim)
#
#         # https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter
#         self.my_param_a = nn.Parameter(torch.zeros(3,3))
#
#         # https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_buffer#torch.nn.Module.register_buffer
#         self.my_buffer_1 = self.register_buffer("A", torch.zeros(3,3))
#         self.my_buffer_2 = self.register_buffer("B", torch.zeros(4,4))
#
#     def forward(self, input_array):
#         h = self.linear1(input_array)
#         y_pred = self.linear2(h)
#         return y_pred
#
# in_d = 5
# hidn = 2
# out_d = 3
# model = NN_Network(in_d, hidn, out_d)
#
# print("\nNAMED PARAMS")
# for name, param in model.named_parameters():
#      print("    ", name, "[", type(name), "]", type(param), param.size())
#
# print("\nNAMED BUFFERS")
# for name, param in model.named_buffers():
#      print("    ", name, "[", type(name), "]", type(param), param.size())
#
# print("\nSTATE DICT (KEYS ONLY)")
# for k, v in model.state_dict().items():
#    print("    ", k)
#
# print("\nSTATE DICT (KEYS VALUE PAIRS)")
# for k, v in model.state_dict().items():
#    print("        ", "(", k, "=>", v, ")")