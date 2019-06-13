import torch
from torch.nn import functional as F
# unsqueeze 是把向量变成矩阵。
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = x.pow(2) 

'''
构建一个回归的神经网络，包含一个隐藏层
'''
class SimpleNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(SimpleNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        hidden_result = self.hidden(x)
        x = F.relu(hidden_result)
        x = self.predict(x)
        return x

mynet = SimpleNet(1, 10, 1)
print(mynet)

optimizer = torch.optim.SGD(mynet.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

# ------ training ---------  
for epoch in range(2000):
    optimizer.zero_grad()

    # forward + backward + optimize
    pred = mynet(x)
    loss = loss_func(pred, y)
    loss.backward()
    optimizer.step()
    
    if(epoch%100 ==0): print(loss.data)

# ------ prediction --------- 
test_data = torch.tensor([-1.0])
pred = mynet(test_data)
print(test_data, pred.data)