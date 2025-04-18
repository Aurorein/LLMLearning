import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn import model_selection

# sklearn生成数据集
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # CrossEntropy需要long类型
y_test = torch.tensor(y_test, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 损失函数
criterion = nn.CrossEntropyLoss()
epochs = 500

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).argmax(dim=1)  # 取概率最大的类别
        accuracy = (predictions == y_test).float().mean()
        print(f"Epoch: {epoch}, Test Accuracy: {accuracy.item() * 100:.2f}%")
