import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)
H_R = 5000
R = 0.012
N = 100
h = 0.0000038

batch_size = 64
input_dim = N
hidden_dim = N * 12
hidden_dim_1 = N * 4
output_dim = N
num_epochs = 200

sigma = np.array([round(i, 1) for i in np.arange(0.1, 1.01, 0.1)])
E = np.array(range(2000, 3001, 10))


model_name = 'SoftplusNet'
model_path = f'E:\\pythonProject4\\{model_name}.pt'
path = 'E:\\pythonProject4'


def approximation(sigma, N, R, h, E, H_R):
    # Инициализируем массивы для коэффициентов прогонки
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)
    F = np.zeros(N)

    # Вычисляем коэффициенты разностной схемы для каждой точки нашей дискретной сетки
    r = np.linspace(h, R, N)
    V = 1 / sigma

    for i in range(0, N):
        r_imh = r[i] - h / 2
        r_iph = r[i] + h / 2
        V_imh = (V)
        V_iph = (V)
        f_i = 2 * sigma * E ** 2
        A[i] = r[i] * V_iph
        B[i] = r_iph * V_iph
        C[i] = r_imh * V_imh + B[i]
        F[i] = -r[i] * f_i * h ** 2
    # Прогонка для нахождения значений H в каждой точке
    H = np.zeros(N)
    alpha = np.zeros(N)
    beta = np.zeros(N)

    beta[N - 1] = H_R

    for i in reversed(range(0, N - 1)):
        alpha[i] = A[i] / (C[i] - B[i] * alpha[i + 1])
        beta[i] = (B[i] * beta[i + 1] + F[i]) / (C[i] - B[i] * alpha[i + 1])

    H[0] = (beta[0] - alpha[0] * beta[1]) / (1 - alpha[0] * alpha[1])

    for i in range(1, N - 1):
        H[i + 1] = alpha[i + 1] * H[i] + beta[i + 1]
    return H, r


X = []
meanings = []
for s in sigma:
    for e in E:
        H, r = approximation(s, N, R, h, e, H_R)

        meanings.append([H])

        X.append([r])

features = np.array(X)/ np.max(np.abs(X))
forecast = np.array(meanings) / np.max(np.abs(meanings))

X_train, X_test, y_train, y_test = train_test_split(X, forecast, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


class ModelsMgmt:
    def save(self, model, path, model_name):
        torch.save(model.state_dict(), os.path.join(path, f'{model_name}.pt'))

    def load(self, path, model, model_name):
        model.load_state_dict(torch.load(os.path.join(path, f'{model_name}.pt')))
        return model


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += torch.isclose(outputs.float(), targets.float(), rtol=0.01).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def evaluate_loss_acc(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            correct += torch.isclose(output.float(), target.float(), rtol=0.01).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy


class SoftplusNet(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, hidden_dim_1=hidden_dim_1, output_dim=output_dim):
        super(SoftplusNet, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Softplus()
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Softplus()
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim_1, bias=True),
            nn.Softplus()
        )

        self.output_layer = nn.Linear(hidden_dim_1, output_dim, bias=True)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set = TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device))
train_loader = DataLoader(train_set, batch_size=batch_size)

test_set = TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float().to(device))
test_loader = DataLoader(test_set, batch_size=batch_size)

# Проверяем, есть ли сохраненные веса для модели
if os.path.exists(model_path):
    # Если веса есть, загружаем их в модель
    net = SoftplusNet()
    net.load_state_dict(torch.load(model_path))
else:
    # Если весов нет, создаем новую модель
    net = SoftplusNet()

net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

verbose = True

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate_loss_acc(net, criterion, test_loader, device)

    ModelsMgmt().save(net, path, model_name)

    if verbose:
        print(('Epoch [%d/%d], Loss (train/test) : %.4f/%.4f,' + 'Acc (train/test): %.4f/%.4f')
              % (epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc))
