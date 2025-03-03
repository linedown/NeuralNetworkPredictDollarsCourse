# Подключение необходимых библиотек
import torch
import torch.nn as nn
import torch.optim as optim
import load_excel_data as led
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Создание и описание класса нейросети
class NeuralNetwork(nn.Module):
    def __init__(self, neurons):
        super(NeuralNetwork, self).__init__()
# Количество входных данных – 3, нейронов в скрытых слоях – указывается при создании экземпляра класса
# Количество выходных данных – 1. Функция активации – гиперболический тангенс
        self.fc1 = torch.nn.Linear(3, neurons)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(neurons, neurons)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(neurons, 1)
    # Процесс передачи входных данных через нейросеть от входного слоя к выходному слою
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
# Оценка точности нейросети в процессе обучения
def getAccuracy(model, xTest, yTest):
    yPred = model(xTest)
    n = int(yPred.shape[0])
    acc = 0.0
    for i in range(n):
        acc += (yPred[i].round() == yTest[i].round()).sum()
        print(yPred[i].detach().item(), yTest[i].detach().item())
    return acc.detach().item() / n
# Отображение графиков с исходными данными и данными, полученными после обучения нейросети
def displayPoints(net, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    yPred = net.forward(x)
    # x1, x2, x3 – входные значения. Числа берутся из тензоров, полученных из excel-файла с помощью
# модуля load_excel_data
    x1 = x[:,0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    y0 = y[:, 0]
# Отображение точек на графике. c – параметр цвета, зависит от выходных значений функции
    img1 = ax.scatter(x1, x2, x3, c = y0)
# Установка динамической шкалы
    fig.colorbar(img1)
    ax = fig.add_subplot(122, projection='3d')
    y1 = yPred[:, 0]
    print(y1)
    img2 = ax.scatter(x1, x2, x3, c = y1.detach().numpy())
    fig.colorbar(img2)
# Показ графиков в отдельном окне
    plt.show()
# Задание начального размера датасета
ds_size = 100000
# Получение данных
dataset = led.load_dataset('dataset.xlsx', ds_size)
# Преобразование массивов значений в тензор библиотеки torch необходимой длины
# Обучающая выборка – 49900 точек
X_train = torch.from_numpy(dataset[0][:19900]).to(torch.float32)
y_train = torch.from_numpy(dataset[1][:19900]).to(torch.float32)
# Тестовая – 100 точек

X_test = torch.from_numpy(dataset[0][99900:]).to(torch.float32)
y_test = torch.from_numpy(dataset[1][99900:]).to(torch.float32)
# Создание экземпляра класса нейросети со 150 нейронами
model = NeuralNetwork(150)
# Используемая функция потерь – среднеквадратичной ошибки 
criterion = nn.MSELoss()
# Оптимизатор – Adam со временем обучения 0.01 и заданными параметрами внутри класса сети
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Обучение модели на 5000 эпохах (шагах) обучения
for i in range(5000):
# Обнуление градиента, передача данных через нейросеть, вызов функции потерь, подсчёт производных
    optimizer.zero_grad()
    yPred = model.forward(X_train)
    loss = criterion(yPred, y_train)
    loss.backward()
    optimizer.step()
# Оценка модели
model.eval()
# Вывод графиков результатов
displayPoints(model, X_test, y_test)
# Подсчёт и вывод точности работы нейросети после обучения
acc = getAccuracy(model, X_test, y_test)
print(f"Средняя ошибка: No,\tТочность: {acc}")
