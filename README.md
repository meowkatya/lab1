# Практическая работа 1
Выполнила: Кузнецова Екатерина группа R3235
Преподаватель: Каканов М.А., Евстафьев М.А.
# Цель работы
Необходимо по данным с мобильных сенсоров при помощи прикладных алгоритмов машинного обучения предсказать активность человека по шести классам движений:
 - Движется по прямой;
 - Движется вверх (например, движение по лестнице вверх);
 - Движется вниз (например, движение по лестнице вниз);
 - Сидит;
 - Стоит;
 - Лежит.
# Ход работы
```python
# Импортируем библиотеки
import os
import numpy as np
import pandas as pd


# Считываем набор данных
def read_data(path, filename):
return pd.read_csv(os.path.join(path, filename))

df = read_data('/content/data/notebook_files', 'train.csv')
df.head()


#Загружаем полный набор данных и сохраняем его под четырьмя переменными: train_X, train_y, test_X, test_y
def load_dataset(label_dict):
train_X = read_data('/content/data/notebook_files', 'train.csv').values[:,:-2]
train_y = read_data('/content/data/notebook_files', 'train.csv')['Activity']
train_y = train_y.map(label_dict).values
test_X = read_data('/content/data/notebook_files', 'test.csv').values[:,:-2]
test_y = read_data('/content/data/notebook_files', 'test.csv')
test_y = test_y['Activity'].map(label_dict).values
return(train_X, train_y, test_X, test_y)
label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}
train_X, train_y, test_X, test_y = load_dataset(label_dict)


# Импортируем модели из библиотеки
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Инициализация моделей
SVCmodel = SVC()
LRmodel = LogisticRegression(solver='liblinear')


# Обучаем модели
SVCmodel.fit(train_X, train_y)
LRmodel.fit(train_X, train_y)


# Прогнозируем модели
SVCyhat = SVCmodel.predict(test_X)
LRyhat = LRmodel.predict(test_X)


#Оценка моделей
from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
print('SVC model')
print(classification_report(test_y, SVCyhat, target_names=target_names))
print('LogisticRegression model')
print(classification_report(test_y, LRyhat, target_names=target_names))
```

# Результат вывода программы
