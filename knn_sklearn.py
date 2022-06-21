## Implementação do KNN com sklearn

# Dataset: https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

# Case: The dataset contains cases from a study that was conducted
# between 1958 and 1970 at the University of Chicago's Billings Hospital
# on the survival of patients who had undergone surgery for breast cancer.

from sklearn.neighbors import KNeighborsClassifier

entradas, saidas = [], []

with open('haberman.data', 'r') as f:
    for linha in f.readlines():
        atributo = linha.replace('\n', '').split(',')
        entradas.append([int(atributo[0]), int(atributo[2])])
        saidas.append(int(atributo[3]))


# porcentagem dos dados de treinamento
p = 0.6

limite = int(p * len(entradas))
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(entradas[:limite], saidas[:limite])
labels = knn.predict(entradas[limite:])
acertos, indice_label = 0, 0
for i in range(limite, len(entradas)):
    if labels[indice_label] == saidas[i]:
        acertos += 1
    indice_label += 1

print('Total de treinamento: %d' % limite)
print('Total de testes: %d' % (len(entradas) - limite))
print('Total de acertos: %d' % acertos)
print('Porcentagem de acertos: %.2f' % (100 * acertos / (len(entradas) - limite)))
