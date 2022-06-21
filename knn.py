## Implementação do KNN

# Dataset: https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

# Case: The dataset contains cases from a study that was conducted
# between 1958 and 1970 at the University of Chicago's Billings Hospital
# on the survival of patients who had undergone surgery for breast cancer.

import math

amostras = []

with open('haberman.data', 'r') as f:
    for linha in f.readlines():
        atributo = linha.replace('\n', '').split(',')
        amostras.append([int(atributo[0]), int(atributo[1]), int(atributo[2]), int(atributo[3])])

def info_dataset(amostras, verbose=True):
    if verbose:
        print('Total de amostras: %d' % len(amostras))
    rotulo1, rotulo2 = 0, 0
    for amostra in amostras:
        if amostra[-1] == 1:
            rotulo1 += 1
        else:
            rotulo2 += 1
    if verbose:
        print('Total rotulo 1: %d' % rotulo1)
        print('Total rotulo 2: %d' % rotulo2)
        print('')
    return [len(amostras), rotulo1, rotulo2]


#info_dataset(amostras)

# porcentagem dos dados de treinamento
p = 0.6

_, rotulo1, rotulo2 = info_dataset(amostras, False)
treinamento, teste = [], []
max_rotulo1, max_rotulo2 = int(p * rotulo1), int(p * rotulo2)
total_rotulo1, total_rotulo2 = 0, 0

for amostra in amostras:
    if (total_rotulo1 + total_rotulo2) < (max_rotulo1 + max_rotulo2):
        treinamento.append(amostra)
        if amostra[-1] == 1 and total_rotulo1 < max_rotulo1:
            total_rotulo1 += 1
        else:
            total_rotulo2 += 1
    else:
        teste.append(amostra)

#info_dataset(treinamento)
#info_dataset(teste)
#info_dataset(amostras)


def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)


# teste da distância euclidiana
v1 = [1, 2, 3]
v2 = [2, 1, 3]
dist_euclidiana(v1, v2)

def knn(treinamento, nova_amostra, K):
    dists, tam_treino = {}, len(treinamento)

    # calcula a distância euclidiana da nova amostra para todos
    # os outros exemplos do conjunto de treinamento
    for i in range(tam_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    # obtém as chaves (índices) dos k-vizinhos mais próximos
    k_vizinhos = sorted(dists, key=dists.get)[:K]

    # votação majoritária
    qtd_rotulo1, qtd_rotulo2 = 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1:
            qtd_rotulo1 += 1
        else:
            qtd_rotulo2 += 1

    if qtd_rotulo1 > qtd_rotulo2:
        return 1
    else:
        return 2


#print(teste[10])
#print(knn(treinamento, teste[10], 13))


acertos, K = 0, 39
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

print('Total de treinamento: %d' % len(treinamento))
print('Total de testes: %d' % len(teste))
print('Total de acertos: %d' % acertos)
print('Porcentagem de acertos: %.2f%%' % (100 * acertos / len(teste)))
