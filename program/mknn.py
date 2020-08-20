import pandas as pd
from math import sqrt
from math import isnan
import numpy as np
import itertools
import random

#Input K-Value to use
k = input()
k = int(k)

#Read dataset
data = pd.read_csv('./dataset/dataset.csv', sep=';')

'''Settings for random sampling on data. Set 'None' if no randomize, set value with state value
to random by state or 'any' for random data with random state value
'''
def dataRandomSet(data, random=None):
    if random != None:
        data_latih = data.sample(n=int(0.7*len(data.index)), random_state=random).drop(['KELAS', 'NO'], axis=1)
        data_latih.index = sorted(data_latih.index)
        data_latih = data.loc[data_latih.index]
        data_latih = data_latih.drop(['KELAS', 'NO'], axis=1)
        testIndex =[i for i in data.index if i not in data_latih.index]
        data_uji = data.iloc[testIndex].drop(['KELAS', 'NO'], axis=1)
        kelas_uji = data['KELAS'].iloc[testIndex]
        kelas_latih = data['KELAS'].iloc[data_latih.index]
    elif random == 'any':
        rand = 0
        rand = random.randrange(0, 1000)
        print(rand)
        data_latih = data.sample(n=int(0.7*len(data.index)),random_state=rand).drop(['KELAS', 'NO'], axis=1)
        data_latih.index = sorted(data_latih.index)
        data_latih = data.loc[data_latih.index]
        data_latih = data_latih.drop(['KELAS', 'NO'], axis=1)
        testIndex =[i for i in data.index if i not in data_latih.index]
        data_uji = data.iloc[testIndex].drop(['KELAS', 'NO'], axis=1)
        kelas_uji = data['KELAS'].iloc[testIndex]
        kelas_latih = data['KELAS'].iloc[data_latih.index]
    else:
        data_latih = data.iloc[:70].drop(['KELAS', 'NO'], axis=1)
        data_uji = data.iloc[70:].drop(['KELAS', 'NO'], axis=1)
        kelas_latih = data['KELAS'].iloc[:70]
        kelas_uji = data['KELAS'].iloc[70:]
    return data_latih, data_uji, kelas_latih, kelas_uji

#Count distance between each train data using Euclidean Distance.
def jarakLatih(data):
    jarakDataLatih = pd.DataFrame(0, index=[i for i in data.index], columns = [i for i in data.index])
    for i in data.columns:
        for j in jarakDataLatih.index:
            for k in jarakDataLatih.columns:
                jarakDataLatih.loc[j, k] = sqrt(np.sum([(data.loc[j] - data.loc[k])**2]))

    return jarakDataLatih

'''Check the closest of a data based on jarakLatih(), and check the both class in each k. If same, then 
count the validity by averaging total of same two closest data with k value.'''
def validity(data, datalatih, kelas, k):
    arrayK = []
    for i in range(k):
        arrayK.append([])
    for i in range(len(arrayK)):
        arrayK[i] = pd.DataFrame(columns=[i+1, 'Data Terdekat', 'Kelas'], index=[data.index])
    nilaiMinSorted = pd.DataFrame(columns=[i for i in range(k)], index=data.index)
    indexJarakMin = {}

    jl2 = data.copy(deep=True)
    for i in jl2.index:
        jl2.loc[i,i] = 100
        indexJarakMin[i] = []

    for i in nilaiMinSorted.index:
        for j in nilaiMinSorted.columns:
            nilaiMinSorted.loc[i, j] = (sorted(jl2.loc[i])[j])
    for i in nilaiMinSorted.index:
        for j in nilaiMinSorted.columns:
            for l in jl2.columns:
                if jl2.loc[i, l] == nilaiMinSorted.loc[i,j]:
                    indexJarakMin[i].append(l)
    for i in nilaiMinSorted.index:
        indexJarakMin[i] = indexJarakMin[i][:k]
    for i in range(len(arrayK)):
        for j in jl2.index:
            for k in jl2.columns:
                arrayK[i].loc[j, i+1] = jl2.loc[j, indexJarakMin[j][i]]
                arrayK[i].loc[j, 'Data Terdekat'] = indexJarakMin[j][i]
    for i in range(len(arrayK)):
        for j in jl2.index:
            for k in jl2.columns:
                arrayK[i].loc[j, 'Kelas'] = kelas_latih[indexJarakMin[j][i]]
                
    validitas = pd.DataFrame(columns=[i for i in range(len(arrayK))], index=[jl2.index])
    validitas['sum'] = 0
    validitas['validity'] = 0

    for i in validitas.index:
        for j in range(len(arrayK)):
            if arrayK[j].loc[i, 'Kelas'] == kelas_latih.loc[i]:
                validitas.loc[i, j] = 1
            else:
                validitas.loc[i, j] = 0
        validitas.loc[i, 'sum'] = sum(validitas.loc[i])
        validitas.loc[i, 'validity'] = validitas.loc[i, 'sum'] / len(arrayK)
    return validitas

#Count distance between train data and test data using Euclidean Distance.
def jarakUji(datalatih, datauji):
    jarakDataUji = pd.DataFrame(0, index=[i for i in datalatih.index], columns = [i for i in datauji.index])
    for i in datauji.columns:
        for j in datalatih.index:
            for k in jarakDataUji.columns:
                jarakDataUji.loc[j, k] = sqrt(np.sum([(datalatih.loc[j] - datauji.loc[k])**2]))
    return jarakDataUji

# Count weight with W = Validity(index) * (1 / distance(train,test) + alpha)  
def weightVoting(jarakLatih, jarakUji, kelasLatih, validity):
    weightVoting = pd.DataFrame(index=[jarakLatih.index], columns=[i for i in range(len(jarakUji.columns))])
    for i in jarakLatih.index:
        weightVoting.loc[i, 'Kelas'] = kelasLatih[i]
    for i in jarakUji.index:
        for j in range(len(jarakUji.columns)):
            weightVoting.loc[i, j] = validity.loc[i, 'validity'] * (1 / (jarakUji.loc[i, jarakUji.columns[j]] + 0.5))
    return weightVoting

#Choose largest weight to determine class of test data and see the result.
def highestWeight(weightVoting, kelasUji, k):
    highestweight = []
    for i in range(len(weightVoting.columns)):
        if weightVoting.columns[i] != 'Kelas':
            highestweight.append([])
            highestweight[i]+=sorted(weightVoting.loc[:, i], reverse=True)[:k]
    hasil = pd.DataFrame(highestweight, columns = [i for i in range(k)], index=[i for i in range(len(highestweight))])
    hasil['Klasifikasi'] = ''
    for i in range(len(highestweight)):
        for j in weightVoting.index:
            if weightVoting.loc[j, i] == max(hasil.loc[i].drop('Klasifikasi')):
                    hasil.loc[i, 'Klasifikasi'] = weightVoting.loc[j, 'Kelas']
    kelas_test = kelas_uji.reset_index()
    del kelas_test['index']
    hasil['Kelas'] = kelas_test
    
    return hasil

# Evaluation using accuracy
def akurasi(hasil):
    trueClassifier = 0
    for i in hasil.index:
        if hasil.loc[i, 'Klasifikasi'] == hasil.loc[i, 'Kelas']:
            trueClassifier += 1
    akurasi = (trueClassifier / len(hasil) * 100)
    return akurasi

data_latih, data_uji, kelas_latih, kelas_uji = dataRandomSet(data)
jarakLatih = jarakLatih(data_latih)
validity = validity(jarakLatih, data_latih, kelas_latih, k)
jarakUji = jarakUji(data_latih, data_uji)
weightVoting = weightVoting(jarakLatih, jarakUji, kelas_latih, validity)
hasil = highestWeight(weightVoting, kelas_uji, k)
akurasi = akurasi(hasil)
print(akurasi)

#Write into txt
hasil.to_csv(r"hasil.txt", sep=';', index=False, mode='a')
file1 = open("hasil.txt","a")#append mode 
file1.write("Akurasi: ")
file1.write(str(akurasi))
file1.write("\n")
file1.write("\n")
file1.close() 