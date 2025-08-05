from datetime import datetime
import pandas as pd
import numpy as np
import math
import csv
import random
import statistics


num_users = 2101
dimentions = 20
top_N = 5


class User:
    train_artist = list()
    test_artist = list()
    mean_rates = 0
    
Users_list = []
for i in range (num_users):
    a = User()
    a.train_artist = list()
    a.train_artist_value = list()
    a.test_artist = list()
    a.test_artist_value = list()
    a.initial_predictions = list()
    a.neighbors = list()
    a.similarity = list()
    a.mean_rates = 0
    Users_list.append(a)


with open('train_artist.csv') as csvfile:
    readCSV_ratings = csv.reader(csvfile, delimiter=',')
    for row in readCSV_ratings:
        u = int(row[0])
        i = int(row[1])
        r = float(row[2])
        Users_list[u].train_artist.append(i)
        Users_list[u].train_artist_value.append(r)


with open('test_artist.csv') as csvfile:
    readCSV_ratings = csv.reader(csvfile, delimiter=',')
    for row in readCSV_ratings:
        u = int(row[0])
        i = int(row[1])
        r = float(row[2])
        Users_list[u].test_artist.append(i)
        Users_list[u].test_artist_value.append(r)


with open('initial_predictions.csv') as csvfile:
    readCSV_ratings = csv.reader(csvfile, delimiter=',')
    for row in readCSV_ratings:
        u = int(row[0])-1
        r = float(row[1])
        Users_list[u].initial_predictions.append(r)      

        
with open('neighbors_similarity.csv') as csvfile:
    readCSV_ratings = csv.reader(csvfile, delimiter=',')
    for row in readCSV_ratings:
        u = int(row[0])-1
        n = int(row[1])
        s = float(row[2])
        Users_list[u].neighbors.append(n) 
        Users_list[u].similarity.append(s)

        
        
empty = []
for u in range (num_users):
    Users_list[u].trainable = 0
    if Users_list[u].train_artist != empty:
        Users_list[u].trainable = 1
        

def Calc_Mean_Rates():
    for i in range(num_users):
        if Users_list[i].trainable == 1:
            sum = 0
            for j in range (len(Users_list[i].train_artist_value)):
                sum += Users_list[i].train_artist_value[j]
            Users_list[i].mean_rates = sum / len(Users_list[i].train_artist_value)


Calc_Mean_Rates()        
        
def Calc_fz():    
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].z = list()
            Users_list[u].fz = list()
            empty=[]
            for i in range (len(Users_list[u].test_artist)):
                item = Users_list[u].test_artist[i]
                b = 0
                for v in range (len(Users_list[u].neighbors)):
                    c = Users_list[u].neighbors[v]
                    if item in Users_list[c].train_artist:
                        b = b + Users_list[u].similarity[v]
                Users_list[u].z.append(b)
            if  Users_list[u].z != empty:   
                medain = statistics.median(Users_list[u].z)
            else:
                medain = 0
            for j in range (len(Users_list[u].test_artist)):
                item = Users_list[u].test_artist[j]
                d = 0
                for v in range (len(Users_list[u].neighbors)):
                    c = Users_list[u].neighbors[v]
                    if item in Users_list[c].train_artist:
                        d = d + Users_list[u].similarity[v]
                if (medain + d) != 0:
                    f = 1 - (medain / (medain + d))
                else:
                    f = 1
                Users_list[u].fz.append(f)          

        
def gama():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].gama = 0
            array = []
            empty = []           
            for i in range (len(Users_list[u].test_artist)):
                item = Users_list[u].test_artist[i]
                b = 0
                a = 0
                for v in range (len(Users_list[u].neighbors)):
                    c = Users_list[u].neighbors[v]
                    if item in Users_list[c].train_artist:
                        index = Users_list[c].train_artist.index(item)
                        sim = Users_list[u].similarity[v] 
                        b = b + sim
                        a = a + (sim * (pow((Users_list[c].train_artist_value[index] - Users_list[c].mean_rates - Users_list[u].initial_predictions[i] + Users_list[u].mean_rates) , 2))) 
                if b != 0:
                    v_u = a / b
                else:
                    v_u = 0
                array.append(v_u)
            if array != empty:    
                medai_v = statistics.median(array)
            else:
                medai_v = 0
            
            a = math.log(0.5 , math.e)
            if medai_v < 1:
                b = math.log(((1 - 0 - medai_v)/(1 - 0)), math.e)
            else:
                b = 0
            if(b != 0):
                gama = a / b
            else:
                gama = 0
            Users_list[u].gama = gama
        

def Calc_fv():
    gama()
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].fv = list()
            for i in range (len(Users_list[u].test_artist)):
                item = Users_list[u].test_artist[i]
                b = 0
                a = 0
                for v in range (len(Users_list[u].neighbors)):
                    c = Users_list[u].neighbors[v]
                    if item in Users_list[c].train_artist:
                        index = Users_list[c].train_artist.index(item)
                        sim = Users_list[u].similarity[v] 
                        b = b + sim
                        a = a + (sim * (pow((Users_list[c].train_artist_value[index] - Users_list[c].mean_rates - Users_list[u].initial_predictions[i] + Users_list[u].mean_rates) , 2))) 
                if b != 0:
                    v_u = a / b
                else:
                    v_u = 0
                f = pow(((1 - 0 - v_u )/(1 - 0)) , Users_list[u].gama)
                Users_list[u].fv.append(f)       


def reliability():
    Calc_fz()       
    Calc_fv()
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].reliability = list()
            for i in range (len(Users_list[u].test_artist)):
                a = Users_list[u].fz[i]
                b = Users_list[u].fv[i]
                reliable = pow( a * (pow(b , a)),(1 / (1 + a)))
                Users_list[u].reliability.append(reliable) 


reliability()




def final_ranks():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].final_ranks = list()
            for i in range (len(Users_list[u].test_artist)):
                predict = Users_list[u].reliability[i] * Users_list[u].initial_predictions[i]
                Users_list[u].final_ranks.append(predict)

final_ranks()


def Top_N_list():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].top_n = list()
            for i in range (top_N):
                mx = -1
                m = -1
                for k in range (len(Users_list[u].final_ranks)):
                    max_pred = Users_list[u].final_ranks[k]
                    max_pred = max_pred.real
                    if max_pred > mx:
                        if  Users_list[u].test_artist[k] not in  Users_list[u].top_n:
                            top_item = Users_list[u].test_artist[k]
                            mx = max_pred
                            m = k
                if m != -1:           
                    if Users_list[u].test_artist[m] not in  Users_list[u].top_n:
                        Users_list[u].top_n.append(top_item)





def prob():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].prob = list ()
            for i in range (len(Users_list[u].test_artist)):
                item = Users_list[u].test_artist[i]
                rand = random.random()
                if rand <= 0.7:
                    Users_list[u].prob.append(item)




def precision():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].precision = 0
            a = 0
            for i in range (len(Users_list[u].top_n)):
                item = Users_list[u].top_n[i]
                if item in Users_list[u].prob :
                    a = a + 1
            b = len(Users_list[u].top_n)
            if b != 0:
                Users_list[u].precision = a / b
            else:
                Users_list[u].precision = -1


def mean_precision():
    precision()
    a = 0
    n = 0
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            if Users_list[u].precision != -1: 
                a = a + (Users_list[u].precision)
                n = n+1
    avg_precision = a / n
    print ('precision :', avg_precision)
    return avg_precision  


def recall():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].recall = 0
            a = 0
            for i in range (len(Users_list[u].top_n)):
                item = Users_list[u].top_n[i]
                if item in Users_list[u].prob :
                    a = a + 1
            b = len(Users_list[u].prob)
            if b != 0:
                Users_list[u].recall = a / b
            else:
                Users_list[u].recall = -1

def mean_recall():
    recall()
    a = 0
    n = 0
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            if Users_list[u].recall != -1: 
                a = a + (Users_list[u].recall)
                n = n+1
    avg_recall = a / n
    print ('recall :', avg_recall)
    return avg_recall  


def f1():
    P = mean_precision()
    R = mean_recall()
    F1 = (2 * P *  R)/(P + R)
    print ('F1 :', F1) 


#calculate DCGmax for NDCG
dc = 0
for i in range (1,top_N):
    dc = dc + (1/(math.log((i+1),2)))
DCGmax = 1 + dc


def NDCG():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].ndcg = 0
            if len(Users_list[u].top_n) != 0:
                rel1 = 0
                item = Users_list[u].top_n[0]
                if item in Users_list[u].prob:
                    rel1 = 1
                dc = 0
                for i in range (1,len(Users_list[u].top_n)):
                    reli = 0
                    item = Users_list[u].top_n[i]
                    if item in Users_list[u].prob:
                        reli = 1
                    dc = dc + (reli / (math.log((i+1),2)))
                dg = rel1 + dc
                Ndg = dg / DCGmax
                Users_list[u].ndcg = Ndg
            else:
                Users_list[u].ndcg = -1            


def mean_NDCG():
    NDCG()
    a = 0
    n = 0
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            if Users_list[u].ndcg != -1:
                a = a + (Users_list[u].ndcg)
                n = n+1
    avg_NDCG = a/n
    print ('NDCG :', avg_NDCG)
    return avg_NDCG 


prob()

Top_N_list()

f1()

mean_NDCG()
