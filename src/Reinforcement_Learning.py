from datetime import datetime
import pandas as pd
import numpy as np
import math
import csv
import random

start=datetime.now()

num_users = 2101
dimentions = 30
number_of_neighbors = 90
top_N = 5


class User:
    deep_artist = list()
    deep_trust = list()
    deep_tag = list()
    train_artist = list()
    test_artist = list()


Users_list = []
for i in range (num_users):
    a = User()
    a.deep_artist = list()
    a.deep_trust = list()
    a.deep_tag = list()
    a.train_artist = list()
    a.train_artist_value = list()
    a.test_artist = list()
    a.test_artist_value = list()
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


artist = pd.read_csv('train_user_artist.csv',encoding='ansi')
d_artist = artist.pivot_table(index='userID',columns='artistID',values='value').fillna(0)
artist_users = int(artist.userID.nunique())

deep_artist = pd.read_csv('deep_user_artist.csv',encoding='ansi')
deep_artist = deep_artist.drop(columns=['user_id'])

for u in range (num_users): 
    for j in range (artist_users):
        if d_artist.index[j] == u:
            for k in range (dimentions):
                Users_list[u].deep_artist.append(deep_artist.values[j,k])

empty = []

for u in range (num_users):
    if Users_list[u].deep_artist == empty:
        for k in range (dimentions):
                Users_list[u].deep_artist.append(0)



tag=pd.read_csv('user_tag.csv',encoding='ansi')
d_tag = tag.pivot_table(index='userID',columns='tagID',values='value').fillna(0)
tag_users = int(tag.userID.nunique())

deep_tag = pd.read_csv('deep_user_tag.csv',encoding='ansi')
deep_tag = deep_tag.drop(columns=['user_id'])

for u in range (num_users): 
    for j in range (tag_users):
        if d_tag.index[j] == u:
            for k in range (dimentions):
                Users_list[u].deep_tag.append(deep_tag.values[j,k])


for u in range (num_users):
    if Users_list[u].deep_tag == empty:
        for k in range (dimentions):
                Users_list[u].deep_tag.append(0)



trust=pd.read_csv('user_friends.csv',encoding='ansi')
d_trust = trust.pivot_table(index='userID',columns='friendID',values='value').fillna(0)
trust_users=int(trust.userID.nunique())

deep_trust = pd.read_csv('deep_user_trust.csv',encoding='ansi')
deep_trust = deep_trust.drop(columns=['user_id'])

for u in range (num_users): 
    for j in range (trust_users):
        if d_trust.index[j] == u:
            for k in range (dimentions):
                Users_list[u].deep_trust.append(deep_trust.values[j,k])


for u in range (num_users):
    if Users_list[u].deep_trust == empty:
        for k in range (dimentions):
                Users_list[u].deep_trust.append(0)


for u in range (num_users):
    Users_list[u].trainable = 1
    if Users_list[u].train_artist == empty:
        Users_list[u].trainable = 0


def Calculate_trust_similar (u,v):
    a = 0
    b = 0
    c = 0
    for i in range (dimentions):
        a = a + (Users_list[u].deep_trust[i] * Users_list[v].deep_trust[i])
        b = b + (pow((Users_list[u].deep_trust[i]),2))
        c = c + (pow((Users_list[v].deep_trust[i]),2))
    b = math.sqrt(b)
    c = math.sqrt(c)
    if b != 0 and c != 0:
        trust_sim = a / (b * c)
    else:
        trust_sim = 0
    return trust_sim


def Calculate_tag_similar (u,v):
    a = 0
    b = 0
    c = 0
    for i in range (dimentions):
        a = a + (Users_list[u].deep_tag[i] * Users_list[v].deep_tag[i])
        b = b + (pow((Users_list[u].deep_tag[i]),2))
        c = c + (pow((Users_list[v].deep_tag[i]),2))
    b = math.sqrt(b)
    c = math.sqrt(c)
    if b != 0 and c != 0:
        tag_sim = a / (b * c)
    else:
        tag_sim = 0
    return tag_sim



def Calculate_artist_similar (u,v):
    a = 0
    b = 0
    c = 0
    for i in range (dimentions):
        a = a + (Users_list[u].deep_artist[i] * Users_list[v].deep_artist[i])
        b = b + (pow((Users_list[u].deep_artist[i]),2))
        c = c + (pow((Users_list[v].deep_artist[i]),2))
    b = math.sqrt(b)
    c = math.sqrt(c)
    if b != 0 and c != 0:
        artist_sim = a / (b * c)
    else:
        artist_sim = 0
    return artist_sim



trust_sim_arr = np.zeros((num_users, num_users))
artist_sim_arr = np.zeros((num_users, num_users))
tag_sim_arr = np.zeros((num_users, num_users))



def silmilarity_list():
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            for v in range (num_users):
                if u != v:
                    if Users_list[v].trainable == 1:
                        trust_similarity = Calculate_trust_similar (u,v)
                        trust_sim_arr[u,v] = trust_similarity
                        artist_similarity = Calculate_artist_similar (u,v)
                        artist_sim_arr[u,v] = artist_similarity
                        tag_similarity = Calculate_tag_similar (u,v)
                        tag_sim_arr[u,v] = tag_similarity

silmilarity_list()



def final_similarity(w1,w2,w3):
    simi = []
    users = []
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].sim_final = list()
            Users_list[u].neighbors_final = list()
            for v in range (num_users):
                if u != v:
                    if Users_list[v].trainable == 1:
                        a = (w1 * trust_sim_arr [u][v]) + (w2 * artist_sim_arr [u][v]) + (w3 * tag_sim_arr [u][v])
                        b = w1 + w2 + w3
                        similarity = a / b
                        simi.append(similarity)
                        users.append(v)
            for j in range (number_of_neighbors):
                mx = 0
                for i in range (len(simi)):
                    if simi[i] > mx:
                        if users[i] not in Users_list[u].neighbors_final: 
                            mx = simi[i]
                            us = users[i]
                if us not in Users_list[u].neighbors_final: 
                    Users_list[u].sim_final.append(mx)
                    Users_list[u].neighbors_final.append(us)
            simi.clear()
            users.clear()   

min_loss = 50  
min_state = 50 

def prediction(w1,w2,w3):
    final_similarity(w1,w2,w3)
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].predictions = list()
            for i in range (len(Users_list[u].test_artist)):
                a = 0
                b = 0
                for v in range (len(Users_list[u].neighbors_final)):   
                    c = Users_list[u].neighbors_final[v]
                    if i in Users_list[c].train_artist:
                        index = Users_list[c].train_artist.index(i)
                        a = a + (Users_list[u].sim_final[v] * Users_list[c].train_artist_value[index])
                        b = b + Users_list[u].sim_final[v]
                    else:
                        b = b + Users_list[u].sim_final[v]
                if b != 0:
                    predict = a / b
                else:
                    predict = 0
                Users_list[u].predictions.append(predict)


#Creating function  to calculate loss or MAE  w1, w1, w3        
def loss (w1,w2,w3):
    prediction(w1,w2,w3)
    empty = []
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            Users_list[u].loss_value = 0
            if Users_list[u].test_artist != empty:
                a = 0
                n = 0
                for i in range (len(Users_list[u].test_artist)):
                    #if Users_list[u].predictions[i] != 0: 
                        real = Users_list[u].test_artist_value[i]
                        pred = Users_list[u].predictions[i]
                        a = a + (pow((real - pred),2))
                        n = n+1
                if n != 0:
                    mean_squre_error = a / n
                    Users_list[u].loss_value = mean_squre_error
                else:
                    mean_squre_error = 0
                    Users_list[u].loss_value = mean_squre_error
    sum_mae = 0
    sum_user = 0                
    for u in range (num_users):
        if Users_list[u].trainable == 1:
            if Users_list[u].loss_value != 0:
                sum_mae = sum_mae + (Users_list[u].loss_value)
                sum_user = sum_user + 1
    return sum_mae / sum_user            



#Create Matrix State
weight1 = 0.05
weight2 = 0.05
weight3 = 0.05 
s = []
state = []
for i in range (20):
    for j in range (20):
        for k in range (20):
            s.append(round(weight1, 2))
            s.append(round(weight2, 2))
            s.append(round(weight3, 2))
            state.append(s)
            del(s)
            s = []
            weight3 += 0.05
        weight2 += 0.05
        weight3 = 0.05
    weight1 += 0.05
    weight2 = 0.05
 
    
    
#implementation algorithm for finding number state
def num_state(a,b,c):
    a = (a / 0.05) - 1
    b = (b / 0.05) - 1
    c = (c / 0.05) - 1
    num = (a * 20 * 20) + (b * 20) + c
    return int(num) 




#Create Matrix Q
rows, cols = (6, 8000) 
Q=[] 
for i in range(cols): 
    col = [] 
    for j in range(rows): 
        col.append(0) 
    Q.append(col)


#implemetation greecy_policy
def greedy_policy(state_m):
    epsilon = 0.1
    rand= random.uniform(0,1)
    if rand <= epsilon:
        num_action = random.uniform(0,5)
        num_action = int(num_action)
    else:
        max_q = Q[state_m][0]
        num_action = 0
        for i in range (6):
            if Q[state_m][i] > max_q:
                max_q = Q[state_m][i]
                num_action = i
    return num_action   



#Create Loss_Array
loss_array = []
for i in range (cols):
    loss_array.append(0)



#6 state Action 
def action(act):
    if act == 0:
        if (w1 + 0.05) <= 1:
            action = np.array([w1 + 0.05, w2, w3])
        else:
            action = np.array([w1 , w2, w3])
    elif act == 1:
        if (w1 - 0.05) >= 0.05:
            action = np.array([w1 - 0.05, w2, w3])
        else:
            action = np.array([w1 , w2, w3])
    elif act == 2:
        if (w2 + 0.05) <= 1:
            action = np.array([w1, w2 + 0.05, w3])
        else:
            action = np.array([w1 , w2, w3])
    elif act == 3:
        if (w2 - 0.05) >= 0.05:
            action = np.array([w1, w2 - 0.05, w3])
        else:
            action = np.array([w1 , w2, w3])
    elif act == 4:
        if (w3 + 0.05) <= 1:
            action = np.array([w1, w2, w3 + 0.05])
        else:
            action = np.array([w1 , w2, w3])
    elif act == 5:
        if (w2 - 0.05) >= 0.05:
            action = np.array([w1, w2, w3 - 0.05])
        else:
            action = np.array([w1 , w2, w3])
    return action


#calculate reward 
def reward(state_m,action_j,R=0):
    global min_loss
    global min_state
    w1 = state[state_m][0]
    w2 = state[state_m][1]
    w3 = state[state_m][2]
    loss_i = 0
    loss_j = 0
    if loss_array[state_m] == 0:
        loss_i = loss(w1,w2,w3)
        loss_array[state_m] = loss_i
        if loss_i < min_loss:
            min_loss = loss_i
            min_state = state_m
    elif loss_array[state_m] != 0:
        loss_i = loss_array[state_m]
    act = []
    act = action(action_j)
    w1 = act[0]
    w2 = act[1]
    w3 = act[2]
    state_m_plus = num_state(w1,w2,w3)
    if loss_array[state_m_plus] == 0:
        loss_j = loss(w1,w2,w3)
        loss_array[state_m_plus] = loss_j
        if loss_j < min_loss:
            min_loss = loss_j
            min_state = state_m_plus
    elif loss_array[state_m_plus] != 0:
        loss_j = loss_array[state_m_plus]        
    if loss_i < loss_j:
        R = -1 + loss_i - loss_j
    elif loss_i > loss_j:
        R = +1 + loss_i - loss_j
    return R


#Update Q
episode = 5
iterator = 20
landa = 0.1
gama = 0.1
for ep in range(episode):
    state_m = 0
    for i in range (iterator):
        w1 = state[state_m][0]
        w2 = state[state_m][1]
        w3 = state[state_m][2]
        max_act = greedy_policy(state_m)
        q_m = Q[state_m][max_act]
        Rew = reward(state_m , max_act)
        w1 = state[state_m][0]
        w2 = state[state_m][1]
        w3 = state[state_m][2]
        act = []
        act = action(max_act)
        w1 = act[0]
        w2 = act[1]
        w3 = act[2]
        state_m_plus = num_state(w1,w2,w2)
        max_q_plus = Q[state_m_plus][0]
        for j in range (6):
            if Q[state_m_plus][j] > max_q_plus:
                max_q_plus = Q[state_m_plus][j] 
        Q[state_m][max_act] = q_m + (gama * (Rew + (landa * max_q_plus) - q_m))
        state_m = state_m_plus
        #print(state_m)
    gama=gama-0.01    
        
          
print(min_state)

w1 = state[min_state][0]
w2 = state[min_state][1]
w3 = state[min_state][2]

print(min_loss)
print(w1 , w2 , w3)

prediction(w1,w2,w3)

f = open("initial_predictions.txt", "a")        
for u in range (num_users):
    if Users_list[u].trainable == 1:
        for i in range (len(Users_list[u].predictions)):
          f.write("%d\t" %((u)+1) )  
          f.write("%f\n" %((Users_list[u].predictions[i])))
f.close()

f = open("neighbors_similarity.txt", "a")        
for u in range (num_users):
    if Users_list[u].trainable == 1:
        for i in range (len(Users_list[u].neighbors_final)):
          f.write("%d\t" %((u)+1) )  
          f.write("%d\t" %((Users_list[u].neighbors_final[i])))
          f.write("%f\n" %((Users_list[u].sim_final[i])))
f.close()

print (datetime.now() - start)







   

