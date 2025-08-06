import csv
import random
import pandas as pd

num_users = 2101

#creat lists of artists and number of artists
arti = []
cont_arti = []
with open('user_taggedartists.csv') as csvfile:
    readCSV_tags = csv.reader(csvfile, delimiter=',')
    for row in readCSV_tags:
        a = int(row[1])
        if a in arti:
            index = arti.index(a)
            cont_arti[index] += 1
        if a not in arti:
            arti.append(a)
            cont_arti.append(1)

#recognition main artists(num_artist > 20) 
main_arti = []
for i in range (len(cont_arti)):
    if cont_arti[i] >= 20:
        art = arti[i]
        main_arti.append(art)

#creat lists of taggs and number of taggs
tag = []
cont_tag = []
with open('user_taggedartists.csv') as csvfile:
    readCSV_tags = csv.reader(csvfile, delimiter=',')
    for row in readCSV_tags:
        t = int(row[2])
        if t in tag:
            index = tag.index(t)
            cont_tag[index] += 1
        if t not in tag:
            tag.append(t)
            cont_tag.append(1)

#recognition main taggs(num_tag > 5) 
main_tag = []
for i in range (len(cont_tag)):
    if cont_tag[i] >= 5:
        tg = tag[i]
        main_tag.append(tg)
        

class User:
    tag = list()
    artist = list()


#creat lists for each user
Users_list = []
for u in range (num_users):
    a = User()
    a.tag = list()
    a.artist = list()
    Users_list.append(a)


#creat artist list and tag list using main artist and main tag
with open('user_taggedartists.csv') as csvfile:
    readCSV_tags = csv.reader(csvfile, delimiter=',')
    for row in readCSV_tags:
        u = int(row[0])
        a = int(row[1])
        t = int(row[2])
        if a in main_arti:
            if a not in Users_list[u].artist:
                Users_list[u].artist.append(a)
        if t in main_tag:
            if t not in Users_list[u].tag:
                Users_list[u].tag.append(t)


#recognition userID that not exist artist
empty = []           
for u in range (num_users):
    Users_list[u].artist_exist = 0
    if Users_list[u].artist != empty:
        Users_list[u].artist_exist = 1
        
#prepare train set and test set            
for u in range (num_users):
    if Users_list[u].artist_exist == 1:
        Users_list[u].train_artist = list()
        Users_list[u].test_artist = list()
        for i in range (len(Users_list[u].artist)):
            rand = random.random()
            artt = Users_list[u].artist[i]
            if rand <= 0.8:
                Users_list[u].train_artist.append(artt)
            else:
                Users_list[u].test_artist.append(artt)
                
#if train is empty then insert one sample of test to train         
for u in range (num_users):
    if Users_list[u].artist_exist == 1:
        if Users_list[u].train_artist == empty:
            art = Users_list[u].test_artist[0]
            Users_list[u].train_artist.append(art)
            Users_list[u].test_artist.pop(0)
        
f = open("train_artists.txt", "a") 
for u in range (num_users):
    if Users_list[u].artist_exist == 1:
        for i in range (len(Users_list[u].train_artist)):
            f.write("%d\t" %((u)))
            f.write("%d\t" %((Users_list[u].train_artist[i])))
            value = 1
            f.write("%d\n" %((value)))
f.close()


f = open("test_artists.txt", "a") 
for u in range (num_users):
    if Users_list[u].artist_exist == 1:
        for i in range (len(Users_list[u].test_artist)):
            f.write("%d\t" %((u)))
            f.write("%d\t" %((Users_list[u].test_artist[i])))
            value = 1
            f.write("%d\n" %((value)))
f.close()



#recognition userID that not exist tag
empty = []           
for u in range (num_users):
    Users_list[u].tag_exist = 0
    if Users_list[u].tag != empty:
        Users_list[u].tag_exist = 1
        


f = open("user_tag.txt", "a") 
for u in range (num_users):
    if Users_list[u].tag_exist == 1:
        for i in range (len(Users_list[u].tag)):
            f.write("%d\t" %((u)))
            f.write("%d\t" %((Users_list[u].tag[i])))
            value = 1
            f.write("%d\n" %((value)))
f.close()




