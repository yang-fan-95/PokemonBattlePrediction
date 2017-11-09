import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('D:\workspace\github\PokemonBattlePrediction\data\pokemon.csv')

#1. INTRODUCTION TO PYTHON
'''
# how to import csv file
# plotting line,scatter and histogram
# basic dictionary features
# basic pandas features like filtering that is actually something always used and main for being data scientist
#While and for loops
'''
'''
# correlation map
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Attack',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

# Scatter Plot
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
plt.show()

# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (15,15))
plt.show()

# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50)
plt.clf()'''
'''
# create dictionary and look its keys and values
dictionary = {'spain':'madrid','usa':'vegas'}
print(dictionary.keys())
print(dictionary.values())

#Keys have to immutable objects likes string , boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barecelona"          # upadate existing entry
print(dictionary)
dictionary['france'] = 'paris'              # add new entry
print(dictionary)
del dictionary['spain']                     # remove entry with key
print(dictionary)
print('france' in dictionary)               # check include or not
dictionary.clear()                          # remove all entries in dict
print(dictionary)
'''
'''
# pandas
series = data['Defense']                    # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]              # data[['Defense']] = data frame
print(type(data_frame))

# 1 - Filtering Pandas data frame
x = data['Defense'] > 200
print(data[x])

# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
print(data[np.logical_and(data['Defense']>200,data['Attack']>100)])
print(data[(data['Defense']>200)&(data['Attack']>100)])
'''
'''
#while and for loops
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1
print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)
'''
#2. PYTHON DATA SCIENCE TOOLBOX
'''
# example of what we learn above
def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)

# guess print what
x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.

# How can we learn what is built in scope
import builtins
dir(builtins)

#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())

# default arguments
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(5))
# what if we want to change default arguments
print(f(5,4,3))

# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 123456)

# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))

number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))

# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration

# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))

# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)

# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
'''
'''
#   lets return pokemon cvs and make list comprehension example
#   lets classify pokemons whether they have high or low speed. Our threshould is average speed.
threshold = sum(data.Speed)/len(data.Speed)
data['speed_level'] = ['high' if i > threshold else 'low' for i in data.Speed]
print(data.loc[:10,['speed_level','Speed']])  #   we will learn loc more detailed later
'''
#3. CLEANING DATA
'''
print(data.head())      #   head shows first 5 rows
print(data.tail())      #   tail shows last 5  rows

#columns gives column names of features
print(data.columns)
# shape gives number og rows and columns in a tuble
print(data.shape)
# info gives data type like dataframe, number of sample or row , number of feature or column, feature types and memory usage
print(data.info())
'''
'''
EXPLOTARY DATA ANALYSIS

value_counts(): Frequency counts 
outliers: the value that is considerably higher or lower from rest of the data 
We will use describe() method. Describe method includes:

count: number of entries
mean: average of entries
std: standart deviation
min: minimum entry
25%: first quantile
50%: median or second quantile
75%: third quantile
max: maximum entry
'''
'''
# for example lets look frequency of pokemom type
print(data['Type 1'].value_counts(dropna = False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon

# For example max HP is 255 or min defense is 5
print(data.describe())
'''
'''
VISUAL EXPLORATORY DATA ANALYSIS
Box plots: visualize basic statistics like outliers, min/max or quantiles
'''
# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
'''
data.boxplot(column='Attack',by = 'Legendary')
'''
'''
# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = data.head()    # I only take 5 rows into new data
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
'''
'''
print(data.dtypes)
# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
print(data.dtypes)
'''
'''
# Lets chech Type 2
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN value
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false
assert  data1['Type 2'].notnull().all() # returns nothing because we drop nan values

data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
'''
