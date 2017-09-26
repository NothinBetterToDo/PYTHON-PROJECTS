import matplotlib.pyplot as plt
from matplotlib import style
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv("Breast-cancer-wisconsin-1.csv", na_values=["?"]) #take first col as index
df.drop([col for col in df.columns if 'Unnamed' in col], axis=1, inplace=True)
    
#Replace missing value with column mean 
df['A7'].fillna((df['A7'].mean()), inplace=True)
        
#Drop Scn and Class(Cluster=2 or 4) 
X = np.array(df.drop(['Scn', 'CLASS'], axis=1).astype(float))


def histogram():
    #Plot histograms for A2 to A10 - total 9 histograms 
    #A2 histogram 
    fig1 = plt.figure()
    sp1 = fig1.add_subplot(1,1,1)
    ticks = sp1.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp1.set_title('Clump thickness')
    sp1.set_xlabel('A2')
    sp1.set_ylabel('Frequency')
    print(sp1.hist(df['A2'], bins=20, color='b', alpha=0.5))
    
    ##A3 histogram 
    fig2 = plt.figure()
    sp2 = fig2.add_subplot(1,1,1)
    ticks = sp2.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp2.set_title('Uniformity of Cell Size')
    sp2.set_xlabel('A3')
    sp2.set_ylabel('Frequency')
    print(sp2.hist(df['A3'], bins=20, color='b', alpha=0.5))
    
    ##A4 histogram 
    fig3 = plt.figure()
    sp3 = fig3.add_subplot(1,1,1)
    ticks = sp3.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp3.set_title('Uniformity of Cell Shape')
    sp3.set_xlabel('A4')
    sp3.set_ylabel('Frequency')
    print(sp3.hist(df['A4'], bins=20, color='b', alpha=0.5))
    
    ##A5 histogram 
    fig4 = plt.figure()
    sp4 = fig4.add_subplot(1,1,1)
    ticks = sp4.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp4.set_title('Marginal Adhesion')
    sp4.set_xlabel('A5')
    sp4.set_ylabel('Frequency')
    print(sp4.hist(df['A5'], bins=20, color='b', alpha=0.5))
    
    ##A6 histogram 
    fig5 = plt.figure()
    sp5 = fig5.add_subplot(1,1,1)
    ticks = sp5.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp5.set_title('Single Epithelial Cell Size')
    sp5.set_xlabel('A6')
    sp5.set_ylabel('Frequency')
    print(sp5.hist(df['A6'], bins=20, color='b', alpha=0.5))
    
    ##A7 histogram 
    fig6 = plt.figure()
    sp6 = fig6.add_subplot(1,1,1)
    ticks = sp6.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp6.set_title('Bare Nuclei')
    sp6.set_xlabel('A7')
    sp6.set_ylabel('Frequency')
    print(sp6.hist(df['A7'], bins=20, color='b', alpha=0.5))
    
    ##A8 histogram 
    fig7 = plt.figure()
    sp7 = fig7.add_subplot(1,1,1)
    ticks = sp7.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp7.set_title('Bland Chromatin')
    sp7.set_xlabel('A8')
    sp7.set_ylabel('Frequency')
    print(sp7.hist(df['A8'], bins=20, color='b', alpha=0.5))
    
    ##A9 histogram 
    fig8 = plt.figure()
    sp8 = fig8.add_subplot(1,1,1)
    ticks = sp8.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp8.set_title('Normal Nucleoli')
    sp8.set_xlabel('A9')
    sp8.set_ylabel('Frequency')
    print(sp8.hist(df['A9'], bins=20, color='b', alpha=0.5))
    
    ##A10 histogram 
    fig9 = plt.figure()
    sp9 = fig9.add_subplot(1,1,1)
    ticks = sp9.set_xticks([1,2,3,4,5,6,7,8,9,10])
    sp9.set_title('Mitoses')
    sp9.set_xlabel('A10')
    sp9.set_ylabel('Frequency')
    print(sp9.hist(df['A10'], bins=20, color='b', alpha=0.5))



def statistics(): 
    col_mean=df[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].mean()
    col_median=df[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].median()
    col_std=df[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].std()
    col_var =df[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].var()

    print('\tMean')
    print(col_mean, '\n')
    print('\tMedian')
    print(col_median, '\n')
    print('\tStandard Deviation')
    print(col_std, '\n')
    print('\tVariance')
    print(col_var, '\n')
    


def Initialization():

    mu2 = X[np.random.choice(X.shape[0],1, replace=False)] #without replacement 
    mu4 = X[np.random.choice(X.shape[0],1, replace=False)] #without replacement
 
    return mu2,mu4
    
                    

def Assignment(X,mu2,mu4):

    
    mu2_distance = euclidean_distances(X, mu2)
    mu4_distance = euclidean_distances(X, mu4)
    
    df['mu2_distance'] = mu2_distance
    df['mu4_distance'] = mu4_distance
        
    def f(df):
            if df['mu4_distance'] < df['mu2_distance']:
                val=4
            else:
                val=2
            return val
    
    df['predicted_class'] = df.apply(f, axis=1)
    

    
def Recalculation():     
       
    #Cluster 2 Grouping Dataset 
    d2 = df[df['predicted_class']==2] 
    cluster2 = d2.drop(['Scn', 'CLASS', 'mu2_distance', 'mu4_distance', 'predicted_class'], axis = 1) 
    cluster2_mean =cluster2[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].mean()
    mu2 = np.array(cluster2_mean)
        
    #Cluster 4 Grouping Dataset 
    d4 = df[df['predicted_class']==4]
    cluster4 = d4.drop(['Scn', 'CLASS', 'mu2_distance', 'mu4_distance', 'predicted_class'], axis = 1)         
    cluster4_mean =cluster4[['A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']].mean()        
    mu4 = np.array(cluster4_mean)
    
    return mu2,mu4


def classified_incorrect(df):
    if df['predicted_class'] != df['CLASS']:
        val=1
    else:
        val=0
    return val
    

def get_error_rate():
    df['classified_incorrect'] = df.apply(classified_incorrect, axis=1)
    a = df[(df['predicted_class'] == 4) & (df['CLASS']==2 )]
    error_mu2 = a.count()['classified_incorrect']
    total_PC2 = df[df["predicted_class"]==2].count()["predicted_class"]
#    print('Total classified incorrect for mu2:', error_mu2)
#    print('Total predicted cluster 2:', total_PC2)
    error_B = error_mu2 / total_PC2 
    print('Error B', error_B)

    b = df[(df['predicted_class'] == 2) & (df['CLASS']==4)]
    error_mu4 = b.count()['classified_incorrect']
    total_PC4 = df[df["predicted_class"]==4].count()["predicted_class"]
#    print('Total classified incorrect for mu4:', error_mu4)
#    print('Total predicted cluster 4:', total_PC4)
    error_M = error_mu4 / total_PC4 
    print('Error M', error_M)
    
    total_error_rate = error_B + error_M
    print('Total error rate', total_error_rate)
    



def main():
    
    ###############phase 1################################    

    histogram()
    statistics()

    ###############phase 2################################
        
    
    mu2,mu4 = Initialization()
    
            
    for i in range(1500):
    
        Assignment(X,mu2,mu4)
        mu2,mu4 = Recalculation()
    
    Assignment(X,mu2,mu4)
    
    print("*********************Final mean*********************")
    print("mu2:", mu2)
    print("mu4:", mu4)
    print("")
    print("")
    print("*********************Cluster assignment*********************")
    print(df[['Scn', 'CLASS', 'predicted_class']][0:21])
    
    
    ###############phase 3################################


    get_error_rate()


main()
