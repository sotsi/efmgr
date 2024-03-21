#!/usr/bin/env python
# coding: utf-8

# In[1]:


# gather the BCx_y datasets...
import glob
import pandas as pd
import numpy as np
bc_appl = glob.glob('ma43-*final.csv') # pv generation for horizontal/East system
print(bc_appl)


# In[2]:


pv1 = pd.read_csv(bc_appl[0],header=0)
pv1['Datetime'] = pd.to_datetime(pv1.time, format='%Y%m%d:%H%M')
from datetime import timedelta
# time delta depends on energy-saving time... https://www.timeanddate.com/time/zone/greece/athens
for i in range(len(pv1)):
    if pv1.iloc[i]['Datetime'].month in [4,5,6,7,8,9,10]:
        dthour = 3
    else: dthour = 2
    pv1['Datetime'].iloc[i] = pv1.iloc[i]['Datetime'] + timedelta(hours=dthour)
pv1['Year'] = pv1['Datetime'].dt.year
pv1['Month'] = pv1['Datetime'].dt.month
pv1['Week'] = pv1['Datetime'].dt.week
pv1['Day'] = pv1['Datetime'].dt.day
pv1['WeekDay'] = pv1['Datetime'].dt.weekday
pv1['Hour'] = pv1['Datetime'].dt.hour
pv1.head()


# In[3]:


pv1['Gs'] = pv1['Gb'] + pv1['Gd'] + pv1['Gr']
pv1.head()


# In[4]:


pv2 = pd.read_csv(bc_appl[1],header=0)
pv2['Datetime'] = pd.to_datetime(pv2.time, format='%Y%m%d:%H%M')
for i in range(len(pv2)):
    if pv2.iloc[i]['Datetime'].month in [4,5,6,7,8,9,10]:
        dthour = 3
    else: dthour = 2
    pv2['Datetime'].iloc[i] = pv2.iloc[i]['Datetime'] + timedelta(hours=dthour)
pv2['Year'] = pv2['Datetime'].dt.year
pv2['Month'] = pv2['Datetime'].dt.month
pv2['Week'] = pv2['Datetime'].dt.week
pv2['Day'] = pv2['Datetime'].dt.day
pv2['WeekDay'] = pv2['Datetime'].dt.weekday
pv2['Hour'] = pv2['Datetime'].dt.hour
pv2 = pv2[['Year','Month','Day','Hour','P']]
pv2.head()


# In[5]:


pv3 = pd.read_csv(bc_appl[2],header=0)
pv3['Datetime'] = pd.to_datetime(pv3.time, format='%Y%m%d:%H%M')
for i in range(len(pv3)):
    if pv3.iloc[i]['Datetime'].month in [4,5,6,7,8,9,10]:
        dthour = 3
    else: dthour = 2
    pv3['Datetime'].iloc[i] = pv3.iloc[i]['Datetime'] + timedelta(hours=dthour)
pv3['Year'] = pv3['Datetime'].dt.year
pv3['Month'] = pv3['Datetime'].dt.month
pv3['Week'] = pv3['Datetime'].dt.week
pv3['Day'] = pv3['Datetime'].dt.day
pv3['WeekDay'] = pv3['Datetime'].dt.weekday
pv3['Hour'] = pv3['Datetime'].dt.hour
pv3 = pv3[['Year','Month','Day','Hour','P']]
pv3.head()


# In[6]:


pnom = 7.5
pv2 = pv2.merge(pv3, on=['Year','Month','Day','Hour'])
pv2['P'] = pv2['P_x'] + pv2['P_y']
pv2['P'] = pv2['P']/pnom # electricity generation per kW
pv2.head()


# In[7]:


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
data12 = pv2.groupby(['Hour']).mean().reset_index()
fig= plt.figure(figsize=(7,4), dpi=100)
#plt.plot(data12.Hour, data12.PV_Wh, label='East')
plt.plot(data12.Hour, data12.P_x/1000, label='Ανατολικά')
plt.plot(data12.Hour, data12.P_y/1000, label='Νότια')
plt.plot(data12.Hour, data12.P/1000, label='Sum')
plt.xlabel('Ώρα')
plt.ylabel('Ισχύς (kWh/h)')
# plt.title('lat = ' + str(lat1) + ', long = ' + str(lon1) + ', PV = ' + str(peakpower) + 'kW' + ', Az = ' + str(surface_azimuth) + ' deg.')
plt.xticks(range(24))
plt.grid()
plt.legend()
plt.show()


# In[8]:


data1 = pv1[['Year','Month','Day','Hour','Gb','Gd','Gs']].merge(pv2[['Year','Month','Day','Hour','P']], on=['Year','Month','Day','Hour'], how='left')
data1.info()


# In[9]:


data1.head()


# In[10]:


data1.describe()


# In[11]:


data1.corr()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(7,4), dpi=80)
data1g = data1.groupby('Hour').mean().reset_index()
plt.plot(data1g.Hour, data1g.Gs, label='Gs')
plt.plot(data1g.Hour, data1g.P, label='P')
plt.xlabel('Hour')
plt.ylabel('W')
plt.grid()
plt.legend()
plt.show()


# In[13]:


# transform to daily data
ml = []
for iyear in data1.Year.unique():
    dfy = data1[data1.Year==iyear]
    for imonth in dfy.Month.unique():
        dfym = dfy[dfy.Month==imonth]
        for iday in dfym.Day.unique():
            dfymd = dfym[dfym.Day==iday]
            gbsum = int(dfymd.Gs.sum())
            psum = int(dfymd.P.sum())
            dayl = len(dfymd[dfymd.Gs>dfym.Gs.mean()])
            if psum>0: ml.append([iyear,imonth,iday,dayl,gbsum,psum])
            print(ml[-1])


# In[14]:


dfdsc = pd.DataFrame(ml,columns=['Year','Month','Day','Dlh','Gbs','Ps'])
dfdsc['Gbs'] = dfdsc['Gbs'].round(-3)/1000
dfdsc['Ps'] = dfdsc['Ps'].round(-3)/1000
dfdsc.head()


# In[15]:


from astropy.table import Table
t1 = Table.from_pandas(dfdsc)
t1.show_in_browser(jsviewer=True)
dfdsc.head()


# In[16]:


dfdsc[['Gbs','Ps']].corr()


# In[ ]:


# define the bin edges
bins = list(np.linspace(0, dfdsc.Ps.max(), num=5))
print(bins)
labels = list(range(len(bins)-1))
print(labels)
# bin the age column
dfdsc['Ps_b'] = pd.cut(dfdsc['Ps'], bins)
dfdsc['Ps_bl'] = pd.cut(dfdsc['Ps'], bins, labels=labels)
dfdsc.head()


# In[ ]:


# define the bin edges
bins = list(list(np.linspace(0, dfdsc.Gbs.max(), num=5)))
print(bins)
labels = list(range(len(bins)-1))
print(labels)
# bin the age column
dfdsc['Gbs_b'] = pd.cut(dfdsc['Gbs'], bins)
dfdsc['Gbs_bl'] = pd.cut(dfdsc['Gbs'], bins, labels=labels)
dfdsc.head()


# In[ ]:


# define the bin edges
bins = list(list(np.linspace(0, dfdsc.Dlh.max(), num=5)))
print(bins)
labels = list(range(len(bins)-1))
print(labels)
# bin the age column
dfdsc['Dlh_b'] = pd.cut(dfdsc['Dlh'], bins, include_lowest=True)
dfdsc['Dlh_bl'] = pd.cut(dfdsc['Dlh'], bins, labels=labels, include_lowest=True)
dfdsc.head()


# In[ ]:


dfdsc['Ps_b'] = dfdsc['Ps_b'].astype(str)
dfdsc['Gbs_b'] = dfdsc['Gbs_b'].astype(str)
dfdsc['Dlh_b'] = dfdsc['Dlh_b'].astype(str)


# In[ ]:


dfdsc.head()


# In[ ]:


fig= plt.figure(figsize=(7,4), dpi=80)
plt.scatter(dfdsc.Month, dfdsc.Gbs, c=dfdsc.Dlh, s=5, cmap='plasma')
plt.xlabel('Month')
plt.ylabel('Gbs')
plt.grid()
plt.show()


# In[17]:


dfdsc['MonthSin'] = np.sin(2 * np.pi * dfdsc['Month']/12.0)
dfdsc['MonthCos'] = np.cos(2 * np.pi * dfdsc['Month']/12.0)
dfdsc.head()


# In[18]:


fig= plt.figure(figsize=(7,4), dpi=80)
plt.scatter(dfdsc.Month, dfdsc.MonthSin)
plt.xlabel('Month')
plt.ylabel('MonthSin')
plt.grid()
plt.show()


# In[19]:


fig= plt.figure(figsize=(5,5), dpi=80)
plt.scatter(dfdsc.Month, dfdsc.MonthCos, c=dfdsc.Ps, s=5, cmap='hot')
plt.xlabel('Month')
plt.ylabel('MonthCos')
plt.grid()
plt.show()


# In[20]:


dfdsc.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from pickle import dump
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,2))
for i in ['Gbs_bl','Dlh_bl']:
    scaler.fit(dfdsc[i].values.reshape(-1,1))
    dump(scaler, open('scaler2'+str(i)+'.pkl', 'wb'))
    dfdsc[i] = scaler.fit_transform(dfdsc[i].values.reshape(-1,1))
    dfdsc[i] = dfdsc[i].round(5)
    print('scaler2'+str(i)+'.pkl')


# In[21]:


# generate SVM model
from sklearn.svm import SVC
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


# In[22]:


dfdsc.columns


# In[23]:


dfdsc = dfdsc.sample(frac=1)
dfdsc.head()


# In[24]:


# We call here our function defined above, with C=1 and sigma = 0.1
X = dfdsc[['Gbs','MonthCos']].values
y = dfdsc[['Ps']].values.reshape(len(X))
print("X.shape:", X.shape, "y.shape:", y.shape)


# In[25]:


dfdsc['Ps'].unique()


# In[26]:


X


# In[27]:


set(y)


# In[28]:


# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 


# In[29]:


# This function trains a nonlinear svm model and returns an object clf 
# that you can use later to make predictions. Read this function:
def train_nonlinear_svm(X, y, C, sigma):
    gamma = 1 / (2 * sigma**2)
    clf = SVC(C=C, kernel="rbf", gamma=gamma).fit(X, y) # Training SVM
    return clf

def train_polynomial_svm(X, y, degree):
    clf = SVC(kernel='poly',degree=degree, random_state=42).fit(X, y)
    return clf

""" TODO:
Write code here to use the validation set Xval, yval to determine the best C and sigma parameter to use.
Try using values of C and sigma from this range: [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30].
"""
rng = [0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3, 3, 10, 30]
params = [(C, sigma) for C in rng for sigma in rng]
degp = [1,2,3,4,5]


# In[30]:


clfs = [train_nonlinear_svm(X_train, y_train, C, sigma) for C, sigma in params]
accs = [100 * np.mean(clf.predict(X_test) == y_test) for clf in clfs]
C_best, sigma_best = params[np.argmax(accs)]
accs_best = accs[np.argmax(accs)]
print("C_best ", C_best, "Sigma_best ", sigma_best, "Score ", accs_best)


# In[31]:


clfs = [train_polynomial_svm(X_train, y_train, degree) for degree in degp]
accs = [100 * np.mean(clf.predict(X_test) == y_test) for clf in clfs]
degree_best = degp[np.argmax(accs)]
accs_best = accs[np.argmax(accs)]
print("degree_best ", degree_best, "Score ", accs_best)


# In[32]:


gamma_best = 1 / (2 * sigma_best**2)
clf = SVC(C=C_best, kernel="rbf", gamma=gamma_best).fit(X, y)


# In[33]:


# test decision tree classifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Create Decision Tree classifer object
clft = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clft = clft.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clft.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


import graphviz
from sklearn import tree
# DOT data
dot_data = tree.export_graphviz(clft, out_file=None, 
                                feature_names=['Gbs','MonthCos'],  
                                class_names=['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


# In[ ]:


# {'(0.0, 1604.5]', '(1604.5, 3209.0]', '(3209.0, 4813.5]', '(4813.5, 6418.0]'}

def nonlinear_svm_train_and_plot(X, y, C, gamma):
    print("Please wait. This might take some time (few seconds) ...")
    
    clf = SVC(C=C, kernel="rbf", gamma=gamma).fit(X, y) # Training
    
    # Plotting the dataset and nonlinear decision boundary
    fig, ax = plt.subplots()
    X0, X1, X2, X3 = X[y=='(0.0, 1604.5]'], X[y=='(1604.5, 3209.0]'], X[y=='(3209.0, 4813.5]'], X[y=='(4813.5, 6418.0]']
    ax.scatter(X0[:, 0], X0[:, 1], marker="$0$", color="gray")
    ax.scatter(X3[:, 0], X3[:, 1], marker="$1$", color="orange")
    
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    plot_x1, plot_x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.004), np.arange(x2_min, x2_max, 0.004))
    Z = clf.predict(np.c_[plot_x1.ravel(), plot_x2.ravel()])
    Z = Z.reshape(plot_x1.shape)
    
    ax.contour(plot_x1, plot_x2, Z, colors="green")
    
    ax.set_title("SVM Decision Boundary with $C = {}, \sigma = {}$".format(C, sigma))
    fig.show()


# We call here our function defined above, with C=1 and sigma = 0.1
# nonlinear_svm_train_and_plot(X, y, C_best, gamma_best)


# In[34]:


dfdsc.head()


# In[35]:


dfdsc['Prediction'] = 111
i=0
for iyear in dfdsc.Year.unique():
    dfy = dfdsc[dfdsc.Year==iyear]
    for imonth in dfy.Month.unique():
        dfym = dfy[dfy.Month==imonth]
        for iday in dfym.Day.unique():
            dfymd = dfym[dfym.Day==iday]
            Xi = np.array(dfymd[['Gbs','MonthCos']]).reshape(1,-1)
            Zi = clf.predict(Xi)
            dfdsc.loc[(dfdsc.Year==iyear)&(dfdsc.Month==imonth)&(dfdsc.Day==iday),'Prediction'] = Zi[0]
    i+=1


# In[36]:


dfdsc.columns


# In[37]:


dfdsc.head()


# In[38]:


dfdsc.MonthCos = dfdsc.MonthCos.round(2)
dfdsc.Ps = dfdsc.Ps.astype(int)
dfdsc.head()


# In[39]:


dfdsc.to_excel('data_pv_day_class_cos.xlsx', index=None)


# In[40]:


dfdsc = dfdsc[['Year', 'Month', 'Day', 'Dlh', 'Gbs', 'MonthCos', 'Ps', 'Prediction']]
dfdsc.head()


# In[41]:


dfdsc['Error'] = np.where(dfdsc['Ps']==dfdsc['Prediction'],'Ok','Error')


# In[42]:


dfdsce = dfdsc[dfdsc['Error']=='Error']
dfdsce.hist(column='Month')


# In[43]:


dfdsc.info()


# In[45]:


import pickle
filename = "ma43-svm.pickle"
pickle.dump(clf, open(filename, "wb"))
# filename = "ma43-dtree.pickle"
# pickle.dump(clft, open(filename, "wb"))


# In[58]:


dfdsc.Gbs.unique()


# In[68]:


dfdsc.MonthCos.unique()


# In[67]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(np.array([4, 0.10]).reshape(1,-1))
print(result)


# In[ ]:




