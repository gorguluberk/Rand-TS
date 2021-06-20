import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from os import listdir
import time
from scipy import sparse,spatial
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

def load_data(directory, data_name):
    path = directory+data_name+"\\"
    train = np.genfromtxt(path+data_name+'_TRAIN.txt', delimiter=',')
    test = np.genfromtxt(path+data_name+'_TEST.txt', delimiter=',')

    labels_train= np.array(train[:,0],dtype=np.int16)
    X_train= train[:,1:train.shape[1]]
    labels_test= np.array(test[:,0],dtype=np.int16)
    X_test= test[:,1:test.shape[1]]
    return train,test,labels_train,labels_test,X_train,X_test


def load_data_new(directory, data_name):
    path = directory+data_name+"\\"
    train = np.genfromtxt(path+data_name+'_TRAIN.txt')
    test = np.genfromtxt(path+data_name+'_TEST.txt')
    
    labels_train= np.array(train[:,0],dtype=np.int16)
    X_train= train[:,1:train.shape[1]]
    labels_test= np.array(test[:,0],dtype=np.int16)
    X_test= test[:,1:test.shape[1]]
    return train,test,labels_train,labels_test,X_train,X_test

def prepare_data_new(X_train,mode):
    times = [range(X_train.shape[1]-1)]
    times_train = np.repeat(times,X_train.shape[0],axis=0).flatten()

    represent_train = pd.DataFrame({'times':times_train})
    
    if mode == "l":
        represent_train['obs'] = X_train[:,1:].flatten()
    elif mode == "d":
        represent_train['diff'] = np.diff(X_train).flatten()
    else:
        represent_train['obs'] = X_train[:,1:].flatten()
        represent_train['diff'] = np.diff(X_train).flatten()

    train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
    train_id.sort()
    return represent_train, train_id

	

def prepare_data(X_train,X_test,mode):
    times = [range(X_train.shape[1]-1)]
    times_train = np.repeat(times,X_train.shape[0],axis=0).flatten()
    times_test =  np.repeat(times,X_test.shape[0],axis=0).flatten()


    represent_train = pd.DataFrame({'times':times_train})
    represent_test = pd.DataFrame({'times':times_test})

    
    if mode == "l":
        represent_train['obs'] = X_train[:,1:].flatten()
        represent_test['obs'] = X_test[:,1:].flatten()
    elif mode == "d":
        represent_train['diff'] = np.diff(X_train).flatten()
        represent_test['diff'] = np.diff(X_test).flatten()
    else:
        represent_train['obs'] = X_train[:,1:].flatten()
        represent_train['diff'] = np.diff(X_train).flatten()
        represent_test['obs'] = X_test[:,1:].flatten()
        represent_test['diff'] = np.diff(X_test).flatten()
		

    train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
    train_id.sort()

    test_id = list(range(X_test.shape[0]))*int(X_test.shape[1]-1)
    test_id.sort()
    return represent_train, represent_test, train_id, test_id


def prepare_data_new_multivariate(X_train,ndims,mode='l'):
    times = [range(X_train.shape[1]-1)]
    times_train = np.repeat(times,X_train.shape[0],axis=0).flatten()

    represent_train = pd.DataFrame({'times':times_train})
    
    if mode=='l':
        represent_train['Obs1']= X_train[:,:-1,0].flatten()

        for i in range(1,ndims):
            represent_train['Obs'+str(i+1)]= X_train[:,:-1,i].flatten()
            
        train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
        train_id.sort()

    elif mode =='d':
        represent_train['Obs1']= np.diff( X_train[:,:,0]).flatten()

        for i in range(1,ndims):
            represent_train['Obs'+str(i+1)]= np.diff( X_train[:,:,i]).flatten()
            
            
        train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
        train_id.sort()
    return represent_train, train_id
    
def load_data_multivariate(directory,data_name):
    from scipy.io.arff import loadarff 

    path = directory+data_name+"/"
    ndims = int((len(listdir(path))-4)/2)

    raw_data_train = loadarff(path+data_name+'Dimension1_TRAIN.arff')
    raw_data_train = pd.DataFrame(raw_data_train[0])
    labels_train = raw_data_train.values[:,-1].astype(str)
    X_train = raw_data_train.values[:,:-1]


    raw_data_test = loadarff(path+data_name+'Dimension1_TEST.arff')
    raw_data_test = pd.DataFrame(raw_data_test[0])
    labels_test = raw_data_test.values[:,-1].astype(str)
    X_test = raw_data_test.values[:,:-1]


    for i in range(2,(ndims+1)):
        raw_data_train = loadarff(path+data_name+'Dimension'+str(i)+'_TRAIN.arff')
        raw_data_train = pd.DataFrame(raw_data_train[0])
        X_train = np.dstack((X_train,raw_data_train.values[:,:-1]))

        raw_data_test = loadarff(path+data_name+'Dimension'+str(i)+'_TEST.arff')
        raw_data_test = pd.DataFrame(raw_data_test[0])
        X_test = np.dstack((X_test,raw_data_test.values[:,:-1]))
    return X_train, labels_train, X_test, labels_test, ndims


def load_prepare_data_multivariate(directory,data_name,mode='l'):
    from scipy.io.arff import loadarff 
    
    path = directory+data_name+"/"
    ndims = int((len(listdir(path))-4)/2)
    
    raw_data_train = loadarff(path+data_name+'Dimension1_TRAIN.arff')
    raw_data_train = pd.DataFrame(raw_data_train[0])
    labels_train = raw_data_train.values[:,-1].astype(str)
    X_train = raw_data_train.values[:,:-1]
    
    
    raw_data_test = loadarff(path+data_name+'Dimension1_TEST.arff')
    raw_data_test = pd.DataFrame(raw_data_test[0])
    labels_test = raw_data_test.values[:,-1].astype(str)
    X_test = raw_data_test.values[:,:-1]
    
    times = [range(X_train.shape[1]-1)]
    times_train = np.repeat(times,X_train.shape[0],axis=0).flatten()
    times_test =  np.repeat(times,X_test.shape[0],axis=0).flatten()

    represent_train = pd.DataFrame({'times':times_train})
    represent_test = pd.DataFrame({'times':times_test})
    
    if mode=='l':
        represent_train['Obs1']= X_train[:,:-1].flatten()
        represent_test['Obs1']= X_test[:,:-1].flatten()

        for i in range(1,ndims):
            raw_data_train = loadarff(path+data_name+'Dimension'+str(i)+'_TRAIN.arff')
            raw_data_train = pd.DataFrame(raw_data_train[0])
            X_train = raw_data_train.values[:,:-1]

            raw_data_test = loadarff(path+data_name+'Dimension'+str(i)+'_TEST.arff')
            raw_data_test = pd.DataFrame(raw_data_test[0])
            X_test = raw_data_test.values[:,:-1]

            represent_train['Obs'+str(i+1)]= X_train[:,:-1].flatten()
            represent_test['Obs'+str(i+1)]= X_test[:,:-1].flatten()
            
            
        train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
        train_id.sort()

        test_id = list(range(X_test.shape[0]))*int(X_test.shape[1]-1)
        test_id.sort()
    elif mode =='d':
        represent_train['Obs1']= np.diff(X_train).flatten()
        represent_test['Obs1']= np.diff(X_test).flatten()

        for i in range(1,ndims):
            raw_data_train = loadarff(path+data_name+'Dimension'+str(i)+'_TRAIN.arff')
            raw_data_train = pd.DataFrame(raw_data_train[0])
            X_train = raw_data_train.values[:,:-1]

            raw_data_test = loadarff(path+data_name+'Dimension'+str(i)+'_TEST.arff')
            raw_data_test = pd.DataFrame(raw_data_test[0])
            X_test = raw_data_test.values[:,:-1]

            represent_train['Obs'+str(i+1)]= np.diff(X_train).flatten()
            represent_test['Obs'+str(i+1)]= np.diff(X_test).flatten()
            
            
        train_id = list(range(X_train.shape[0]))*int(X_train.shape[1]-1)
        train_id.sort()

        test_id = list(range(X_test.shape[0]))*int(X_test.shape[1]-1)
        test_id.sort()
        
    return represent_train, represent_test, train_id, test_id,labels_train,labels_test

def learn_representation(represent_train, represent_test,labels_train,labels_test,train_id,test_id,depth, ntree,random_seed,is_terminal=True,normal=False):
    rt = RandomTreesEmbedding(max_depth=depth,n_estimators=ntree,random_state =random_seed,n_jobs=-1)

    traincv=represent_train
    testcv=represent_test
    trainind=np.unique(train_id)
    testind=np.unique(test_id)

    trainlabels=labels_train
    testlabels=labels_test
    
    randTrees=rt.fit(traincv.values)
    
    if is_terminal:
        trainRep=randTrees.transform(traincv.values)
        testRep=randTrees.transform(testcv.values)
    else:
        trainRep=randTrees.decision_path(traincv)[0]
        testRep=randTrees.decision_path(testcv)[0]

    newId=np.unique(train_id)
    Mask = sparse.csr_matrix((np.ones(traincv.shape[0],int),(train_id, np.arange(traincv.shape[0]))), shape=(newId.shape[0],traincv.shape[0]))

    trainbow = Mask * trainRep

    newId=np.unique(test_id)
    Mask = sparse.csr_matrix((np.ones(testcv.shape[0],int),(test_id, np.arange(testcv.shape[0]))), shape=(newId.shape[0],testcv.shape[0]))

    testbow = Mask * testRep              

    if normal:
        trainbow = normalize(trainbow, norm='l1', axis=1)
        testbow = normalize(testbow, norm='l1', axis=1)
	
    return trainbow,testbow

def train_f(c,id):
    return sparse.coo_matrix(([1]*len(id), (id, c)))

def learn_representation_sparse(represent_train, represent_test,labels_train,labels_test,train_id,test_id,depth, ntree,random_seed,is_terminal=True,normal=False):
    rt = RandomTreesEmbedding(max_depth=depth,n_estimators=ntree,random_state =random_seed,n_jobs=-1)

    traincv=represent_train
    testcv=represent_test
    trainind=np.unique(train_id)
    testind=np.unique(test_id)

    trainlabels=labels_train
    testlabels=labels_test
    
    randTrees=rt.fit(traincv.values)
    
    trainRep=randTrees.apply(traincv.values)
    testRep=randTrees.apply(testcv.values)
            
    trainbow = np.apply_along_axis(train_f, 0, trainRep,train_id)
    trainbow = sparse.hstack(trainbow)
    
    testbow = np.apply_along_axis(train_f, 0, testRep,test_id)
    testbow = sparse.hstack(testbow)
    
    if normal:
        trainbow = normalize(trainbow, norm='l1', axis=1)
        testbow = normalize(testbow, norm='l1', axis=1)
    
    return trainbow,testbow

def learn_representation_sparse_2(represent_train, represent_test,labels_train,labels_test,train_id,test_id,depth, ntree,random_seed,is_terminal=True,normal=False):
    
    rt = RandomTreesEmbedding(max_depth=depth,n_estimators=ntree,random_state =random_seed,n_jobs=-1)

    traincv=represent_train
    trainind=np.unique(train_id)
    trainlabels=labels_train

    randTrees=rt.fit(traincv.values)	
    trainRep=randTrees.apply(traincv.values)

    if represent_test is not None:
        
        testcv=represent_test
        testind=np.unique(test_id)
        testlabels=labels_test
        testRep=randTrees.apply(testcv.values)
	

        allRep = np.vstack((trainRep,testRep))
        allids = np.concatenate((np.array(train_id),np.array(test_id)+np.max(train_id)+1),axis = 0)
    else:
        allRep = trainRep
        allids = np.array(train_id)

    ids=np.tile(allids,ntree)
    
    
    increments=np.arange(0,ntree)*(2**depth)
    allRep=allRep+increments
    node_ids=allRep.flatten('F')
    
    data=np.repeat(1,len(ids))
    
    allbow=sparse.coo_matrix((data,(ids,node_ids)), dtype=np.int8).tocsr()
    
    select_ind =trainind.shape[0]
    

    allbow = normalize(allbow, norm='l1', axis=1)
		
    return allbow[:select_ind,:],allbow[select_ind:,:]



def learn_representation_sparse_new(represent_train,labels_train,train_id,depth, ntree,random_seed,is_terminal=True,normal=False):
    rt = RandomTreesEmbedding(max_depth=depth,n_estimators=ntree,random_state =random_seed,n_jobs=-1)
    
    traincv=represent_train
    trainind=np.unique(train_id)
    trainlabels=labels_train

    randTrees=rt.fit(traincv.values)    
    trainRep=randTrees.apply(traincv.values)
    allRep = trainRep
    allids = np.array(train_id)

    ids=np.tile(allids,ntree)
        
    increments=np.arange(0,ntree)*(2**depth)
    allRep=allRep+increments
    node_ids=allRep.flatten('F')
    
    data=np.repeat(1,len(ids))
    allbow=sparse.coo_matrix((data,(ids,node_ids)), dtype=np.int8).tocsr()
    select_ind =trainind.shape[0]

    allbow = normalize(allbow, norm='l1', axis=1)
        
    return allbow,randTrees,trainRep,train_id


def get_terminalNode_representation(represent_test,test_id,trainRep,train_id,randTree,depth, ntree):

    testcv=represent_test
    
    testind=np.unique(test_id)
    test_time = time.time()
    testRep=randTree.apply(testcv.values)
    
    test_time = time.time()-test_time
    
    allRep = np.vstack((trainRep,testRep))
    allids = np.concatenate((np.array(train_id),np.array(test_id)+np.max(train_id)+1),axis = 0)

    ids=np.tile(allids,ntree)
    
    increments=np.arange(0,ntree)*(2**depth)
    allRep=allRep+increments
    node_ids=allRep.flatten('F')
    
    data=np.repeat(1,len(ids))
    
    allbow=sparse.coo_matrix((data,(ids,node_ids)), dtype=np.int8).tocsr()
    
    select_ind =testind.shape[0]
    
    allbow = normalize(allbow, norm='l1', axis=1)
    
    return allbow[allbow.shape[0]-select_ind:,:]


    
    

    










def learn_representation_sparse_2_complex(represent_train, represent_test,labels_train,labels_test,train_id,test_id,depth, ntree,random_seed,is_terminal=True,normal=False):
	
    rt = RandomTreesEmbedding(max_depth=depth,n_estimators=ntree,random_state =random_seed,n_jobs=-1)

    traincv=represent_train
    trainind=np.unique(train_id)
    trainlabels=labels_train

    randTrees=rt.fit(traincv.values)	
    trainRep=randTrees.apply(traincv.values)

    if represent_test is not None:
		
        testcv=represent_test
        testind=np.unique(test_id)
        testlabels=labels_test
        test_time = time.time()
        testRep=randTrees.apply(testcv.values)
        test_time = time.time()-test_time

        allRep = np.vstack((trainRep,testRep))
        allids = np.concatenate((np.array(train_id),np.array(test_id)+np.max(train_id)+1),axis = 0)
    else:
        allRep = trainRep
        allids = np.array(train_id)

    ids=np.tile(allids,ntree)
    
    
    increments=np.arange(0,ntree)*(2**depth)
    allRep=allRep+increments
    node_ids=allRep.flatten('F')
    
    data=np.repeat(1,len(ids))
    
    allbow=sparse.coo_matrix((data,(ids,node_ids)), dtype=np.int8).tocsr()
    
    select_ind =trainind.shape[0]
    return allbow[:select_ind,:],allbow[select_ind:,:],test_time


def feature_selection(trainbow,testbow,labels_train,cv=5):
    regr = LassoCV(cv=cv, random_state=0, fit_intercept=False,n_jobs =-1,max_iter=1000000,tol=0.001)
    regr.fit(trainbow, labels_train)	
    coeffs = np.where(regr.coef_ != 0)[0]
    trainbow = trainbow[:,coeffs]
    testbow = testbow[:,coeffs]
    
    return trainbow,testbow
    
def loo_cv(trainbow,labels_train,metric = "manhattan"):
    distTrain=pairwise_distances(trainbow,metric=metric,n_jobs=-1)
    np.fill_diagonal(distTrain, 1000000000000)
    nncl=KNeighborsClassifier(n_neighbors=1,metric='precomputed',n_jobs = -1)
    nnfit=nncl.fit(distTrain,labels_train)
    predicted_train = nnfit.predict(distTrain)
    return accuracy_score(labels_train,predicted_train)

def loo_cv_2(trainbow,labels_train,metric = "cityblock"):
    distTrain=spatial.distance.pdist(trainbow.todense(),metric='cityblock',n_jobs=-1)
    np.fill_diagonal(distTrain, 1000000000000)
    nncl=KNeighborsClassifier(n_neighbors=1,metric='precomputed',n_jobs = -1)
    nnfit=nncl.fit(distTrain,labels_train)
    predicted_train = nnfit.predict(distTrain)
    return accuracy_score(labels_train,predicted_train)

def test_model(trainbow,testbow,labels_train,labels_test,metric = "manhattan"):
    distTrain=pairwise_distances(trainbow,metric=metric,n_jobs = -1)
    nncl=KNeighborsClassifier(n_neighbors=1,metric='precomputed',n_jobs = -1)
    nnfit=nncl.fit(distTrain,labels_train)
    a= time.time()
    distTrainTest=pairwise_distances(testbow,trainbow,metric=metric,n_jobs = -1) 
    predicted_test=nnfit.predict(distTrainTest)
    accuracy=accuracy_score(labels_test,predicted_test)
    return accuracy

def test_model_2(trainbow,testbow,labels_train,labels_test,metric = "manhattan"):
    a= time.time()
    distTrainTest=pairwise_distances(trainbow,testbow,metric='manhattan',n_jobs = -1) 
    predicted_test = np.array(labels_train)[np.argmin(distTrainTest, axis=0)]
    accuracy=accuracy_score(labels_test,predicted_test)
    return accuracy

def data_summary(directory):
    datasets = listdir(directory)
    data_s= pd.DataFrame(columns=['data','train','test','total'])
    for data_name in datasets:
        path = directory+data_name+"\\"
        train = np.genfromtxt(path+data_name+'_TRAIN.txt', delimiter=',')
        test = np.genfromtxt(path+data_name+'_TEST.txt', delimiter=',')
        temp =pd.DataFrame({'data':[data_name],'train':[train.shape[0]],'test':[test.shape[0]],'total':[train.shape[0]+test.shape[0]]})
        data_s=data_s.append(temp)
    return data_s
