import pandas as pd
from src import *
import gc
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
class RandTS():
    def __init__(self, method = 'l',depth = 5,ntree=10,var=10, featureSelection = None):
        self.featureSelection = featureSelection
        self.method = method
        self.n = False
        self.depth = depth
        self.var = var
        self.ntree = ntree
        
        
        self.datasetDirectory = None
        self.multivariate = False
        self.train = None
        self.test = None
        self.labels_train = None
        self.labels_test = None
        self.randTrees_list = []
        self.train_representations = []
        self.trainReps = []
        self.trainIds = []
        self.unsupervisedSelector = []
		
    def fit(self, train, train_labels):
        self.train = train
        self.labels_train = train_labels
        batch_list2 = list(range(10)) 
        first = True
        counter = 0

        rep = np.random.randint(1,10000)
        for b in batch_list2:

            represent_train, train_id = prepare_data_new(self.train,mode=self.method)

            represent_train['Id'] = train_id
            represent_train = represent_train.dropna()

            train_id = represent_train['Id'].values
            represent_train = represent_train.drop(columns = ['Id'])

       	
            trainbow,randTrees,trainRep,train_id = learn_representation_sparse_new(represent_train,self.labels_train, train_id, self.depth, int(self.ntree/len(batch_list2)),rep*1000+counter, True,normal=False)
            self.trainReps.append(trainRep)
            self.trainIds.append(train_id)
            
            counter = counter +1
			
            self.randTrees_list.append(randTrees)
            self.train_representations.append(trainbow)
		
		
    def predict(self,test,test_labels = None):
        self.test = test
        if test_labels is not None:
            self.labels_test = test_labels
        batch_list2 = list(range(10)) 
        first = True
        counter = 0

        rep = np.random.randint(1,10000)
        for b in batch_list2:

            represent_test, test_id = prepare_data_new(self.test,mode=self.method)

            represent_test['Id'] = test_id
            represent_test = represent_test.dropna()
            test_id = represent_test['Id'].values
            represent_test = represent_test.drop(columns = ['Id'])

            testbow = get_terminalNode_representation(represent_test,test_id,self.trainReps[counter],self.trainIds[counter],self.randTrees_list[counter],self.depth, int(self.ntree/len(batch_list2)))
            
            trainbow_updated = self.train_representations[counter]
            
            if self.featureSelection == 'Unsupervised':
                sel = VarianceThreshold(threshold=0)
                trainbow_updated = sel.fit_transform(trainbow_updated)
                testbow = sel.transform(testbow)                                    
                                        
                vals = np.percentile(np.var(trainbow_updated.todense(),axis=0),self.var,axis=1)      
                sel = VarianceThreshold(threshold=vals)
                trainbow_updated = sel.fit_transform(trainbow_updated)
                testbow = sel.transform(testbow)

            elif self.featureSelection == 'Supervised':
                clf = ExtraTreesClassifier(n_estimators=100)
                clf = clf.fit(trainbow_updated, self.labels_train)
                model = SelectFromModel(clf, prefit=True)

                trainbow_updated = model.transform(trainbow_updated)
                testbow = model.transform(testbow)
            

            if first is True:
                distTrainTest=pairwise_distances(trainbow_updated,testbow,metric='manhattan',n_jobs=-1) 
                first = False
            else:
                distTrainTest = distTrainTest +pairwise_distances(trainbow_updated,testbow,metric='manhattan',n_jobs = -1) 
            del testbow
            counter = counter +1
            gc.collect()

        predicted_test = np.array(self.labels_train)[np.argmin(distTrainTest, axis=0)]
        if test_labels is not None:
            self.test_accuracy=accuracy_score(test_labels,predicted_test)

        #temp_results = pd.DataFrame({'data_name':[data_name],'method':method,'distance_measure':[distance_measure],'normalize':[n],'rep':[rep+1],'depth':[depth],'ntree':[400],'type':[is_terminal],'test_acc':[test_accuracy],'param_time':[np.round(param_time,1)],'train_time':[np.round(train_time,1)],'test_time':[np.round(test_time,1)]})
        #test_results = test_results.append(temp_results,sort=False)
        return predicted_test 
		
		
    def addDatasetDirectory(self,directory):
        self.datasetDirectory = directory
    
    def selectParameters(self, train, train_labels, param_kwargs={'depth_cv':[3,5,10], 'ntree_cv':[100], 'rep_num':1, 'method_cv':['l','d','b']}):
        self.train = train
        self.labels_train = train_labels
        depth_cv = param_kwargs['depth_cv']
        ntree_cv = param_kwargs['ntree_cv']
        term_cv = [True]
        rep_num = param_kwargs['rep_num']
        method_cv = param_kwargs['method_cv']
        distance_list = ['manhattan']
        normal_list = [False]
        batch_list = list(range(10))


        results = pd.DataFrame(columns=['method','distance_measure','normalize','rep','depth','ntree','var','oob_acc'])

        for method in method_cv:
            for distance_measure in distance_list:
                for rep in range(rep_num):
                    for depth in depth_cv:
                        for ntree in ntree_cv:
                            for is_terminal in term_cv: 
                                for n in [0]:
                                    first = True
                                    counter = 0
                                    for b in batch_list:
                                        represent_train, train_id = prepare_data_new(self.train,mode=method)
                                        represent_train['Id'] = train_id
                                        represent_train = represent_train.dropna()
                                        train_id = represent_train['Id'].values
                                        represent_train = represent_train.drop(columns = ['Id'])

                                        
                                        trainbow,randTrees,trainRep,train_id = learn_representation_sparse_new(represent_train,self.labels_train, train_id, depth, int(ntree/len(batch_list)),rep*1000+counter, True,normal=False)
                                        counter = counter +1                    
                                        
                                        if first is True:
                                            distTrain=pairwise_distances(trainbow,metric="manhattan",n_jobs=-1)       
                                            first = False
                                        else:
                                            distTrain=distTrain+ pairwise_distances(trainbow,metric="manhattan",n_jobs=-1)
                                        del trainbow
                                        gc.collect()
                                    np.fill_diagonal(distTrain, 1000000000000)
                                    nncl=KNeighborsClassifier(n_neighbors=1,metric='precomputed')
                                    nnfit=nncl.fit(distTrain,self.labels_train)
                                    predicted_train = nnfit.predict(distTrain)
                                    cv_accuracy = accuracy_score(self.labels_train,predicted_train)
                                    temp_results = pd.DataFrame({'method':method,'distance_measure':[distance_measure],'normalize':[False],'rep':[rep+1],'depth':[depth],'ntree':[ntree],'var':[n],'type':[is_terminal],'oob_acc':[cv_accuracy]})

                                    results = results.append(temp_results,sort=False)
                                        
        summary = results.groupby(['method','normalize','depth','var']).mean().reset_index()[results.groupby(['method','normalize','depth','var']).mean().reset_index()['oob_acc'] == results.groupby(['method','normalize','depth','var']).mean().reset_index()['oob_acc'].max()].reset_index(drop=True).sort_values(['method'],ascending=False).iloc[0]

        self.method = summary['method']
        self.n = summary['normalize']
        self.depth =  summary['depth']
        self.var = summary['var']
        print('Parameter selection is completed')
        print(summary)

