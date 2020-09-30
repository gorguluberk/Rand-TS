import pandas as pd
from src import *
import gc
class RandTSMultivariate():
    def __init__(self,featureSelection = 'None'):
        self.featureSelection = featureSelection
        self.method = None
        self.n = False
        self.depth = None
        self.var = 0
        self.datasetDirectory = None
        self.multivariate = False

	def addDatasetDirectory(self,directory):
        self.datasetDirectory = directory
    
    def selectParameters(self, dataset_name, param_kwargs={'depth_cv':[3,5,10], 'ntree_cv':[100], 'rep_num':1, 'method_cv':['l','d']}):
        depth_cv = param_kwargs['depth_cv']
        ntree_cv = param_kwargs['ntree_cv']
        term_cv = [True]
        rep_num = param_kwargs['rep_num']
        method_cv = param_kwargs['method_cv']
        distance_list = ['manhattan']
        normal_list = [False]
        batch_list = list(range(10))


        data_name = dataset_name
        train,test,labels_train,labels_test,X_train,X_test = load_data(directory,data_name)
        results = pd.DataFrame(columns=['data_name','method','distance_measure','normalize','rep','depth','ntree','var','oob_acc'])

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
                                        represent_train, represent_test, train_id, test_id,labels_train,labels_test = load_prepare_data_multivariate(directory,data_name,method)
                                        represent_train['Id'] = train_id
                                        represent_train = represent_train.dropna()
                                        train_id = represent_train['Id'].values
                                        represent_train = represent_train.drop(columns = ['Id'])


                                        represent_test['Id'] = test_id
                                        represent_test = represent_test.dropna()
                                        test_id = represent_test['Id'].values
                                        represent_test = represent_test.drop(columns = ['Id'])

                                        trainbow,testbow = learn_representation_sparse_2(represent_train, None,labels_train,labels_test, train_id, test_id, depth, int(ntree/len(batch_list)),rep*1000+counter, is_terminal,normal=False)
                                        counter = counter +1

                                        if first is True:
                                            distTrain=pairwise_distances(trainbow,metric="manhattan",n_jobs=-1)       
                                            first = False
                                        else:
                                            distTrain=distTrain+ pairwise_distances(trainbow,metric="manhattan",n_jobs=-1)
                                        del trainbow
                                        del testbow
                                        gc.collect()
                                    np.fill_diagonal(distTrain, 1000000000000)
                                    nncl=KNeighborsClassifier(n_neighbors=1,metric='precomputed')
                                    nnfit=nncl.fit(distTrain,labels_train)
                                    predicted_train = nnfit.predict(distTrain)
                                    cv_accuracy = accuracy_score(labels_train,predicted_train)
                                    temp_results = pd.DataFrame({'data_name':[data_name],'method':method,'distance_measure':[distance_measure],'normalize':[False],'rep':[rep+1],'depth':[depth],'ntree':[ntree],'var':[n],'type':[is_terminal],'oob_acc':[cv_accuracy]})

                                    results = results.append(temp_results,sort=False)
                                        
        summary = results.groupby(['method','normalize','depth','var']).mean().reset_index()[results.groupby(['method','normalize','depth','var']).mean().reset_index()['oob_acc'] == results.groupby(['method','normalize','depth','var']).mean().reset_index()['oob_acc'].max()].reset_index(drop=True).sort_values(['method'],ascending=False).iloc[0]

        self.method = summary['method']
        self.n = summary['normalize']
        self.depth =  summary['depth']
        self.var = summary['var']
        print('Parameter selection is completed')
        print(summary)


    def train_test(self,dataset_name, method=None,depth=None):
        if depth is None:
            depth = self.depth
        if method is None:
            method = self.method
        batch_list2 = list(range(100)) 
        first = True
        counter = 0

        rep = np.random.randint(1,10000)
        for b in batch_list2:

            represent_train, represent_test, train_id, test_id,labels_train,labels_test = load_prepare_data_multivariate(directory,data_name,method)
            represent_train['Id'] = train_id
            represent_train = represent_train.dropna()
            train_id = represent_train['Id'].values
            represent_train = represent_train.drop(columns = ['Id'])


            represent_test['Id'] = test_id
            represent_test = represent_test.dropna()
            test_id = represent_test['Id'].values
            represent_test = represent_test.drop(columns = ['Id'])
            trainbow,testbow = learn_representation_sparse_2(represent_train, represent_test,labels_train,labels_test, train_id, test_id, depth, int(500/len(batch_list2)),rep*1000+counter, is_terminal,normal=n)
            counter = counter +1
            #print(rep+counter)

            clf = ExtraTreesClassifier(n_estimators=100)
            clf = clf.fit(trainbow, labels_train)
            model = SelectFromModel(clf, prefit=True)
            trainbow_1 = model.transform(trainbow)
            testbow_1 = model.transform(testbow)     
            

            if first is True:
                distTrainTest=pairwise_distances(trainbow_1,testbow_1,metric='manhattan',n_jobs=-1) 
                first = False
            else:
                distTrainTest = distTrainTest +pairwise_distances(trainbow_1,testbow_1,metric='manhattan',n_jobs = -1) 
            del trainbow
            del testbow
            gc.collect()
            
        predicted_test = np.array(labels_train)[np.argmin(distTrainTest, axis=0)]
        test_accuracy=accuracy_score(labels_test,predicted_test)

        #temp_results = pd.DataFrame({'data_name':[data_name],'method':method,'distance_measure':[distance_measure],'normalize':[n],'rep':[rep+1],'depth':[depth],'ntree':[400],'type':[is_terminal],'test_acc':[test_accuracy],'param_time':[np.round(param_time,1)],'train_time':[np.round(train_time,1)],'test_time':[np.round(test_time,1)]})
        #test_results = test_results.append(temp_results,sort=False)

        print('Test Accuracy :', test_accuracy)
        return test_accuracy, predicted_test