# # Imports

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,KFold
from numpy.linalg import inv


##For ignoring warning caused due to scipy LAPACK : 
#alternative - https://github.com/scipy/scipy/issues/5998
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# # Loading the Dataset : Diabetes and Boston
#Diabetes Dataset
db = datasets.load_diabetes()
db_X=db.data
db_Y=db.target

#Boston DataSet
bt=datasets.load_boston()
bt_X=bt.data
bt_Y=bt.target


# # Defining the regularize value and percentage of test data = 1-train data
#Percenatge of Dataset in Training for Q1
P = np.array([0.5,0.4,0.3,0.2,0.05,0.01]) ## this is test percentage as in question
#Regularizer
L = np.array([0,0.01,0.1,1])

# # Building the model and final calls: Diabetes/Boston Q1

# ## Plot Function

# col represent the column in subplot graph
# name is dataset name like 'B' =boston 
def plot_TT(col,train,test,r2,X,name,figNum,lambdas,str):
    
    if name=='D':
        name='Diabetes'
        plt.figure(num=figNum,figsize=(18,10),dpi=100)
        plt.suptitle('Diabetes', fontsize=16)
        #plt.gca().set_ylim(2500, 3600)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.suptitle('Long Suptitle', fontsize=24)
    else:
        name='Boston'
        plt.figure(num=figNum,figsize=(18,10),dpi=100)
        plt.suptitle('Boston', fontsize=16)
        #plt.gca().set_ylim(15,30)
        
    plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
    fig = plt.gcf()
    fig.canvas.set_window_title(str)
    plt.subplot(2,4,col)
    plt.grid(True)
    plt.xlabel('-------TrainingSet Size------>',size=10)
    plt.ylabel('---------Error---------->',size=10)
    plt.plot(X,train,color='blue',marker='o',markersize=10,label='Train Error')
    plt.plot(X,test,color='red',marker='o',markersize=10,label='Test Error')
    #plt.gca().set_ylim(2500, 3600)
    plt.legend(loc=0,shadow=True)
    plt.title(name+':Train/Test Error - $\lambda$ ={:.2f}'.format(lambdas),size=7)
    
    # R2 plot
    plt.subplot(2,4,col+4)
    plt.grid(True)
    plt.xlabel('Training Set Size')
    plt.ylabel('R2 Score')
    plt.plot(X,r2,color='black',marker='o',markersize=10,label='R2 Score')
    #plt.gca().set_ylim(0.4, 0.6)
    plt.legend(loc=0)
    plt.title(name+':R2 Score ',size=7)

def generalized_q1(P,L,which_data,_data,figNum,str):
    # declaring the empty array for each regularizer
    i=1
    db_X=_data.data
    db_Y=_data.target
    for l in np.nditer(L):
        res_train=[]
        res_test=[]
        res_R2=[]
        X=[]
        diabetes_reg_model = linear_model.Ridge(alpha=l, fit_intercept=True, normalize=True, copy_X=True,tol=0.001)
        for p in np.nditer(P):
            X_train,X_test,Y_train,Y_test = train_test_split(db_X,db_Y,test_size=p,random_state=0,shuffle=True)
            #print(X_train.shape,'gth',X_test.shape)
            X.append(X_train.shape[0])
            diabetes_reg_model.fit(X_train,Y_train)
            ## finding the predicted labels
            db_predict_Y = diabetes_reg_model.predict(X_train)
            db_predict_tY=diabetes_reg_model.predict(X_test)
            
            ##calculating mean sqaure error
            res_train.append(mean_squared_error(db_predict_Y,Y_train))
            res_test.append(mean_squared_error(db_predict_tY,Y_test))
            
            # calculating R2 score
            res_R2.append(r2_score(Y_train,db_predict_Y))
        plot_TT(i,res_train,res_test,res_R2,X,which_data,figNum,float(l),str)
        i=i+1
    plt.show()
        
# ## Ridge Regression and Cross Validation - Q2 
# ## Lasso with and without cross validation - Q3

def kfoldCV(X,Y,model):
    res=[]
    kf=KFold(n_splits=5)
    for train_i,test_i in kf.split(X):
        Xtr,Xt,Ytr,Yt = X[train_i],X[test_i],Y[train_i],Y[test_i]
        model.fit(Xtr,Ytr)
        predict_Yt=model.predict(Xt)
        #print("Train",len(train_i),"Test:",len(test_i))
        res.append(mean_squared_error(predict_Yt,Yt))
    #print("kf",len(kf))
    return np.mean(res)

################################################################# Q2 and Q3:Ridge and Lasso #####################################################
# regularizer coefficicents
L2 =np.array([0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5])
#percenatges
P2=np.array([0.01,0.1,0.2,0.3])

# col represent the column in subplot graph
# name is dataset name like 'B' =boston
# col represent the column in subplot graph
# name is dataset name like 'B' =boston 
#
#
# 
def plot_TT2(col,train,test,valid,r2,X,name,figNum,p,self,str):
    
    if name=='D':
        name='Diabetes'
        plt.figure(num=figNum,figsize=(18,10),dpi=100)
        plt.suptitle('Diabetes', fontsize=16)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.suptitle('Long Suptitle', fontsize=24)
    else:
        name='Boston'
        plt.figure(num=figNum,figsize=(18,10),dpi=100)
        plt.suptitle('Boston', fontsize=16)
        
    plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
    fig = plt.gcf()
    fig.canvas.set_window_title(str)
    plt.subplot(2,4,col)
    plt.grid(True)
    plt.xlabel('------- $\lambda$ Value------>',size=10)
    plt.ylabel('---------Error---------->',size=10)
    #plt.xlim(2000,4000)
    
    #ax =plt.gca()
    #ax.set_ylim([1000,4000])
    #ax.set_xlim([150,450])
    plt.plot(X,train,color='blue',marker='o',markersize=10,label='Train Error')
    plt.plot(X,test,color='red',marker='o',markersize=10,label='Test Error')
    #valid error plot
    if self!=4:
        plt.plot(X,train,'r--',markersize=10,label='Validation Error')
    #plt.axis('equal')
    #plt.gca().set_ylim(2500, 3500)
    plt.legend(loc=0)
    plt.title(name+'Train Data Percentage ={:.2f}'.format(p),size=7)
    
    
    
    # R2 plot
    plt.subplot(2,4,col+4)
    plt.grid(True)
    plt.xlabel('------- $\lambda$ Value------>',size=10)
    plt.ylabel('R2 Score')
    #plt.ylim(0.50,0.56)
    #plt.xlim(15,3100)
    #ax =plt.gca()
    #ax.set_ylim([0.4,0.5])
    #ax.set_xlim([150,450])
    plt.plot(X,r2,color='black',marker='o',markersize=10,label='R2 Score')
    #plt.gca().set_ylim(0.51, 0.56)
    plt.legend(loc=0)
    plt.title(name+':R2 Score ',size=7)


def generalized_q2(P2,L2,which_data,_data,figNum,Q,str):
    # declaring the empty array for each regularizer
    i=1
    db_X=_data.data
    db_Y=_data.target
    loop_count=1
    for p in np.nditer(P2):
        res_train=[]
        res_test=[]
        res_R2=[]
        valid=[]
        X=[]
        #diabetes_reg_model = linear_model.Ridge(alpha=l, fit_intercept=True, normalize=True, copy_X=True,tol=0.001)
        X_train,X_test,Y_train,Y_test = train_test_split(db_X,db_Y,test_size=p,random_state=0,shuffle=True)
        for l in np.nditer(L2):
            #if loop_count==1:
            if (Q == 3 and l >0):
                _model = linear_model.Lasso(alpha=l,fit_intercept=True,normalize=True,copy_X=True,random_state=0)
            else:
                _model = linear_model.Ridge(alpha=l, fit_intercept=True, normalize=True, copy_X=True,tol=0.001)
                #loop_count+=1
            X.append(l)
            _model.fit(X_train,Y_train)
            ## finding the predicted labels
            db_predict_Y = _model.predict(X_train)
            db_predict_tY=_model.predict(X_test)
            
            ##calculating mean sqaure error
            res_train.append(mean_squared_error(db_predict_Y,Y_train))
            res_test.append(mean_squared_error(db_predict_tY,Y_test))
            # calculating validation Score
            valid.append(kfoldCV(X_train,Y_train,_model))
            # calculating R2 score
            res_R2.append(r2_score(Y_train,db_predict_Y))
        #print("Train result:",res_train)
        #print("Test Result:",res_test)
        plot_TT2(i,res_train,res_test,valid,res_R2,X,which_data,figNum,(1-p)*100,1,str)
        i=i+1
    plt.show()


############################################################# Q4- Own Implementation ###############################################
def _buildModel(dataset,P,L,Q,name,figNum,str):
    X=dataset.data
    Y=dataset.target
    row =X.shape[0]
    col=X.shape[1]
    #print(row)
    one=np.ones([row,1],dtype=float) 

    # Calculating mean and variance
    #X
    mean_X=np.mean(X,axis=0)
    var_X=np.var(X,axis=0)[1]
    # Y
    mean_Y=np.mean(Y,axis=0)
    var_Y=np.var(Y,axis=0)
    
    X_copy=X
    Y_copy=Y
    
    #normalizing the data
    X_norm=(X_copy-mean_X)/var_X
    Y_norm=(Y_copy-mean_Y)/var_Y
    #Y_norm_mean = np.mean(Y_norm,axis=0)
    
    ## adding bias
    X_copy=np.hstack((one,X_copy))
    X_norm=np.hstack((one,X_norm))
    ## for linear regression
    if Q==1:
        ## finding W
        i=1
        for l in np.nditer(L):
            #print("Lamda=",l)
            res_train=[]
            res_test=[]
            res_R2=[]
            X=[]
            for p in np.nditer(P):
                
                # splitting original data
                X_dtrain,X_dtest,Y_dtrain,Y_dtest = train_test_split(X_copy,Y_copy,test_size=p,random_state=0,shuffle=True)
                
                #splitting the norm data
                X_train,X_test,Y_train,Y_test = train_test_split(X_norm,Y_norm,test_size=p,random_state=0,shuffle=True)
                
                #W = np.matmul((np.matmul(inv(np.dot(X_train.transpose(),X_train)+(l*np.identity(X_train.shape[1]))),(X_train.transpose()))),Y_train)
                
                #### calculating the regression coefficient #### 
                W= np.matmul(np.matmul(inv(np.matmul(X_train.transpose(),X_train)+l*np.identity(X_train.shape[1])),X_train.transpose()),Y_train)
                
                predicted_Ytr = np.dot(X_train,W)
                predicted_Yt =  np.dot(X_test,W)
                
                predicted_Ytr= (predicted_Ytr*var_Y)+mean_Y
                predicted_Yt= (predicted_Yt*var_Y)+ mean_Y
                
                train_error=np.sum(np.power((np.subtract(Y_dtrain,predicted_Ytr)),2),axis=0)/(X_train.shape[0])
                test_error =np.sum(np.power(np.subtract(Y_dtest,predicted_Yt),2),axis=0)/(X_test.shape[0])
                
                explained_variance = np.var(predicted_Ytr,axis=0)
                #print("Explained:",explained_variance,"Total:",var_Y)
                #print(var_Y)
                X.append(X_train.shape[0])
                res_train.append(train_error)
                res_test.append(test_error)
                res_R2.append(explained_variance/var_Y)
            #print("R2:",res_R2)
            plot_TT(i,res_train,res_test,res_R2,X,name,figNum,float(l),str)
            i=i+1

                
    ## for Ridge
    elif Q==2:
        i=1
        for p in np.nditer(P):
            res_train=[]
            res_test=[]
            res_R2=[]
            valid=[]
            X=[]
            
            X_dtrain,X_dtest,Y_dtrain,Y_dtest = train_test_split(X_copy,Y_copy,test_size=p,random_state=0,shuffle=True)
            X_train,X_test,Y_train,Y_test = train_test_split(X_norm,Y_norm,test_size=p,random_state=0,shuffle=True)
            
            
            for l in np.nditer(L):
                W= np.matmul(np.matmul(inv(np.matmul(X_train.transpose(),X_train)+l*np.identity(X_train.shape[1])),X_train.transpose()),Y_train)
                
                #predicting Y_train and Y_test 
                predicted_Ytr = np.dot(X_train,W)
                predicted_Yt =  np.dot(X_test,W)
                
                #denormalizing the predicted values
                predicted_Ytr= (predicted_Ytr*var_Y)+mean_Y
                predicted_Yt= (predicted_Yt*var_Y)+mean_Y
                
                train_error=np.sum(np.power((np.subtract(Y_dtrain,predicted_Ytr)),2),axis=0)/(X_train.shape[0])
                test_error =np.sum(np.power(np.subtract(Y_dtest,predicted_Yt),2),axis=0)/(X_test.shape[0])
                explained_variance=np.var(predicted_Ytr,axis=0)
                
                X.append(l)
                res_train.append(train_error)
                res_test.append(test_error)
                res_R2.append(explained_variance/var_Y)
            plot_TT2(i,res_train,res_test,valid,res_R2,X,name,figNum,(1-p)*100,4,str)
            i=i+1
    plt.show()



####################################################################################################################################
####################################################### -- MAIN -- #################################################################

if __name__== "__main__":

    # Question 1 plot
    generalized_q1(P,L,'D',db,1,"Fig1:Answer 1 Linear Regression:Diabetes")
    generalized_q1(P,L,'B',bt,2,'Fig2:Answer 1 Linear Regression:Boston')
    
    #Question 2 plot
    generalized_q2(P2,L2,'D',db,3,2,'Fig3:Answer 2-Ridge And Cross Validation:Diabetes')
    generalized_q2(P2,L2,'B',bt,4,2,'Fig4:Answer 2-Ridge And Cross Validation:Boston')
    
    ## Question 3 Plot##
    generalized_q2(P2,L2,'D',db,5,3,'Fig5:Answer 3-Lasso And Cross Validation:Diabetes')
    generalized_q2(P2,L2,'B',bt,6,3,'Fig6:Answer 3-Lasso And Cross Validation:Boston')
    
    ## Question 4 plot
    #### Q1 self build algorithm
    _buildModel(db,P,L,1,'D',7,'Fig7:Self:Answer 1 Linear Regression:Diabetes')
    _buildModel(bt,P,L,1,'B',8,'Fig8:Self:Answer 1 Linear Regression:Boston')
    #### Q2 self build algorithm
    _buildModel(db,P2,L2,2,'D',9,'Fig9:Self:Answer 2 Linear Regression:Diabetes')
    _buildModel(bt,P2,L2,2,'B',10,'Fig10:Self:Answer 2 Linear Regression:Boston')

    plt.close('all')










