import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
card = pd.read_csv(r'/kaggle/input/creditcardfraud/creditcard.csv')
card.head()
card.info()
card.describe() #statistic of the data set with regards to each variable
card.shape() #to know the number of observations and variables in the card dataset

#Data Quality Check

round(100 * (card.isnull().sum()/len(card)),2).sort_values(ascending=False) #Check for missing values in colomn.

round(100 * (card.isnull().sum(axis=1)/len(card)),2).sort_values(ascending=False) #check for missing values in row

card_d=card.copy() #create a duplicate of data set 
card_d.drop_duplicates(subset=None, inplace=True) # check if there are duplicate observations and drop them.

card_d.shape # since there is a difference in observations and duplicates have been removed

card=card_d #Assign removed dup[licate dataset as the original
del card_d

#DATA EXPLORATORY ANALYSIS

#Construct a histogram for for all the variables in the dataset. 
#set color of chat as midnight blue 
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()  
    plt.show()
draw_histograms(card,card.columns,8,4)

#graphical representation of the distribution of the "class" variable in the dataset.
ax=sns.countplot(x='Class',data=card);
ax.set_yscale('log')

#Plot the correlation of the data set

plt.figure(figsize = (40,10))
sns.heatmap(card.corr(), annot = True, cmap="tab20c")
plt.show()

#The heatmap clearly shows which all variable are multicollinear in nature,
#and which variable have high collinearity with the target variable.

#LOGISTIC REGRESSION

#For the model our estimators are the all the variables excluding Time 
# and the dependent variable is "Class"
estimators=[ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X1 = card[estimators]
y = card['Class']

col=X1.columns[:-1]

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()

results_logit.summary()

#Use backward elimation to remove all insignificant pvalues

def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.0001):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(X,card.Class,col)
result.summary()

#Interpreting the result
params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))

new_features=card[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20','V21', 'V22', 'V23', 'V25', 'V26', 'V27','Class']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

#Model Accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#Confussion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

#The confusion matrix shows 56642+54 = 56698 correct predictions and 19+41=50 incorrect ones.
#True Positives: 59
#True Negatives: 56641
#False Positives: 9 (Type I error)
#False Negatives: 41 ( Type II error)

#MODEL EVALUATION
print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) =       ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy =                  ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) =       ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) =       ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) =               ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) =               ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

#it is clear that the model is highly specific than sensitive. The negative values are predicted more accurately than the positives.

y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of Not Fraud (0)','Prob of Fraud (1)'])
y_pred_prob_df.head()

#Since the model is predicting Fraud too many type II errors is not advisable.
#A False Negative ( ignoring the probability of Fraud when there actualy is one) is more dangerous than a False Positive in this case.
#Hence inorder to increase the sensitivity, threshold can be lowered.

from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    
    #ROC CURVE
    
   from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Fraud classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

#A common way to visualize the trade-offs of different thresholds is by using an ROC curve, 
#a plot of the true positive rate (# true positives/ total # positives) versus the false positive rate (# false positives / total # negatives) for all possible choices of thresholds.
#A model with good classification accuracy should have significantly more true positives than false positives at all thresholds.
#he optimum position for roc curve is towards the top left corner where the specificity and sensitivity are at optimum levels

#ARea Under The Curve (AUC) The area under the ROC curve quantifies model classification accuracy; the higher the area, 
#the greater the disparity between true and false positives, and the stronger the model in classifying members of the training dataset. 
#An area of 0.5 corresponds to a model that performs no better than random classification and a good classifier stays as far away from that as possible. 
#An area of 1 is ideal. The closer the AUC to 1 the better

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_prob_yes[:,1])

#Conclusion
#All attributes selected after the elimination process show Pvalues lower than 5% and thereby suggesting significant role in the fraud Prediction.
#The Area under the ROC curve is 95.71 which is good
#Overall model could be improved with more data.
