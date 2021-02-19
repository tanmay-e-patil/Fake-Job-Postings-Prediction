from sklearn.metrics import confusion_matrix
import numpy as np

# Get all the evaluations for the model
def get_results(model,X,y):
    predictions = model.predict(X)
    tn,fn,fp,tp = confusion_matrix(y, np.round(predictions)).ravel()
    print(tn,fn,fp,tp)
    accuracy = model.evaluate(X,y,verbose=2)[1]
    print("Accuracy",accuracy)
    sensitivity = tp/(tp+fn)
    print("Sensitivity",sensitivity)
    specificity = tn/(tn+fp)
    print("Specificity",specificity)
    precision = tp/(tp+fp)
    print("Precision",precision)
    recall =  sensitivity
    print("Recall",recall)
    f1_score = 2*precision*recall/(precision+recall)
    print("F1 score",f1_score)
    return accuracy, sensitivity, specificity, precision, f1_score
    
    
    