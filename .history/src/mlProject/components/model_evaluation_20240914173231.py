from mlProject.config.configuration import *
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score
from mlProject.utils.common import *



class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
        
    def eval_metrics(self,actual,pred):
        acc_score=accuracy_score(actual,pred)
        
        return acc_score
    
    
    def save_metrics(self):
        try:
            test_data=pd.read_csv(self.config.test_data_path)
            model=joblib.load(self.config.model_path)
            
            target_col='Exited'
            
            X_test=test_data.drop(columns=['RowNumber','CustomerId','Surname','Exited'],axis=1)
            y_test=test_data[target_col]
            
            preprocess_obj=load_object(self.config.preprocess_path)
            
            X_test = preprocess_obj.transform(X_test)
            
            pred=model.predict(X_test)
            
            
            acc_score=self.eval_metrics()
            
            acc_score=acc_score(y_test,pred)
            logging.info(f"Model accuracy: {acc_score}")
                
                
            score = {'accuracy_score': acc_score}
            save_json(Path(self.config.metric_file_path), data=score)     
            
        ex       
        
        
        
            