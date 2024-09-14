from mlProject.config.configuration import *
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score
from mlProject.utils.common import *
from imblearn.combine import SMOTETomek, SMOTEENN

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self, actual, pred):
        acc_score = accuracy_score(actual, pred)
        return acc_score
    
    def save_metrics(self):
        try:
            # Load the test data and the saved model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)
            
            target_col = 'Exited'
            
            # Drop unnecessary columns
            X_test = test_data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
            y_test = test_data[target_col]
            
            # Load the preprocessor object
            preprocess_obj = load_object(self.config.preprocess_path)
            
            # Preprocess the test data
            X_test = preprocess_obj.transform(X_test)
            
            # Do NOT apply SMOTEENN on the test set
            pred = model.predict(X_test)
            
            # Evaluate the model
            acc_score = self.eval_metrics(y_test, pred)
            logging.info(f"Model accuracy: {acc_score}")
            
            # Save the accuracy score as a JSON file
            score = {'accuracy_score': acc_score}
            save_json(Path(self.config.metrics_file_path), data=score)     
            
        except Exception as e:
            logging.exception("Error during model evaluation.")
            raise e
