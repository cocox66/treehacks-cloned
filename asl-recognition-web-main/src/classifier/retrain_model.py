import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import string

# Create labels
ascii_string = string.ascii_lowercase.upper() + "?"
labels_dict = {idx: value for idx, value in enumerate(ascii_string)}

def train_new_model():
    try:
        # Load the data
        with open("../data/data.pickle", "rb") as f:
            data_dict = pickle.load(f)
            
        # Convert data to numpy arrays
        X = np.array(data_dict["data"])
        y = np.array(data_dict["labels"])
        
        # Train a new model with current scikit-learn version
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        model.fit(X, y)
        
        # Save the model
        with open("classify_letter_model.p", "wb") as f:
            pickle.dump({"model": model}, f, protocol=4)
            
        print("Successfully trained and saved new model!")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    train_new_model() 