import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Handling missing values
    df = df.fillna(method='ffill')
    
    # Feature engineering (example: converting categorical to numerical)
    df = pd.get_dummies(df, drop_first=True)
    
    # Splitting features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    data = load_data('data/raw/hospital_readmissions.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
