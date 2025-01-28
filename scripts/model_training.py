import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_models(X_train, y_train):
    """Train multiple models and return them."""
    models = {
        'Gradient Boosting': GradientBoostingClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and print metrics."""
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics[name] = {'Accuracy': acc, 'F1 Score': f1}
        print(f"{name} - Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    return metrics

def save_model(model, file_path):
    """Save the best model to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data, preprocess_data, split_data

    # Load and preprocess data
    data = load_data('data/raw/hospital_readmissions.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    metrics = evaluate_models(trained_models, X_test, y_test)

    # Save the best model (example: Gradient Boosting)
    best_model = trained_models['Gradient Boosting']
    save_model(best_model, 'results/models/best_model.pkl')
