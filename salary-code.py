import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Load your data (replace with your actual data loading)
# data = pd.read_csv('your_dataset.csv')

# Data Preprocessing
def preprocess_data(data):
    # Label Encoding for categorical columns
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 
                       'race', 'gender', 'native-country']
    
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])
    
    # Separate features and target
    x = data.drop(columns=['income'])
    y = data['income']
    
    return x, y

# Calculate class weights for handling imbalance
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

# Feature Engineering and Scaling
def scale_features(x):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

# Model Training and Evaluation
def train_and_evaluate_models(xtrain, xtest, ytrain, ytest, class_weights):
    results = {}
    
    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn.fit(xtrain, ytrain)
    predict1 = knn.predict(xtest)
    
    results['KNN'] = {
        'model': knn,
        'accuracy': accuracy_score(ytest, predict1),
        'classification_report': classification_report(ytest, predict1),
        'confusion_matrix': confusion_matrix(ytest, predict1),
        'roc_auc': roc_auc_score(ytest, knn.predict_proba(xtest)[:, 1]) if hasattr(knn, 'predict_proba') else None
    }
    
    # Logistic Regression with class weighting
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, solver='liblinear')
    lr.fit(xtrain, ytrain)
    predict2 = lr.predict(xtest)
    
    results['LogisticRegression'] = {
        'model': lr,
        'accuracy': accuracy_score(ytest, predict2),
        'classification_report': classification_report(ytest, predict2),
        'confusion_matrix': confusion_matrix(ytest, predict2),
        'roc_auc': roc_auc_score(ytest, lr.predict_proba(xtest)[:, 1])
    }
    
    # Enhanced MLP Classifier
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(50, 25), 
                       random_state=23, max_iter=2000, early_stopping=True,
                       activation='relu', alpha=0.0001, learning_rate='adaptive')
    clf.fit(xtrain, ytrain)
    predict3 = clf.predict(xtest)
    
    results['MLPClassifier'] = {
        'model': clf,
        'accuracy': accuracy_score(ytest, predict3),
        'classification_report': classification_report(ytest, predict3),
        'confusion_matrix': confusion_matrix(ytest, predict3),
        'roc_auc': roc_auc_score(ytest, clf.predict_proba(xtest)[:, 1])
    }
    
    # Random Forest with class weighting
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=23)
    rf.fit(xtrain, ytrain)
    predict4 = rf.predict(xtest)
    
    results['RandomForest'] = {
        'model': rf,
        'accuracy': accuracy_score(ytest, predict4),
        'classification_report': classification_report(ytest, predict4),
        'confusion_matrix': confusion_matrix(ytest, predict4),
        'roc_auc': roc_auc_score(ytest, rf.predict_proba(xtest)[:, 1])
    }
    
    return results

# Main Execution
def main():
    # Load and preprocess data
    x, y = preprocess_data(data)
    
    # Calculate class weights
    class_weights = get_class_weights(y)
    
    # Scale features
    x_scaled = scale_features(x)
    
    # Train-test split with stratification to maintain class distribution
    xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, 
                                                   test_size=0.2, 
                                                   random_state=23, 
                                                   stratify=y)
    
    # Train and evaluate models
    results = train_and_evaluate_models(xtrain, xtest, ytrain, ytest, class_weights)
    
    # Display results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
    
    # Feature Importance Analysis (for tree-based models)
    if 'RandomForest' in results:
        feature_importances = results['RandomForest']['model'].feature_importances_
        features = data.drop(columns=['income']).columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title('Top 10 Important Features (Random Forest)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
