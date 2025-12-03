"""
Smart Expense Tracker - Model Training Script
Trains machine learning models to automatically categorize expenses
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def load_data():
    """Load expense data from CSV"""
    print("📂 Loading expense data...")
    try:
        df = pd.read_csv('expenses.csv')
        print(f"✅ Loaded {len(df)} transactions")
        return df
    except FileNotFoundError:
        print("❌ Error: expenses.csv not found!")
        print("➡️  Please run 'python generate_data.py' first")
        exit(1)


def prepare_features(df):
    """
    Prepare features and labels for training

    Args:
        df: DataFrame with expense data

    Returns:
        X_train, X_test, y_train, y_test, vectorizer, label_encoder
    """
    print("\n🔧 Preparing features...")

    # Extract features (descriptions) and labels (categories)
    X = df['description'].values
    y = df['category'].values

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )

    # Transform text to TF-IDF features
    X_tfidf = vectorizer.fit_transform(X)
    print(f"✅ Created {X_tfidf.shape[1]} TF-IDF features")

    # Encode category labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"✅ Encoded {len(label_encoder.classes_)} categories")

    # Split data into train and test sets (80/20 split with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and select the best one

    Args:
        X_train, X_test, y_train, y_test: Training and test data

    Returns:
        best_model, best_model_name, best_accuracy
    """
    print("\n🤖 Training machine learning models...")
    print("="*60)

    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': accuracy}

        print(f"✅ {name} Accuracy: {accuracy*100:.2f}%")

    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    print("\n" + "="*60)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Accuracy: {best_accuracy*100:.2f}%")
    print("="*60)

    return best_model, best_model_name, best_accuracy


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance with detailed metrics

    Args:
        model: Trained model
        X_test, y_test: Test data
        label_encoder: Label encoder for category names
    """
    print("\n📊 Detailed Model Evaluation:")
    print("="*60)

    # Make predictions
    y_pred = model.predict(X_test)

    # Classification report
    print("\n📈 Classification Report:")
    print("-"*60)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Expense Category Prediction',
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()

    # Calculate per-category accuracy
    category_accuracy = []
    for i, category in enumerate(label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum()
            category_accuracy.append({'Category': category, 'Accuracy': acc})

    # Plot category-wise accuracy
    acc_df = pd.DataFrame(category_accuracy).sort_values(
        'Accuracy', ascending=True)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(acc_df['Accuracy'])
    bars = plt.barh(acc_df['Category'], acc_df['Accuracy'], color=colors)
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Category', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy by Category', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.0)

    # Add percentage labels
    for i, (bar, acc) in enumerate(zip(bars, acc_df['Accuracy'])):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{acc*100:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('category_accuracy.png', dpi=300, bbox_inches='tight')
    print("✅ Category accuracy chart saved as 'category_accuracy.png'")
    plt.close()

    print("\n" + "="*60)


def save_models(model, vectorizer, label_encoder):
    """Save trained models to disk"""
    print("\n💾 Saving models...")

    with open('expense_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved as 'expense_model.pkl'")

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Vectorizer saved as 'vectorizer.pkl'")

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Label encoder saved as 'label_encoder.pkl'")


def test_predictions(model, vectorizer, label_encoder):
    """Test model with sample descriptions"""
    print("\n🧪 Testing Model with Sample Descriptions:")
    print("="*60)

    test_cases = [
        "Starbucks coffee",
        "Uber ride",
        "Amazon purchase",
        "Netflix subscription",
        "Electric bill payment",
        "CVS pharmacy",
        "Whole Foods groceries",
        "Hotel booking",
        "McDonald's lunch",
        "Gas station fill up",
        "Target shopping",
        "Spotify premium",
        "Water bill",
        "Walgreens prescription",
        "Trader Joe's",
        "Flight ticket"
    ]

    for description in test_cases:
        # Transform description
        X_test = vectorizer.transform([description])

        # Predict category
        prediction = model.predict(X_test)[0]
        category = label_encoder.inverse_transform([prediction])[0]

        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[0]
            confidence = proba[prediction] * 100
            print(
                f"'{description:30s}' → {category:20s} (Confidence: {confidence:.1f}%)")
        else:
            print(f"'{description:30s}' → {category:20s}")

    print("="*60)


def main():
    """Main execution function"""
    print("\n" + "🤖 " * 20)
    print("SMART EXPENSE TRACKER - MODEL TRAINING")
    print("🤖 " * 20 + "\n")

    # Load data
    df = load_data()

    # Prepare features
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = prepare_features(
        df)

    # Train models
    best_model, best_model_name, best_accuracy = train_models(
        X_train, X_test, y_train, y_test)

    # Evaluate best model
    evaluate_model(best_model, X_test, y_test, label_encoder)

    # Save models
    save_models(best_model, vectorizer, label_encoder)

    # Test predictions
    test_predictions(best_model, vectorizer, label_encoder)

    print("\n" + "✨ " * 20)
    print("MODEL TRAINING COMPLETE!")
    print("✨ " * 20 + "\n")

    if best_accuracy >= 0.85:
        print("🎉 Excellent! Model accuracy is above 85%")
    else:
        print("⚠️  Model accuracy is below 85%. Consider generating more data.")

    print("\n➡️  Next step: Run 'streamlit run app.py' to start the web application\n")


if __name__ == "__main__":
    main()
