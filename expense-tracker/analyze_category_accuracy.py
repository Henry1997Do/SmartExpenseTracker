"""
Category Accuracy Analysis for Expenses
Analyzes expense categories and generates accuracy metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
import pickle
import os


def load_expenses():
    """Load expense data from CSV"""
    print("📂 Loading expense data...")
    try:
        df = pd.read_csv('expenses.csv')
        print(f"✅ Loaded {len(df)} transactions")
        return df
    except FileNotFoundError:
        print("❌ Error: expenses.csv not found!")
        exit(1)


def analyze_categories(df):
    """Analyze category distribution and statistics"""
    print("\n📊 Category Analysis:")
    print("="*60)

    # Count transactions per category
    category_counts = df['category'].value_counts()

    # Calculate statistics per category
    category_stats = []
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        stats = {
            'Category': category,
            'Count': len(cat_data),
            'Total_Amount': cat_data['amount'].sum(),
            'Avg_Amount': cat_data['amount'].mean(),
            'Min_Amount': cat_data['amount'].min(),
            'Max_Amount': cat_data['amount'].max()
        }
        category_stats.append(stats)

    stats_df = pd.DataFrame(category_stats).sort_values(
        'Count', ascending=False)

    print("\nCategory Statistics:")
    print("-"*60)
    for _, row in stats_df.iterrows():
        print(f"{row['Category']:20s} | Count: {row['Count']:2d} | "
              f"Total: ${row['Total_Amount']:7.2f} | Avg: ${row['Avg_Amount']:6.2f}")

    return stats_df, category_counts


def calculate_category_accuracy(df):
    """
    Calculate category accuracy based on keyword matching
    This simulates how well expenses are categorized
    """
    print("\n🎯 Calculating Category Accuracy...")
    print("="*60)

    # Define keywords for each category
    category_keywords = {
        'Food & Dining': ['starbucks', 'coffee', 'pizza', 'mcdonald', 'lunch', 'dinner',
                          'restaurant', 'chipotle', 'subway', 'taco', 'domino'],
        'Transportation': ['uber', 'lyft', 'ride', 'gas', 'station', 'bus', 'train',
                           'parking', 'taxi', 'fuel'],
        'Groceries': ['grocery', 'walmart', 'target', 'whole foods', 'costco',
                      'produce', 'market', 'shopping'],
        'Entertainment': ['netflix', 'spotify', 'disney', 'hbo', 'movie', 'theater',
                          'subscription', 'streaming'],
        'Bills & Utilities': ['bill', 'electric', 'water', 'gas bill', 'internet',
                              'utility', 'payment'],
        'Shopping': ['h&m', 'amazon', 'zara', 'nike', 'forever', 'purchase', 'shirt',
                     'jacket', 'shoes', 'dress'],
        'Healthcare': ['doctor', 'pharmacy', 'prescription', 'dentist', 'medical',
                       'health', 'copay', 'eye exam', 'cvs', 'walgreens']
    }

    # Calculate accuracy for each category
    category_accuracy = []

    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        keywords = category_keywords.get(category, [])

        # Count how many descriptions contain relevant keywords
        correct = 0
        for desc in cat_data['description']:
            desc_lower = desc.lower()
            if any(keyword in desc_lower for keyword in keywords):
                correct += 1

        accuracy = correct / len(cat_data) if len(cat_data) > 0 else 0
        category_accuracy.append({
            'Category': category,
            'Accuracy': accuracy,
            'Correct': correct,
            'Total': len(cat_data)
        })

    acc_df = pd.DataFrame(category_accuracy).sort_values(
        'Accuracy', ascending=False)

    print("\nCategory Accuracy Results:")
    print("-"*60)
    for _, row in acc_df.iterrows():
        print(f"{row['Category']:20s} | Accuracy: {row['Accuracy']*100:5.1f}% | "
              f"Correct: {row['Correct']:2d}/{row['Total']:2d}")

    return acc_df


def plot_category_distribution(category_counts):
    """Plot category distribution pie chart"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(range(len(category_counts)))

    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
    plt.title('Expense Distribution by Category',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Category distribution chart saved as 'category_distribution.png'")
    plt.close()


def generate_confusion_matrix(df):
    """Generate and save a confusion matrix using the trained ML model if available.

    Falls back to keyword rules if model artifacts are missing.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'expense_model.pkl')
        vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
        encoder_path = os.path.join(script_dir, 'label_encoder.pkl')

        use_model = os.path.exists(model_path) and os.path.exists(
            vectorizer_path) and os.path.exists(encoder_path)

        if use_model:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)

            X = vectorizer.transform(df['description'].astype(str).str.lower())
            y_true = df['category'].astype(str)

            # Align true labels to encoder space; unseen labels will be filtered out
            valid_mask = y_true.isin(label_encoder.classes_)
            if not valid_mask.all():
                y_true = y_true[valid_mask]
                X = X[valid_mask]

            y_pred_idx = model.predict(X)
            y_pred = label_encoder.inverse_transform(y_pred_idx)

            labels = list(label_encoder.classes_)
        else:
            # Fallback: keyword-based predictions (reuse rules from calculate_category_accuracy)
            category_keywords = {
                'Food & Dining': ['starbucks', 'coffee', 'pizza', 'mcdonald', 'lunch', 'dinner',
                                  'restaurant', 'chipotle', 'subway', 'taco', 'domino'],
                'Transportation': ['uber', 'lyft', 'ride', 'gas', 'station', 'bus', 'train',
                                   'parking', 'taxi', 'fuel'],
                'Groceries': ['grocery', 'walmart', 'target', 'whole foods', 'costco',
                              'produce', 'market', 'shopping'],
                'Entertainment': ['netflix', 'spotify', 'disney', 'hbo', 'movie', 'theater',
                                  'subscription', 'streaming'],
                'Bills & Utilities': ['bill', 'electric', 'water', 'gas bill', 'internet',
                                      'utility', 'payment'],
                'Shopping': ['h&m', 'amazon', 'zara', 'nike', 'forever', 'purchase', 'shirt',
                             'jacket', 'shoes', 'dress'],
                'Healthcare': ['doctor', 'pharmacy', 'prescription', 'dentist', 'medical',
                               'health', 'copay', 'eye exam', 'cvs', 'walgreens']
            }

            def rule_predict(desc: str):
                d = str(desc).lower()
                best_cat = None
                best_hits = 0
                for cat, kws in category_keywords.items():
                    hits = sum(1 for kw in kws if kw in d)
                    if hits > best_hits:
                        best_hits = hits
                        best_cat = cat
                return best_cat if best_cat is not None else 'Other'

            y_true = df['category'].astype(str)
            y_pred = df['description'].astype(str).apply(rule_predict)
            labels = sorted(set(y_true) | set(y_pred))

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(max(8, len(labels)*0.8), max(6, len(labels)*0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Expense Category Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    except Exception as e:
        print(f"❌ Failed to generate confusion matrix: {e}")


def plot_category_accuracy(acc_df):
    """Plot category accuracy bar chart"""
    plt.figure(figsize=(12, 8))

    # Sort by accuracy for better visualization
    acc_df_sorted = acc_df.sort_values('Accuracy', ascending=True)

    # Create color gradient based on accuracy
    colors = plt.cm.RdYlGn(acc_df_sorted['Accuracy'])

    bars = plt.barh(acc_df_sorted['Category'],
                    acc_df_sorted['Accuracy'], color=colors)
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Category', fontsize=12, fontweight='bold')
    plt.title('Category Accuracy Analysis', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.0)

    # Add percentage labels
    for i, (bar, acc) in enumerate(zip(bars, acc_df_sorted['Accuracy'])):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{acc*100:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('category_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Category accuracy chart saved as 'category_accuracy_analysis.png'")
    plt.close()


def plot_spending_by_category(stats_df):
    """Plot total spending by category"""
    plt.figure(figsize=(12, 8))

    stats_df_sorted = stats_df.sort_values('Total_Amount', ascending=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(stats_df_sorted)))
    bars = plt.barh(stats_df_sorted['Category'],
                    stats_df_sorted['Total_Amount'], color=colors)

    plt.xlabel('Total Amount ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Category', fontsize=12, fontweight='bold')
    plt.title('Total Spending by Category', fontsize=16, fontweight='bold')

    # Add amount labels
    for bar, amount in zip(bars, stats_df_sorted['Total_Amount']):
        plt.text(amount + 5, bar.get_y() + bar.get_height()/2,
                 f'${amount:.2f}', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('spending_by_category.png', dpi=300, bbox_inches='tight')
    print("✅ Spending by category chart saved as 'spending_by_category.png'")
    plt.close()


def generate_summary_report(df, stats_df, acc_df):
    """Generate a text summary report"""
    print("\n📝 Generating Summary Report...")

    report = []
    report.append("="*70)
    report.append("EXPENSE CATEGORY ACCURACY REPORT")
    report.append("="*70)
    report.append(f"\nTotal Transactions: {len(df)}")
    report.append(f"Total Amount: ${df['amount'].sum():.2f}")
    report.append(f"Average Transaction: ${df['amount'].mean():.2f}")
    report.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    report.append(f"\nNumber of Categories: {df['category'].nunique()}")
    report.append("\n" + "-"*70)
    report.append("CATEGORY STATISTICS")
    report.append("-"*70)

    for _, row in stats_df.iterrows():
        report.append(f"\n{row['Category']}:")
        report.append(f"  Transactions: {row['Count']}")
        report.append(f"  Total Amount: ${row['Total_Amount']:.2f}")
        report.append(f"  Average: ${row['Avg_Amount']:.2f}")
        report.append(
            f"  Range: ${row['Min_Amount']:.2f} - ${row['Max_Amount']:.2f}")

    report.append("\n" + "-"*70)
    report.append("CATEGORY ACCURACY")
    report.append("-"*70)

    for _, row in acc_df.sort_values('Accuracy', ascending=False).iterrows():
        report.append(f"{row['Category']:20s}: {row['Accuracy']*100:5.1f}% "
                      f"({row['Correct']}/{row['Total']} correct)")

    overall_accuracy = acc_df['Correct'].sum() / acc_df['Total'].sum()
    report.append("\n" + "-"*70)
    report.append(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    report.append("="*70)

    report_text = "\n".join(report)

    with open('category_accuracy_report.txt', 'w') as f:
        f.write(report_text)

    print("✅ Summary report saved as 'category_accuracy_report.txt'")
    print("\n" + report_text)


def main():
    """Main execution function"""
    print("\n" + "📊 " * 20)
    print("CATEGORY ACCURACY ANALYSIS")
    print("📊 " * 20 + "\n")

    # Load data
    df = load_expenses()

    # Analyze categories
    stats_df, category_counts = analyze_categories(df)

    # Calculate accuracy
    acc_df = calculate_category_accuracy(df)

    # Generate visualizations
    print("\n📈 Generating Visualizations...")
    print("="*60)
    plot_category_distribution(category_counts)
    plot_category_accuracy(acc_df)
    plot_spending_by_category(stats_df)
    generate_confusion_matrix(df)

    # Generate summary report
    generate_summary_report(df, stats_df, acc_df)

    print("\n" + "✨ " * 20)
    print("ANALYSIS COMPLETE!")
    print("✨ " * 20 + "\n")

    print("📁 Generated Files:")
    print("  • category_distribution.png")
    print("  • category_accuracy_analysis.png")
    print("  • spending_by_category.png")
    print("  • category_accuracy_report.txt")
    print()


if __name__ == "__main__":
    main()
