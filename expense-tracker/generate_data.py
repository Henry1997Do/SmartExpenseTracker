"""
Smart Expense Tracker - Data Generation Script
Generates synthetic expense data for training the ML model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define expense categories with realistic merchants and price ranges
EXPENSE_CATEGORIES = {
    'Food & Dining': {
        'weight': 0.25,
        'merchants': [
            'Starbucks', 'McDonald\'s', 'Chipotle', 'Pizza Hut', 'Subway',
            'Local Restaurant', 'Cafe Corner', 'Burger King', 'Taco Bell',
            'Panera Bread', 'Domino\'s Pizza', 'KFC', 'Five Guys',
            'Shake Shack', 'Dunkin Donuts', 'Wendy\'s', 'Chick-fil-A',
            'Panda Express', 'Olive Garden', 'Red Lobster', 'Applebee\'s',
            'Buffalo Wild Wings', 'Denny\'s', 'IHOP', 'Cheesecake Factory'
        ],
        'amount_range': (5, 80)
    },
    'Transportation': {
        'weight': 0.15,
        'merchants': [
            'Uber', 'Lyft', 'Gas Station', 'Shell', 'Chevron', 'Metro Card',
            'Parking Garage', 'BP Gas', 'Exxon', 'Mobil', 'Parking Meter',
            'City Transit', 'Taxi Cab', 'Car Wash', 'Toll Road',
            'Valero', 'Sunoco', 'Parking Lot', 'Public Transit Pass'
        ],
        'amount_range': (3, 100)
    },
    'Shopping': {
        'weight': 0.20,
        'merchants': [
            'Amazon', 'Target', 'Walmart', 'Best Buy', 'Shopping Mall',
            'Costco', 'Home Depot', 'Lowe\'s', 'IKEA', 'Macy\'s',
            'Nordstrom', 'Gap', 'H&M', 'Zara', 'Old Navy', 'TJ Maxx',
            'Ross', 'Marshall\'s', 'Kohl\'s', 'JCPenney', 'Sephora',
            'Ulta', 'Apple Store', 'Nike Store', 'Adidas Store'
        ],
        'amount_range': (10, 300)
    },
    'Entertainment': {
        'weight': 0.12,
        'merchants': [
            'Netflix', 'Spotify', 'Movie Theater', 'Bar & Grill',
            'Gaming Store', 'Concert Tickets', 'AMC Theaters', 'Regal Cinema',
            'Bowling Alley', 'Mini Golf', 'Arcade', 'Comedy Club',
            'Sports Event', 'Museum', 'Zoo', 'Amusement Park',
            'Disney+', 'Hulu', 'HBO Max', 'PlayStation Store',
            'Xbox Store', 'Steam Games', 'Apple Music', 'YouTube Premium'
        ],
        'amount_range': (10, 150)
    },
    'Bills & Utilities': {
        'weight': 0.10,
        'merchants': [
            'Electric Bill', 'Water Bill', 'Internet Service', 'Phone Bill',
            'Rent Payment', 'Insurance Premium', 'Gas Bill', 'Cable TV',
            'HOA Fees', 'Mortgage Payment', 'Property Tax', 'Car Insurance',
            'Health Insurance', 'Life Insurance', 'Home Insurance',
            'Verizon', 'AT&T', 'T-Mobile', 'Comcast', 'Spectrum'
        ],
        'amount_range': (30, 500)
    },
    'Healthcare': {
        'weight': 0.05,
        'merchants': [
            'Pharmacy', 'CVS', 'Walgreens', 'Doctor Visit', 'Hospital',
            'Dental Clinic', 'Rite Aid', 'Urgent Care', 'Lab Tests',
            'Eye Doctor', 'Physical Therapy', 'Chiropractor',
            'Mental Health', 'Prescription', 'Medical Supplies',
            'Health Clinic', 'Specialist Visit', 'X-Ray', 'MRI Scan'
        ],
        'amount_range': (15, 300)
    },
    'Groceries': {
        'weight': 0.10,
        'merchants': [
            'Whole Foods', 'Trader Joe\'s', 'Safeway', 'Kroger',
            'Farmers Market', 'Albertsons', 'Publix', 'Wegmans',
            'Stop & Shop', 'Food Lion', 'Giant Eagle', 'Aldi',
            'Lidl', 'Sprouts', 'Fresh Market', 'Harris Teeter',
            'Winn-Dixie', 'ShopRite', 'Acme', 'Vons'
        ],
        'amount_range': (20, 200)
    },
    'Travel': {
        'weight': 0.03,
        'merchants': [
            'Hotel Booking', 'Airbnb', 'Flight Ticket', 'Car Rental',
            'Booking.com', 'Expedia', 'Hotels.com', 'Marriott',
            'Hilton', 'Hyatt', 'Holiday Inn', 'Enterprise Rent-A-Car',
            'Hertz', 'Budget Car Rental', 'Delta Airlines', 'United Airlines',
            'American Airlines', 'Southwest Airlines', 'Travel Agency',
            'Vacation Package', 'Cruise Line', 'Resort Booking'
        ],
        'amount_range': (50, 800)
    }
}


def generate_expense_data(num_transactions=1500):
    """
    Generate synthetic expense data for training
    
    Args:
        num_transactions: Number of transactions to generate
        
    Returns:
        DataFrame with columns: date, description, amount, category
    """
    print(f"🔄 Generating {num_transactions} synthetic transactions...")
    
    # Calculate number of transactions per category based on weights
    transactions = []
    
    for category, details in EXPENSE_CATEGORIES.items():
        num_cat_transactions = int(num_transactions * details['weight'])
        
        for _ in range(num_cat_transactions):
            # Random date within past 365 days
            days_ago = random.randint(0, 365)
            transaction_date = datetime.now() - timedelta(days=days_ago)
            
            # Random merchant from category
            merchant = random.choice(details['merchants'])
            
            # Random amount within range
            min_amount, max_amount = details['amount_range']
            amount = round(random.uniform(min_amount, max_amount), 2)
            
            # Create transaction
            transactions.append({
                'date': transaction_date.strftime('%Y-%m-%d'),
                'description': merchant,
                'amount': amount,
                'category': category
            })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Shuffle to mix dates
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df


def print_statistics(df):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    print(f"\n✅ Total Transactions: {len(df)}")
    print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"💰 Total Spending: ${df['amount'].sum():,.2f}")
    print(f"📈 Average Transaction: ${df['amount'].mean():.2f}")
    print(f"📉 Median Transaction: ${df['amount'].median():.2f}")
    
    print("\n💳 Spending by Category:")
    print("-" * 60)
    category_stats = df.groupby('category').agg({
        'amount': ['sum', 'count', 'mean']
    }).round(2)
    category_stats.columns = ['Total ($)', 'Count', 'Avg ($)']
    category_stats = category_stats.sort_values('Total ($)', ascending=False)
    
    for category, row in category_stats.iterrows():
        percentage = (row['Total ($)'] / df['amount'].sum()) * 100
        print(f"{category:20s} | ${row['Total ($)']:>10,.2f} | "
              f"{int(row['Count']):>4} txns | ${row['Avg ($)']:>6.2f} avg | "
              f"{percentage:>5.1f}%")
    
    print("\n" + "="*60)


def main():
    """Main execution function"""
    print("\n" + "🎯 " * 20)
    print("SMART EXPENSE TRACKER - DATA GENERATION")
    print("🎯 " * 20 + "\n")
    
    # Generate data
    df = generate_expense_data(num_transactions=1500)
    
    # Save to CSV
    output_file = 'expenses.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to '{output_file}'")
    
    # Print statistics
    print_statistics(df)
    
    # Show sample data
    print("\n📋 Sample Transactions (First 10):")
    print("-" * 60)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "✨ " * 20)
    print("DATA GENERATION COMPLETE!")
    print("✨ " * 20 + "\n")
    print("➡️  Next step: Run 'python train_model.py' to train the ML model\n")


if __name__ == "__main__":
    main()
