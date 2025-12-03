# 💰 Smart Personal Expense Tracker with AI Categorization

A complete, production-ready machine learning web application for tracking and analyzing personal expenses with AI-powered automatic categorization. Built to be completed in one day (6-8 hours) and perfect for beginners!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

✅ **AI-Powered Categorization** - Machine learning automatically classifies expenses into 8 categories  
✅ **Beautiful Dashboard** - Interactive visualizations with Plotly  
✅ **Real-time Insights** - AI-generated financial advice based on spending patterns  
✅ **Easy Data Entry** - Quick expense addition with smart predictions  
✅ **Advanced Analytics** - Detailed spending analysis and trends  
✅ **Data Export** - Download expenses and reports in CSV format  
✅ **Responsive Design** - Works on desktop, tablet, and mobile  
✅ **Fast Performance** - Optimized with caching for instant loading

## 🎯 What You'll Build

A fully functional web application that:

1. **Tracks Expenses** - Record daily transactions with date, description, amount, and category
2. **Predicts Categories** - Uses ML to automatically categorize expenses with 85-95% accuracy
3. **Visualizes Data** - Beautiful charts showing spending patterns and trends
4. **Provides Insights** - AI-powered recommendations for better financial management
5. **Analyzes Spending** - Detailed breakdowns by category, time period, and merchant
6. **Exports Data** - Download your financial data for external analysis

### 8 Expense Categories

- 🍔 **Food & Dining** - Restaurants, cafes, fast food
- 🚗 **Transportation** - Uber, gas, parking, public transit
- 🛍️ **Shopping** - Amazon, retail stores, online shopping
- 🎬 **Entertainment** - Netflix, movies, games, concerts
- 💡 **Bills & Utilities** - Rent, electricity, internet, insurance
- 🏥 **Healthcare** - Pharmacy, doctor visits, medical supplies
- 🥗 **Groceries** - Supermarkets, farmers markets
- ✈️ **Travel** - Hotels, flights, car rentals, vacation

## 📋 Prerequisites

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Python package installer (comes with Python)
- **Basic terminal/command line knowledge**
- **Text editor or IDE** (VS Code, PyCharm, or any editor)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:

- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning
- streamlit - Web framework
- plotly - Interactive visualizations
- matplotlib & seaborn - Additional plotting

### Step 2: Generate Training Data & Train Model

```bash
# Generate 1,500 synthetic transactions
python generate_data.py

# Train the ML model (takes 1-2 minutes)
python train_model.py
```

**What happens:**

- Creates `expenses.csv` with realistic transaction data
- Trains 3 ML models and selects the best one
- Saves trained models as `.pkl` files
- Generates accuracy reports and visualizations
- Expected accuracy: 85-95%

### Step 3: Launch the Web App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

**🎉 That's it! You're ready to use your expense tracker!**

---

## 📖 How to Use

### 📈 Dashboard Tab

Your financial command center:

- **Spending Overview** - Pie chart showing category breakdown
- **Top Categories** - Bar chart of highest spending areas
- **Monthly Trends** - Line chart tracking spending over time
- **Recent Transactions** - Table of latest 15 expenses
- **Quick Stats** - Total spent, average transaction, daily spending

**Sidebar Features:**

- Filter by date range
- View quick statistics
- See top spending category
- Get helpful tips

### ➕ Add Expense Tab

Two ways to add expenses:

**Manual Entry:**

1. Select date (defaults to today)
2. Enter description (e.g., "Starbucks coffee")
3. Enter amount
4. Choose category from dropdown
5. Click "Save Expense"

**Bulk Import:**

1. Prepare CSV with columns: date, description, amount, category
2. Upload file in the expandable section
3. Preview data
4. Click "Import Data"

### 🤖 AI Categorize Tab

Let AI do the work:

1. **Enter Description** - Type transaction description
2. **Predict** - Click "Predict Category" button
3. **View Results** - See predicted category with confidence scores
4. **Quick Save** - Add date and amount, then save with one click

**Try Examples:**

- Click any example button to test the AI
- See confidence scores for all categories
- Understand how the model makes decisions

### 💡 AI Insights Tab

Get personalized financial advice:

1. **Select Category** - Choose category to analyze
2. **Generate Insights** - Click button for AI analysis
3. **View Metrics** - See total spent, transactions, averages
4. **Read Advice** - Get AI-generated spending recommendations
5. **Analyze Trends** - View spending patterns over time
6. **Top Merchants** - See where you spend most in each category

**AI Advice Examples:**

- High spending alerts (>30% of budget)
- Optimization tips (20-30% of budget)
- Positive reinforcement (<20% of budget)

### 📊 Analytics Tab

Deep dive into your finances:

**Time Period Analysis:**

- Daily, Weekly, or Monthly views
- Area chart showing spending trends
- Compare different time periods

**Category Analytics:**

- Average transaction by category
- Transaction count by category
- Identify spending patterns

**Data Export:**

- Download complete expense data as CSV
- Generate summary report with statistics
- Use data in Excel or other tools

---

## 🔄 Daily Usage Workflow

### Morning Routine (2 minutes)

1. Open app: `streamlit run app.py`
2. Check dashboard for yesterday's spending
3. Review any high-spending alerts

### Throughout the Day (30 seconds per expense)

1. Make a purchase
2. Open "Add Expense" or "AI Categorize" tab
3. Enter description and amount
4. Let AI predict category or select manually
5. Save

### Evening Review (5 minutes)

1. Check "Dashboard" for daily summary
2. Review "AI Insights" for spending patterns
3. Adjust budget if needed
4. Plan for tomorrow

### Weekly Analysis (15 minutes)

1. Use "Analytics" tab for weekly trends
2. Compare spending across categories
3. Identify areas for improvement
4. Export data for detailed analysis

---

## 🧪 Testing the Model

After training, test with these descriptions:

| Description      | Expected Category |
| ---------------- | ----------------- |
| Starbucks coffee | Food & Dining     |
| Uber ride        | Transportation    |
| Amazon purchase  | Shopping          |
| Netflix          | Entertainment     |
| Electric bill    | Bills & Utilities |
| CVS pharmacy     | Healthcare        |
| Whole Foods      | Groceries         |
| Hotel booking    | Travel            |

**Model Performance Metrics:**

- Overall Accuracy: 85-95%
- Precision: High for all categories
- Recall: Consistent across categories
- F1-Score: Balanced performance

---

## 🎨 Customization Guide

### Add More Categories

1. **Edit `generate_data.py`:**

   ```python
   EXPENSE_CATEGORIES = {
       'Your New Category': {
           'weight': 0.05,
           'merchants': ['Merchant1', 'Merchant2'],
           'amount_range': (10, 100)
       }
   }
   ```

2. **Regenerate data and retrain:**
   ```bash
   python generate_data.py
   python train_model.py
   ```

### Adjust Data Volume

Change number of transactions in `generate_data.py`:

```python
generate_expense_data(num_transactions=3000)  # Default: 1500
```

More data = Better accuracy (but longer training time)

### Customize Colors

Edit the CSS in `app.py`:

```python
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #your-color1, #your-color2);
    }
    </style>
""", unsafe_allow_html=True)
```

### Change ML Model

In `train_model.py`, modify model parameters:

```python
RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=10,      # Deeper trees
    random_state=42
)
```

---

## 📊 Project Statistics

- **Lines of Code**: ~1,200
- **Files**: 7
- **ML Models Trained**: 3 (Naive Bayes, Logistic Regression, Random Forest)
- **Features Extracted**: 100 TF-IDF features
- **Training Time**: 1-2 minutes
- **App Load Time**: <2 seconds
- **Categories**: 8
- **Sample Data**: 1,500 transactions

---

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution:**

```bash
pip install -r requirements.txt --upgrade
```

### Issue: "expenses.csv not found"

**Solution:**

```bash
python generate_data.py
```

### Issue: "Model files not found"

**Solution:**

```bash
python train_model.py
```

### Issue: Low model accuracy (<80%)

**Solution:**

1. Generate more data:
   ```python
   generate_expense_data(num_transactions=3000)
   ```
2. Retrain model:
   ```bash
   python train_model.py
   ```

### Issue: Streamlit won't start

**Solution:**

```bash
# Check if port 8501 is in use
lsof -ti:8501 | xargs kill -9

# Try different port
streamlit run app.py --server.port 8502
```

### Issue: Slow performance

**Solution:**

- Clear Streamlit cache: Click menu → "Clear cache"
- Reduce data volume in date filter
- Close other browser tabs

---

## 📁 File Structure

```
expense-tracker/
│
├── requirements.txt          # Python dependencies
├── generate_data.py          # Synthetic data generation
├── train_model.py           # ML model training
├── app.py                   # Main Streamlit application
├── README.md                # This file
├── DAY_TIMELINE.md          # Hour-by-hour completion guide
├── quick_start.sh           # Automated setup script
│
├── expenses.csv             # Generated expense data
├── expense_model.pkl        # Trained ML model
├── vectorizer.pkl           # TF-IDF vectorizer
├── label_encoder.pkl        # Category encoder
├── confusion_matrix.png     # Model evaluation chart
└── category_accuracy.png    # Per-category accuracy chart
```

---

## 💡 Tips for Success

### For Beginners

1. **Follow the timeline** - Use `DAY_TIMELINE.md` for structured progress
2. **Read error messages** - They usually tell you exactly what's wrong
3. **Test frequently** - Run the app after each major change
4. **Use examples** - Try the example transactions to understand features
5. **Ask for help** - Check documentation and online resources

### For Better Results

1. **Quality descriptions** - More detailed = Better predictions
2. **Consistent naming** - Use same merchant names (e.g., always "Starbucks")
3. **Regular updates** - Add expenses daily for accurate trends
4. **Review insights** - Act on AI recommendations
5. **Export regularly** - Backup your data weekly

### For Advanced Users

1. **Experiment with models** - Try different ML algorithms
2. **Feature engineering** - Add amount-based features
3. **Custom categories** - Tailor to your spending habits
4. **API integration** - Connect to bank APIs for auto-import
5. **Deploy online** - Use Streamlit Cloud for remote access

---

## 📚 Learning Outcomes

After completing this project, you'll understand:

### Machine Learning

- Text classification with TF-IDF
- Model training and evaluation
- Hyperparameter tuning
- Model persistence with pickle

### Data Science

- Data generation and preprocessing
- Exploratory data analysis
- Statistical analysis
- Data visualization

### Web Development

- Streamlit framework
- Interactive dashboards
- Form handling
- State management

### Software Engineering

- Project structure
- Code organization
- Error handling
- User experience design

---

## 🎯 Expected Results

After setup, you should have:

✅ Working web application accessible at localhost:8501  
✅ ML model with 85-95% accuracy  
✅ 1,500 synthetic transactions for testing  
✅ Interactive dashboard with 5 tabs  
✅ AI-powered categorization  
✅ Financial insights and recommendations  
✅ Data export functionality  
✅ Beautiful visualizations

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data generated (`python generate_data.py`)
- [ ] Model trained with >85% accuracy (`python train_model.py`)
- [ ] App launches successfully (`streamlit run app.py`)
- [ ] Can add new expenses
- [ ] AI categorization works
- [ ] Dashboard displays correctly
- [ ] All visualizations render
- [ ] Can export data
- [ ] Tested on sample transactions

---

## 🚀 Next Steps

### Enhancements to Consider

1. **Budget Tracking** - Set monthly budgets per category
2. **Recurring Expenses** - Auto-add regular bills
3. **Multi-currency** - Support different currencies
4. **Mobile App** - Create mobile version
5. **Bank Integration** - Auto-import from bank APIs
6. **Notifications** - Email alerts for overspending
7. **Goals** - Set and track savings goals
8. **Comparison** - Compare with previous months
9. **Sharing** - Export reports as PDF
10. **Authentication** - Add user login system

### Deployment Options

1. **Streamlit Cloud** - Free hosting for Streamlit apps
2. **Heroku** - Deploy with custom domain
3. **AWS/GCP** - Enterprise-grade hosting
4. **Docker** - Containerize for easy deployment

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

Built with:

- [Streamlit](https://streamlit.io/) - Web framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation

---

## 📞 Support

Having issues? Check:

1. **README.md** - This file (you're reading it!)
2. **DAY_TIMELINE.md** - Step-by-step guide
3. **Troubleshooting section** - Common issues above
4. **Error messages** - Read them carefully
5. **Documentation** - Check library docs

---

## 🎉 Congratulations!

You've built a complete machine learning web application! This project demonstrates:

- ✅ Machine Learning skills
- ✅ Data Science capabilities
- ✅ Web Development proficiency
- ✅ Software Engineering practices

**Perfect for your portfolio, resume, or personal use!**

---

**Made with ❤️ for learners and builders**

**Happy Tracking! 💰📊🚀**
