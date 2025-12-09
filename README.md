# 💰 Smart Expense Tracker (AI-Powered)

A streamlined Streamlit web app for tracking personal expenses with AI-powered automatic categorization. Generate sample data, train an ML model, and launch an interactive dashboard with insights and visualizations.

## 🌟 Highlights

- **AI Categorization**: Predicts expense categories with ~85–95% accuracy
- **Interactive Dashboard**: Category breakdowns, trends, recent transactions
- **AI Insights**: Data-driven tips and category analysis
- **Fast Add & Import**: Manual entry or CSV bulk import
- **Exportable Data**: Download transactions and reports

## 🧰 Tech Stack

- Python, Streamlit, Plotly, Pandas, scikit-learn
- TF-IDF text features + ML models (Naive Bayes, Logistic Regression, Random Forest)

## 🚀 Quick Start

From the `expense-tracker/` folder:

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

Then open http://localhost:8501.

## 🗂️ Repository Layout

- `expense-tracker/` — App source, data generation, model training, and docs
  - `README.md` — Full documentation and usage guide
  - `GET_STARTED.md` — Step-by-step setup guide

## 📚 Learn More

- Full docs: `expense-tracker/README.md`
- Getting started: `expense-tracker/GET_STARTED.md`

## 📝 License

MIT License.

## 🙏 Acknowledgments

Built with Streamlit, scikit-learn, Plotly, and Pandas.
