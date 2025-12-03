"""
Smart Expense Tracker - Main Web Application
A beautiful ML-powered expense tracking application built with Streamlit
Version: 2.0 - Updated with modern dark theme and sample data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS (Inspired by Copilot)
st.markdown("""
    <style>
    /* Dark background */
    .stApp {
        background-color: #0f1419;
        color: #e1e8ed;
    }
    
    /* Main content area */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Modern tabs - Rounded button style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: none;
        display: flex;
        width: 100%;
        padding: 1rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        flex: 1;
        font-size: 1rem;
        font-weight: 600;
        color: #8b95a1;
        background-color: #1a202c;
        border: 2px solid #2d3748;
        border-radius: 25px;
        text-align: center;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #243447;
        color: #e1e8ed;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        border: 2px solid #60a5fa;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.5);
    }
    
    /* Card styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60a5fa;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b95a1;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Buttons - Rounded pill style */
    .stButton button {
        background-color: #1a202c;
        color: #8b95a1;
        border: 2px solid #2d3748;
        border-radius: 25px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton button:hover {
        background-color: #243447;
        border-color: #60a5fa;
        color: #e1e8ed;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: 2px solid #60a5fa;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.5);
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    .stButton button[kind="secondary"] {
        background-color: #2d3748;
        color: #cbd5e0;
        border: 2px solid #4a5568;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #374151;
        border-color: #6b7280;
    }
    
    /* Input fields - Blue theme to match dropdown */
    input, textarea {
        background-color: #1e3a5f !important;
        border: 1px solid #60a5fa !important;
        color: #60a5fa !important;
        border-radius: 20px !important;
        padding: 0.75rem 1rem !important;
        caret-color: #60a5fa !important;
        box-shadow: 0 0 0 0 transparent !important;
        transition: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        /* Remove ring utilities */
        --tw-ring-offset-shadow: 0 0 transparent !important;
        --tw-ring-shadow: 0 0 transparent !important;
        --tw-ring-color: transparent !important;
        --tw-ring-offset-width: 0px !important;
        --tw-ring-offset-color: transparent !important;
    }
    
    /* Placeholder text */
    input::placeholder, textarea::placeholder {
        color: #60a5fa !important;
        opacity: 0.7 !important;
    }
    
    /* Remove ALL focus effects */
    input:focus, textarea:focus,
    input:focus-visible, textarea:focus-visible,
    input:active, textarea:active {
        border: 1px solid #60a5fa !important;
        box-shadow: 0 0 0 0 transparent !important;
        outline: none !important;
        outline-width: 0 !important;
        outline-offset: 0 !important;
        background-color: #1e3a5f !important;
        /* Remove ring on focus */
        --tw-ring-offset-shadow: 0 0 transparent !important;
        --tw-ring-shadow: 0 0 transparent !important;
        --tw-ring-color: transparent !important;
    }
    
    input:hover, textarea:hover {
        border: 1px solid #60a5fa !important;
        background-color: #2d5a8f !important;
        box-shadow: none !important;
    }
    
    /* Override Streamlit input wrapper - Remove all borders and shadows */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div,
    .stTextInput > div,
    .stNumberInput > div,
    .stTextArea > div {
        box-shadow: none !important;
        border: none !important;
        background: transparent !important;
    }
    
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stTextArea > div > div:focus-within,
    .stTextInput > div:focus-within,
    .stNumberInput > div:focus-within,
    .stTextArea > div:focus-within {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
        background: transparent !important;
    }
    
    /* Target Streamlit's inner input container */
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-baseweb="base-input"]:focus-within,
    [data-baseweb="base-input"] > div:focus-within {
        box-shadow: none !important;
        border: none !important;
        background: transparent !important;
    }
    
    /* Nuclear option - hide all wrapper borders */
    .stTextInput, .stNumberInput, .stTextArea, .stSelectbox {
        --tw-ring-shadow: 0 0 transparent !important;
        --tw-ring-offset-shadow: 0 0 transparent !important;
        overflow: hidden !important;
        padding: 0 !important;
    }
    
    /* Clip the outer glow using overflow */
    .stTextInput > div,
    .stNumberInput > div,
    .stTextArea > div {
        overflow: hidden !important;
        border-radius: 20px !important;
    }
    
    /* Date and number inputs */
    input[type="date"], input[type="number"] {
        padding: 0.75rem 1rem !important;
        color: #60a5fa !important;
        background-color: #1e3a5f !important;
        border: 1px solid #60a5fa !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
    }
    
    /* Select dropdowns - Override Streamlit defaults */
    .stSelectbox select,
    div[data-baseweb="select"] select,
    select {
        cursor: pointer !important;
        background-color: #1e3a5f !important;
        color: #60a5fa !important;
        border: 1px solid #60a5fa !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox select:focus,
    div[data-baseweb="select"] select:focus,
    select:focus {
        background-color: #1e3a5f !important;
        border-color: #60a5fa !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stSelectbox select:hover,
    div[data-baseweb="select"] select:hover,
    select:hover {
        background-color: #2d5a8f !important;
        border-color: #60a5fa !important;
    }
    
    /* Select option styling */
    .stSelectbox select option,
    div[data-baseweb="select"] select option,
    select option {
        background-color: #1e3a5f !important;
        color: #60a5fa !important;
    }
    
    /* Streamlit selectbox container */
    div[data-baseweb="select"] > div {
        background-color: #1e3a5f !important;
        border-color: #60a5fa !important;
        border-radius: 20px !important;
    }
    
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {
        background-color: #2d5a8f !important;
        border-color: #60a5fa !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e1e8ed !important;
        font-weight: 700 !important;
    }
    
    /* Text */
    p, span, div, label {
        color: #cbd5e0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0e14;
        border-right: 1px solid #2d3748;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e1e8ed;
    }
    
    /* Category badges */
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-food { background-color: #fbbf24; color: #78350f; }
    .badge-transport { background-color: #60a5fa; color: #1e3a8a; }
    .badge-shopping { background-color: #f472b6; color: #831843; }
    .badge-bills { background-color: #a78bfa; color: #4c1d95; }
    .badge-entertainment { background-color: #f87171; color: #7f1d1d; }
    .badge-groceries { background-color: #34d399; color: #064e3b; }
    .badge-healthcare { background-color: #fb923c; color: #7c2d12; }
    .badge-travel { background-color: #818cf8; color: #312e81; }
    
    /* Transaction cards */
    .transaction-card {
        background-color: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }
    .transaction-card:hover {
        border-color: #60a5fa;
        transform: translateX(4px);
    }
    
    /* Chart background */
    .stPlotlyChart {
        background-color: #1a202c;
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load or train ML models"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(script_dir, 'expense_model.pkl')
        vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
        encoder_path = os.path.join(script_dir, 'label_encoder.pkl')

        # Try to load existing models
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            return model, vectorizer, label_encoder
        else:
            # Train models from scratch using expense data
            st.info(
                "🤖 Training ML models for the first time... This will take a moment.")
            df = load_expense_data()

            if len(df) < 10:
                st.warning(
                    "⚠️ Not enough data to train models. Need at least 10 transactions.")
                return None, None, None

            # Prepare features
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            X = vectorizer.fit_transform(df['description'].str.lower())

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df['category'])

            # Train model
            model = MultinomialNB()
            model.fit(X, y)

            # Save models for next time
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(label_encoder, f)
                st.success("✅ Models trained and saved successfully!")
            except Exception as e:
                st.warning(f"Models trained but couldn't be saved: {e}")

            return model, vectorizer, label_encoder

    except Exception as e:
        st.error(
            f"⚠️ Error with ML models: {str(e)}. AI Categorize feature will be disabled.")
        return None, None, None


@st.cache_data
def load_expense_data():
    """Load expense data from CSV"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'expenses.csv')
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'description', 'amount', 'category'])


def save_expense_data(df):
    """Save expense data to CSV"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'expenses.csv')
    df_save = df.copy()
    df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d')
    df_save.to_csv(csv_path, index=False)
    st.cache_data.clear()


def predict_category(description, model, vectorizer, label_encoder):
    """Predict expense category from description"""
    if model is None:
        return None, None

    X = vectorizer.transform([description])
    prediction = model.predict(X)[0]
    category = label_encoder.inverse_transform([prediction])[0]

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        prob_dict = {label_encoder.inverse_transform([i])[0]: prob
                     for i, prob in enumerate(probabilities)}
        return category, prob_dict

    return category, None


def get_spending_insights(category, amount, df):
    """Generate AI-powered spending insights"""
    if len(df) == 0:
        return "Not enough data for insights."

    category_data = df[df['category'] == category]
    total_spent = df['amount'].sum()
    category_spent = category_data['amount'].sum()
    category_percentage = (category_spent / total_spent) * 100

    avg_transaction = category_data['amount'].mean()

    # Generate insights based on spending patterns
    insights = []

    if category_percentage > 30:
        insights.append(f"⚠️ **High Spending Alert**: You're spending {category_percentage:.1f}% of your budget on {category}. "
                        f"This is significantly above average. Consider reviewing your {category.lower()} expenses.")
        insights.append(
            f"💡 **Tip**: Try to reduce {category.lower()} spending by 10-15% next month.")
    elif category_percentage > 20:
        insights.append(f"📊 **Moderate Spending**: {category} represents {category_percentage:.1f}% of your total expenses. "
                        f"This is within a reasonable range.")
        insights.append(
            f"💡 **Tip**: Look for opportunities to optimize your {category.lower()} expenses.")
    else:
        insights.append(
            f"✅ **Good Control**: You're managing {category} expenses well at {category_percentage:.1f}% of your budget.")
        insights.append(
            f"💡 **Tip**: Keep up the good work maintaining this spending level!")

    if amount > avg_transaction * 1.5:
        insights.append(f"📈 **Above Average**: This transaction (${amount:.2f}) is higher than your average "
                        f"{category.lower()} expense (${avg_transaction:.2f}).")

    return "\n\n".join(insights)


def main():
    """Main application function"""

    # Load models and data
    model, vectorizer, label_encoder = load_models()
    df = load_expense_data()

    # Check if setup is complete
    if model is None:
        st.error("⚠️ **Setup Required!**")
        st.warning("""
        The ML models haven't been trained yet. Please complete the setup:

        1. Run `python generate_data.py` to create training data
        2. Run `python train_model.py` to train the ML model
        3. Refresh this page

        See README.md for detailed instructions.
        """)
        return

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/money-bag.png", width=80)
        st.title("💰 Expense Tracker")
        st.markdown("---")

        # Quick stats
        if len(df) > 0:
            st.subheader("📊 Quick Stats")
            total_spent = df['amount'].sum()
            avg_transaction = df['amount'].mean()
            total_transactions = len(df)

            # Calculate daily average
            date_range = (df['date'].max() - df['date'].min()).days + 1
            daily_avg = total_spent / date_range if date_range > 0 else 0

            st.metric("Total Spent", f"${total_spent:,.2f}")
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
            st.metric("Total Transactions", f"{total_transactions:,}")
            st.metric("Daily Average", f"${daily_avg:.2f}")

            st.markdown("---")

            # Date range filter
            st.subheader("📅 Filter by Date")
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) == 2:
                df = df[(df['date'].dt.date >= date_range[0]) &
                        (df['date'].dt.date <= date_range[1])]

            st.markdown("---")

            # Top spending category
            top_category = df.groupby('category')['amount'].sum().idxmax()
            top_amount = df.groupby('category')['amount'].sum().max()
            st.subheader("🏆 Top Category")
            st.info(f"**{top_category}**\n\n${top_amount:,.2f}")
        else:
            st.info("👋 Welcome! Add your first expense to get started.")

        st.markdown("---")
        st.caption(
            "💡 **Tip**: Use AI Categorize to automatically classify expenses!")

    # Main content
    st.title("💰 Smart Personal Expense Tracker")
    st.markdown("### AI-Powered Financial Management Dashboard")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard",
        "➕ Add Expense",
        "🤖 AI Categorize",
        "💡 AI Insights",
        "📊 Analytics"
    ])

    # TAB 1: Dashboard
    with tab1:
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.header("📈 Financial Dashboard")
        with col_header2:
            if st.button("🔄 Refresh Data", type="secondary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        if len(df) == 0:
            st.info(
                "📝 No expenses recorded yet. Add your first expense in the 'Add Expense' tab!")
        else:
            # Welcome message
            st.markdown(f"### Welcome back! Here's your financial overview.")
            st.markdown("---")

            # Two-column layout for charts
            col1, col2 = st.columns(2)

            with col1:
                # Pie chart: Spending by category
                category_spending = df.groupby(
                    'category')['amount'].sum().reset_index()
                category_spending = category_spending.sort_values(
                    'amount', ascending=False)

                # Modern color palette
                colors = ['#60a5fa', '#34d399', '#fbbf24', '#f472b6',
                          '#a78bfa', '#fb923c', '#818cf8', '#f87171']

                fig_pie = go.Figure(data=[go.Pie(
                    labels=category_spending['category'],
                    values=category_spending['amount'],
                    hole=0.5,
                    marker=dict(colors=colors, line=dict(
                        color='#1a202c', width=2)),
                    textfont=dict(size=14, color='white'),
                    textposition='inside'
                )])
                fig_pie.update_layout(
                    title=dict(text="Spending by Category",
                               font=dict(size=18, color='#e1e8ed')),
                    height=400,
                    showlegend=True,
                    paper_bgcolor='#1a202c',
                    plot_bgcolor='#1a202c',
                    font=dict(color='#cbd5e0'),
                    legend=dict(
                        bgcolor='#1a202c',
                        font=dict(color='#cbd5e0', size=12),
                        orientation='v',
                        yanchor='middle',
                        y=0.5,
                        xanchor='left',
                        x=1.05
                    ),
                    margin=dict(l=20, r=150, t=60, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Horizontal bar chart: Top 5 categories
                top_5 = category_spending.head(5)

                fig_bar = go.Figure(data=[go.Bar(
                    y=top_5['category'],
                    x=top_5['amount'],
                    orientation='h',
                    marker=dict(
                        color=['#2563eb', '#1d4ed8',
                               '#1e40af', '#1e3a8a', '#1e3a5f'],
                        line=dict(color='#60a5fa', width=1)
                    ),
                    text=[f'${x:,.2f}' for x in top_5['amount']],
                    textposition='outside',
                    textfont=dict(color='#60a5fa', size=12)
                )])
                fig_bar.update_layout(
                    title=dict(text="Top 5 Spending Categories",
                               font=dict(size=18, color='#e1e8ed')),
                    xaxis=dict(
                        title="Amount ($)",
                        gridcolor='#2d3748',
                        color='#cbd5e0'
                    ),
                    yaxis=dict(
                        title="Category",
                        color='#cbd5e0'
                    ),
                    height=400,
                    paper_bgcolor='#1a202c',
                    plot_bgcolor='#1a202c',
                    font=dict(color='#cbd5e0')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Monthly spending trend
            st.markdown("---")
            st.subheader("📅 Monthly Spending Trend")

            df_monthly = df.copy()
            df_monthly['month'] = df_monthly['date'].dt.to_period(
                'M').astype(str)
            monthly_spending = df_monthly.groupby(
                'month')['amount'].sum().reset_index()

            fig_line = go.Figure(data=[go.Scatter(
                x=monthly_spending['month'],
                y=monthly_spending['amount'],
                mode='lines+markers',
                line=dict(color='#60a5fa', width=3),
                marker=dict(size=12, color='#2563eb',
                            line=dict(color='#60a5fa', width=2)),
                fill='tozeroy',
                fillcolor='rgba(96, 165, 250, 0.1)'
            )])
            fig_line.update_layout(
                title=dict(text="Monthly Spending Trend",
                           font=dict(size=18, color='#e1e8ed')),
                xaxis=dict(
                    title="Month",
                    gridcolor='#2d3748',
                    color='#cbd5e0'
                ),
                yaxis=dict(
                    title="Amount ($)",
                    gridcolor='#2d3748',
                    color='#cbd5e0'
                ),
                height=400,
                hovermode='x unified',
                paper_bgcolor='#1a202c',
                plot_bgcolor='#1a202c',
                font=dict(color='#cbd5e0')
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Recent transactions
            st.markdown("---")
            st.subheader("📝 Recent Transactions")

            recent_df = df.sort_values('date', ascending=False).head(15).copy()

            # Show confirmation message if delete was clicked (above transactions)
            for idx, row in recent_df.iterrows():
                if st.session_state.get(f"delete_confirm_{idx}", False):
                    # Compact inline confirmation with message and buttons together
                    col_msg, col_yes, col_no, col_space = st.columns([
                                                                     4, 1, 1, 4])

                    with col_msg:
                        st.markdown(f"""
                            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 12px; border-radius: 4px;">
                                <span style="color: #856404;">🗑️ Delete <strong>{row['description']}</strong> (${row['amount']:.2f})?</span>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_yes:
                        if st.button("Yes", key=f"yes_{idx}", type="primary", use_container_width=True):
                            df = df.drop(idx)
                            save_expense_data(df)
                            st.session_state[f"delete_confirm_{idx}"] = False
                            st.success("✅ Deleted!")
                            st.rerun()
                    with col_no:
                        if st.button("No", key=f"no_{idx}", use_container_width=True):
                            st.session_state[f"delete_confirm_{idx}"] = False
                            st.rerun()

                    st.markdown("<br>", unsafe_allow_html=True)
                    break  # Only show one confirmation at a time

            # Column headers
            col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 1])
            with col1:
                st.markdown("**Date**")
            with col2:
                st.markdown("**Description**")
            with col3:
                st.markdown("**Amount**")
            with col4:
                st.markdown("**Category**")
            with col5:
                st.markdown("**Action**")

            st.markdown("---")

            # Display transactions with delete buttons
            for idx, row in recent_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 1])

                with col1:
                    st.text(row['date'].strftime('%Y-%m-%d'))
                with col2:
                    st.text(row['description'])
                with col3:
                    st.text(f"${row['amount']:.2f}")
                with col4:
                    st.text(row['category'])
                with col5:
                    # Delete button with custom styling
                    delete_clicked = st.button(
                        "🗑",
                        key=f"delete_{idx}",
                        help="Delete transaction",
                        type="secondary"
                    )
                    if delete_clicked:
                        st.session_state[f"delete_confirm_{idx}"] = True
                        st.rerun()

    # TAB 2: Add Expense
    with tab2:
        st.header("➕ Add New Expense")

        with st.form("add_expense_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                expense_date = st.date_input(
                    "Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
                expense_description = st.text_input(
                    "Description",
                    placeholder="e.g., Starbucks coffee"
                )

            with col2:
                expense_amount = st.number_input(
                    "Amount ($)",
                    min_value=0.01,
                    step=0.01,
                    format="%.2f"
                )
                expense_category = st.selectbox(
                    "Category",
                    options=sorted(label_encoder.classes_)
                )

            submitted = st.form_submit_button(
                "💾 Save Expense", type="primary", use_container_width=True)

            if submitted:
                if expense_description and expense_amount > 0:
                    # Add new expense
                    new_expense = pd.DataFrame({
                        'date': [pd.to_datetime(expense_date)],
                        'description': [expense_description],
                        'amount': [expense_amount],
                        'category': [expense_category]
                    })

                    df = pd.concat([df, new_expense], ignore_index=True)
                    save_expense_data(df)

                    st.success(
                        f"✅ Expense added successfully! ${expense_amount:.2f} for {expense_description}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Please fill in all fields correctly.")

        # Bulk import section
        st.markdown("---")
        with st.expander("📤 Bulk Import from CSV"):
            st.markdown("""
            Upload a CSV file with columns: `date`, `description`, `amount`, `category`

            **Format**:
            - Date: YYYY-MM-DD
            - Description: Text
            - Amount: Number
            - Category: One of the predefined categories
            """)

            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    import_df = pd.read_csv(uploaded_file)
                    import_df['date'] = pd.to_datetime(import_df['date'])

                    st.write(f"Preview ({len(import_df)} rows):")
                    st.dataframe(import_df.head(), use_container_width=True)

                    if st.button("Import Data", type="primary"):
                        df = pd.concat([df, import_df], ignore_index=True)
                        save_expense_data(df)
                        st.success(
                            f"✅ Successfully imported {len(import_df)} expenses!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error importing file: {str(e)}")

    # TAB 3: AI Categorize
    with tab3:
        st.header("🤖 AI-Powered Expense Categorization")
        st.markdown("Let our AI model automatically categorize your expenses!")

        # Initialize session state for description input
        if 'description_text' not in st.session_state:
            st.session_state.description_text = ""

        # Input for prediction in horizontal layout
        col1, col2 = st.columns([4, 1], gap="small")

        with col1:
            description_input = st.text_input(
                "Enter transaction description",
                value=st.session_state.description_text,
                placeholder="e.g., Starbucks morning coffee"
            )
            # Update session state with current input
            if description_input != st.session_state.description_text:
                st.session_state.description_text = description_input

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.button(
                "🔮 Predict Category",
                type="primary",
                use_container_width=True
            )

        if predict_button:
            if not description_input:
                st.warning("⚠️ Please enter a transaction description first!")
            else:
                predicted_category, probabilities = predict_category(
                    description_input, model, vectorizer, label_encoder
                )

                if predicted_category:
                    st.success(
                        f"### 🎯 Predicted Category: **{predicted_category}**")

                    if probabilities:
                        st.markdown("---")
                        st.subheader("📊 Confidence Scores")

                        # Sort probabilities
                        prob_df = pd.DataFrame(
                            list(probabilities.items()),
                            columns=['Category', 'Confidence']
                        ).sort_values('Confidence', ascending=False)

                        # Create horizontal bar chart
                        fig_conf = go.Figure(data=[go.Bar(
                            y=prob_df['Category'],
                            x=prob_df['Confidence'] * 100,
                            orientation='h',
                            marker=dict(
                                color=prob_df['Confidence'] * 100,
                                colorscale='Blues',
                                showscale=False
                            ),
                            text=[
                                f'{x:.1f}%' for x in prob_df['Confidence'] * 100],
                            textposition='auto'
                        )])
                        fig_conf.update_layout(
                            xaxis_title="Confidence (%)",
                            yaxis_title="Category",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                    # Quick save section
                    st.markdown("---")
                    st.subheader("💾 Quick Save")

                    with st.form("quick_save_form"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            quick_date = st.date_input(
                                "Date", value=datetime.now())
                        with col2:
                            quick_amount = st.number_input(
                                "Amount ($)", min_value=0.01, step=0.01)
                        with col3:
                            quick_category = st.selectbox(
                                "Category",
                                options=sorted(label_encoder.classes_),
                                index=list(sorted(label_encoder.classes_)).index(
                                    predicted_category)
                            )

                        if st.form_submit_button("💾 Save with AI Category", type="primary"):
                            new_expense = pd.DataFrame({
                                'date': [pd.to_datetime(quick_date)],
                                'description': [description_input],
                                'amount': [quick_amount],
                                'category': [quick_category]
                            })

                            df = pd.concat([df, new_expense],
                                           ignore_index=True)
                            save_expense_data(df)

                            st.success(
                                f"✅ Saved ${quick_amount:.2f} as {quick_category}!")
                            st.balloons()

        # Example descriptions
        st.markdown("---")
        st.subheader("💡 Try These Examples")

        examples = [
            "Starbucks coffee", "Uber ride downtown", "Amazon Prime subscription",
            "Netflix monthly", "Electric bill payment", "CVS pharmacy",
            "Whole Foods groceries", "Marriott hotel", "Shell gas station",
            "Target shopping"
        ]

        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(example, key=f"example_{idx}", use_container_width=True):
                    st.session_state.description_text = example
                    st.rerun()

    # TAB 4: AI Insights
    with tab4:
        st.header("💡 AI-Powered Financial Insights")

        if len(df) == 0:
            st.info("📝 Add some expenses first to get personalized insights!")
        else:
            # Category selector
            selected_category = st.selectbox(
                "Select a category to analyze",
                options=sorted(df['category'].unique())
            )

            if st.button("🧠 Generate Insights", type="primary"):
                category_data = df[df['category'] == selected_category]

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                total_spent = category_data['amount'].sum()
                num_transactions = len(category_data)
                avg_transaction = category_data['amount'].mean()
                percentage = (total_spent / df['amount'].sum()) * 100

                with col1:
                    st.metric("Total Spent", f"${total_spent:,.2f}")
                with col2:
                    st.metric("Transactions", f"{num_transactions}")
                with col3:
                    st.metric("Average", f"${avg_transaction:.2f}")
                with col4:
                    st.metric("% of Budget", f"{percentage:.1f}%")

                st.markdown("---")

                # AI-generated insights
                st.subheader("🤖 AI Analysis")
                insights = get_spending_insights(
                    selected_category, avg_transaction, df)
                st.markdown(insights)

                st.markdown("---")

                # Spending trend for category
                st.subheader(f"📈 {selected_category} Spending Trend")

                category_monthly = category_data.copy()
                category_monthly['month'] = category_monthly['date'].dt.to_period(
                    'M').astype(str)
                monthly_cat_spending = category_monthly.groupby(
                    'month')['amount'].sum().reset_index()

                fig_cat_trend = go.Figure(data=[go.Scatter(
                    x=monthly_cat_spending['month'],
                    y=monthly_cat_spending['amount'],
                    mode='lines+markers',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=10),
                    fill='tozeroy'
                )])
                fig_cat_trend.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    height=350
                )
                st.plotly_chart(fig_cat_trend, use_container_width=True)

                # Top merchants in category
                st.markdown("---")
                st.subheader(f"🏪 Top 5 Merchants in {selected_category}")

                top_merchants = category_data.groupby('description')['amount'].agg([
                    'sum', 'count']).reset_index()
                top_merchants.columns = [
                    'Merchant', 'Total Spent', 'Transactions']
                top_merchants = top_merchants.sort_values(
                    'Total Spent', ascending=False).head(5)
                top_merchants['Total Spent'] = top_merchants['Total Spent'].apply(
                    lambda x: f"${x:.2f}")

                st.dataframe(
                    top_merchants, use_container_width=True, hide_index=True)

    # TAB 5: Analytics
    with tab5:
        st.header("📊 Advanced Analytics")

        if len(df) == 0:
            st.info("📝 Add some expenses first to view analytics!")
        else:
            # Time period selector
            time_period = st.selectbox(
                "Select time period",
                options=["Daily", "Weekly", "Monthly"]
            )

            # Prepare data based on time period
            df_time = df.copy()
            if time_period == "Daily":
                df_time['period'] = df_time['date'].dt.strftime('%Y-%m-%d')
            elif time_period == "Weekly":
                df_time['period'] = df_time['date'].dt.to_period(
                    'W').astype(str)
            else:  # Monthly
                df_time['period'] = df_time['date'].dt.to_period(
                    'M').astype(str)

            period_spending = df_time.groupby(
                'period')['amount'].sum().reset_index()

            # Area chart
            fig_area = go.Figure(data=[go.Scatter(
                x=period_spending['period'],
                y=period_spending['amount'],
                mode='lines',
                fill='tozeroy',
                line=dict(color='#8b5cf6', width=2),
                fillcolor='rgba(139, 92, 246, 0.3)'
            )])
            fig_area.update_layout(
                title=f"{time_period} Spending Over Time",
                xaxis_title="Period",
                yaxis_title="Amount ($)",
                height=400
            )
            st.plotly_chart(fig_area, use_container_width=True)

            st.markdown("---")

            # Two columns for additional analytics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💵 Average Transaction by Category")
                avg_by_category = df.groupby(
                    'category')['amount'].mean().sort_values(ascending=True)

                fig_avg = go.Figure(data=[go.Bar(
                    y=avg_by_category.index,
                    x=avg_by_category.values,
                    orientation='h',
                    marker=dict(color='#3b82f6'),
                    text=[f'${x:.2f}' for x in avg_by_category.values],
                    textposition='auto'
                )])
                fig_avg.update_layout(
                    xaxis_title="Average Amount ($)",
                    height=400
                )
                st.plotly_chart(fig_avg, use_container_width=True)

            with col2:
                st.subheader("📊 Transaction Count by Category")
                count_by_category = df.groupby(
                    'category').size().sort_values(ascending=True)

                fig_count = go.Figure(data=[go.Bar(
                    y=count_by_category.index,
                    x=count_by_category.values,
                    orientation='h',
                    marker=dict(color='#10b981'),
                    text=count_by_category.values,
                    textposition='auto'
                )])
                fig_count.update_layout(
                    xaxis_title="Number of Transactions",
                    height=400
                )
                st.plotly_chart(fig_count, use_container_width=True)

            # Export section
            st.markdown("---")
            st.subheader("📥 Export Data")

            col1, col2 = st.columns(2)

            with col1:
                # Export as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Export summary report
                summary = f"""
EXPENSE TRACKER SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS
------------------
Total Spent: ${df['amount'].sum():,.2f}
Total Transactions: {len(df)}
Average Transaction: ${df['amount'].mean():.2f}
Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}

SPENDING BY CATEGORY
--------------------
"""
                for category in sorted(df['category'].unique()):
                    cat_data = df[df['category'] == category]
                    summary += f"\n{category}:\n"
                    summary += f"  Total: ${cat_data['amount'].sum():,.2f}\n"
                    summary += f"  Transactions: {len(cat_data)}\n"
                    summary += f"  Average: ${cat_data['amount'].mean():.2f}\n"

                st.download_button(
                    label="📊 Download Summary Report",
                    data=summary,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
