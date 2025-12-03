import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import re
from textblob import TextBlob

# Visualization 1: Distribution of Customer Ratings (Histogram)
def plot_rating_distribution(df): 
    # Ensure date column is in datetime format for the time series plot
    df['review_date'] = pd.to_datetime(df['review_date'])
    # --- Visualization 1: Distribution of Ratings (Histogram) ---
    # Define custom color mapping: green for high ratings, red for low ratings
    rating_colors = {
        1: "#d62728",  # red
        2: "#ff7f0e",  # orange
        3: "#ffeb3b",  # yellow
        4: "#2ca02c",  # green
        5: "#1a7f37",  # dark green
    }
    # Map colors to ratings in the data
    df['rating_color'] = df['rating'].map(rating_colors)

    fig1 = px.histogram(
        df,
        x='rating',
        nbins=10,
        title='Distribution of Customer Ratings',
        labels={'rating': 'Rating (1-5)'},
        color='rating',
        color_discrete_map=rating_colors,
        template=plotly_template
    )
    fig1.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray', dtick=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    fig1.update_traces(marker_line_width=1, marker_line_color='black')
    return fig1

# Visualization 2: Distribution of Ratings by Occasion (Box Plot)
def plot_ratings_by_occasion(df): 
    fig2 = px.box(
        df, 
        x='rented_for', 
        y='rating', 
        color='rented_for',
        title='Distribution of Ratings by Occasion',
        labels={'rented_for': 'Occasion', 'rating': 'Rating (0-10)'},
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )
    fig2.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    return fig2

# Visualization 3: Average Rating Trend Over Time (Line Chart)
def plot_average_rating_trend(df): 
    # Ensure date column is in datetime format for the time series plot
    df['review_date'] = pd.to_datetime(df['review_date'])
    # Aggregate data by month to smooth out the trend
    df_monthly_rating = (
        df
        .set_index('review_date')['rating']
        .resample('M')
        .mean()
        .reset_index()
    )
    fig3 = px.line(
        df_monthly_rating, 
        x='review_date', 
        y='rating', 
        title='Average Rating Trend Over Time',
        markers=True,
        labels={'review_date': 'Date', 'rating': 'Average Rating'},
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )
    fig3.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    return fig3


# Set page config
st.set_page_config(page_title="Rent The Runway Dashboard", layout="wide")
# Set a consistent template and color palette
plotly_template = "plotly_white"
color_sequence = px.colors.qualitative.Set2
font_family = "Arial"
fig_width = 1000
fig_height = 500

# Load the dataset from JSON file (JSON Lines format)
@st.cache_data
def load_renttherunway():
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz"
    return pd.read_json(url, compression="gzip", lines=True)

rent_the_runway_final_data = load_renttherunway()
#rent_the_runway_final_data = pd.read_json('renttherunway_final_data.json', lines=True)

rent_the_runway_final_data.columns = (
    rent_the_runway_final_data.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

def parse_weight(w):
    if isinstance(w, str):
        return int(w.replace('lbs', ''))
    return w

def parse_height(h):
    if isinstance(h, str) and "'" in h:
        feet, inches = h.split("' ")
        inches = inches.replace('"', '')
        return (int(feet) * 12) + int(inches)
    return h

def clean_datset(df=rent_the_runway_final_data):

    # Data cleaning and preparation for rent_the_runway_final_data

    # 1. Remove duplicates
    df = df.drop_duplicates()


    # 2. Drop rows with missing essential values
    essential_cols = ['item_id', 'review_text', 'rating']
    df = df.dropna(subset=essential_cols)

    # Apply the cleaning
    df['weight_num'] = df['weight'].apply(parse_weight)
    df['height_num'] = df['height'].apply(parse_height)
    df['weight_num'] = pd.to_numeric(df['weight'].apply(parse_weight), errors='coerce')
    df['height_num'] = pd.to_numeric(df['height'].apply(parse_height), errors='coerce')

    # 3. Fill missing values in less critical columns
    df['category'] = df['category'].fillna('unknown')
    df['bust_size'] = df['bust_size'].fillna('unknown')
    df['body_type'] = df['body_type'].fillna('unknown')
    df['height'] = df['height'].fillna('unknown')
    df['review_summary'] = df['review_summary'].fillna('unknown')
    df['rented_for'] = df['rented_for'].fillna('unknown')

    # Apply the cleaning
    df['weight_num'] = df['weight'].apply(parse_weight)
    df['height_num'] = df['height'].apply(parse_height)
    # For numerical columns, fill missing values with median
    for col in ['size', 'age']:
        df[col] = df[col].fillna(df[col].median())

    # 4. Strip whitespace from string columns
    string_cols = ['review_text', 'category', 'bust_size', 'body_type', 'height', 'weight', 'review_summary', 'rented_for']
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    df['rating'] = df['rating'] / 2

    # 5. Reset index after cleaning
    df = df.reset_index(drop=True)
    return df
df = clean_datset()

if df is not None:
    # --- Sidebar ---
    st.sidebar.title("Navigation")
    lst_page = [
        "Rating Distribution",
        "Ratings by Occasion",
        "Average Rating Trend",
        "Top Categories",
        "Category-Occasion Treemap",
        "Age by Body Type",
        "Height vs. Weight",
        "Body Type Distribution",
        "Fit Feedback by Body Type",
        "Size Range by Category",
        "Sentiment Analysis",
        "Sentiment Rating Treemap"
    ]
    page = st.sidebar.radio("Go to", lst_page)

    # 1 Rating Distribution
    if page == "Rating Distribution":
        st.title("Distribution of Customer Ratings")
        fig = plot_rating_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    # 2 Ratings by Occasion
    elif page == "Ratings by Occasion":
        st.title("Distribution of Ratings by Occasion")
        fig = plot_ratings_by_occasion(df)
        st.plotly_chart(fig, use_container_width=True)

    # 3 Average Rating Trend
    elif page == "Average Rating Trend":
        st.title("Average Customer Rating Over Time")
        fig = plot_average_rating_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    