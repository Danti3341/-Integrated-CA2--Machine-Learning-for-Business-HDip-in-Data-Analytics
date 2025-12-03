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

# Visualization 4: Top 5 Most Popular Categories (Horizontal Bar Chart)
def plot_top_categories_bar(df): 
    top_categories = (
        df['category']
        .value_counts()
        .head(5)
        .reset_index()
    )
    top_categories.columns = ['category', 'count'] # Rename columns for cleaner plotting
    fig4 = px.bar(
        top_categories,
        x='count',
        y='category',
        orientation='h',
        title='Top 5 Most Popular Categories',
        labels={'count': 'Number of Rentals', 'category': 'Category'},
        color='count',
        color_continuous_scale=px.colors.sequential.Reds,
        text='count',
        template=plotly_template
    )
    fig4.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        yaxis=dict(autorange="reversed", showgrid=True, gridcolor='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig4

# Visualization 5: Inventory Hierarchy: Categories within Occasions (Treemap)
def plot_category_occasion_treemap(df): 
    # 1. Prepare the data: Aggregate counts by 'rented_for' and 'category'
    treemap_data = (
        df
        .groupby(['rented_for', 'category'])
        .size()
        .reset_index(name='count')
    )

    # 2. Create the treemap
    fig5 = px.treemap(
        treemap_data,
        path=['rented_for', 'category'],
        values='count',
        title='Inventory Hierarchy: Categories within Occasions',
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )

    fig5.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16)
    )

    return fig5

# Visualization 6: Age Distribution by Body Type (Violin Plot)
def plot_age_by_body_type_violin(df): 
    # Drop rows where height or weight couldn't be parsed (for cleaner plots)
    df_clean_body = df.dropna(subset=['weight_num', 'height_num', 'age'])
    fig6 = px.violin(
        df_clean_body,
        x='body_type',
        y='age',
        box=True,
        title='Age Distribution by Body Type',
        labels={'body_type': 'Body Type', 'age': 'Age'},
        color='body_type',
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )
    fig6.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig6

# Visualization 7: Height vs. Weight Correlation (Scatter Plot)
def plot_height_weight_scatter(df): 
    # Drop rows where height or weight couldn't be parsed (for cleaner plots)
    df_clean_body = df.dropna(subset=['weight_num', 'height_num'])

    fig7 = px.scatter(
        df_clean_body,
        x='weight_num',
        y='height_num',
        color='body_type',
        title='Height vs. Weight Correlation',
        labels={'weight_num': 'Weight (lbs)', 'height_num': 'Height (inches)', 'body_type': 'Body Type'},
        opacity=0.6,
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )
    fig7.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(range=[80, 300], showgrid=True, gridcolor='lightgray'),
        yaxis=dict(range=[50, 80], showgrid=True, gridcolor='lightgray')
    )

    return fig7

# Visualization 8: Distribution of User Body Types (Treemap)
def plot_body_type_distribution_treemap(df):
    # Drop rows where height or weight couldn't be parsed (for cleaner plots)
    df_clean_body = df.dropna(subset=['weight_num', 'height_num', 'age'])
    
    # Count the body types
    body_type_counts = df_clean_body['body_type'].value_counts().reset_index()
    body_type_counts.columns = ['body_type', 'count']

    # Create the treemap
    fig8 = px.treemap(
        body_type_counts,
        path=['body_type'],
        values='count',
        title='Distribution of User Body Types',
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )

    fig8.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16)
    )

    return fig8

# Visualization 9: Fit Feedback Breakdown by Body Type (Stacked Bar Chart)
def plot_fit_feedback_by_body_type(df): 
    # 1. Prepare the data: Group by 'body_type' and 'fit' to get counts
    fit_feedback = (
        df
        .groupby(['body_type', 'fit'])
        .size()
        .reset_index(name='count')
    )

    # 2. Create the stacked bar chart
    fig9 = px.bar(
        fit_feedback,
        x='body_type',
        y='count',
        color='fit',
        title='Fit Feedback Breakdown by Body Type',
        labels={'body_type': 'Body Type', 'count': 'Number of Reviews', 'fit': 'Fit Feedback'},
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )

    fig9.update_layout(
        barmode='stack',
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig9

# Visualization 10: Size Range Distribution by Top 20 Categories (Box Plot)
def plot_size_range_by_category(df):
    # Get top 10 categories
    top_10_categories = df['category'].value_counts().head(10).index
    df_top_categories = df[df['category'].isin(top_10_categories)]
    fig10 = px.box(
        df_top_categories,
        x='category',
        y='size',
        title='Size Range Distribution by Top 10 Categories',
        labels={'category': 'Category', 'size': 'Size'},
        color='category',
        color_discrete_sequence=color_sequence,
        template=plotly_template
    )
    fig10.update_layout(
        width=fig_width,
        height=fig_height,
        font=dict(family=font_family, size=16),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig10

# Visualization 11: Sentiment Analysis of Review Texts (Histogram)
def plot_sentiment_analysis_histogram(df):
    # Take a sample of text reviews
    sample_reviews = df.sample(n=10000, random_state=42)

    # Compute sentiment polarity for each review in the sample
    sample_reviews['sentiment'] = sample_reviews['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Interactive histogram of sentiment polarity
    fig_sentiment = px.histogram(
        sample_reviews,
        x='sentiment',
        nbins=50,
        title='Sentiment Analysis of Review Texts',
        labels={'sentiment': 'Sentiment Polarity'},
        template=plotly_template,
        width=fig_width,
        height=fig_height,
        color=sample_reviews['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative'),
        color_discrete_map={'Positive': 'green', 'Negative': 'red'}
    )
    fig_sentiment.update_layout(
        font=dict(family=font_family, size=16),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    return fig_sentiment

# Visualization 12: Treemap of Positive and Negative Feelings by Rating (Treemap)
def plot_sentiment_rating_treemap(df):
    # Take a sample of text reviews
    sample_reviews = df.sample(n=10000, random_state=42)

    # Compute sentiment polarity for each review in the sample
    sample_reviews['sentiment'] = sample_reviews['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Classify reviews as positive, negative, or neutral
    def sentiment_label(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    sample_reviews['sentiment_label'] = sample_reviews['sentiment'].apply(sentiment_label)

    # Aggregate counts by sentiment label and occasion
    sentiment_treemap_data = (
        sample_reviews
        .groupby(['sentiment_label', 'rating'])
        .size()
        .reset_index(name='count')
    )

    # Plot treemap
    fig_sentiment_treemap = px.treemap(
        sentiment_treemap_data,
        path=['sentiment_label', 'rating'],
        values='count',
        title='Treemap of Positive and Negative Feelings by Rating',
        color='sentiment_label',
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
        template=plotly_template,
        width=fig_width,
        height=fig_height
    )
    fig_sentiment_treemap.update_layout(font=dict(family=font_family, size=16))

    return fig_sentiment_treemap

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

# Load the dataset from JSON file (JSON Lines format)
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

    # 4 Top Categories
    elif page == "Top Categories":
        st.title("Top 5 Most Popular Categories")
        fig = plot_top_categories_bar(df)
        st.plotly_chart(fig, use_container_width=True)

    # 5 Category-Occasion Treemap
    elif page == "Category-Occasion Treemap":
        st.title("Inventory Hierarchy: Categories within Occasions")
        fig = plot_category_occasion_treemap(df)
        st.plotly_chart(fig, use_container_width=True)

    # 6 Age by Body Type
    elif page == "Age by Body Type":
        st.title("Age Distribution by Body Type")
        fig = plot_age_by_body_type_violin(df)
        st.plotly_chart(fig, use_container_width=True)

    # 7 Height vs. Weight
    elif page == "Height vs. Weight":
        st.title("Height vs. Weight Correlation")
        fig = plot_height_weight_scatter(df)
        st.plotly_chart(fig, use_container_width=True)

    # 8 Body Type Distribution
    elif page == "Body Type Distribution":
        st.title("Distribution of User Body Types")
        fig = plot_body_type_distribution_treemap(df)
        st.plotly_chart(fig, use_container_width=True)

    # 9 Fit Feedback by Body Type
    elif page == "Fit Feedback by Body Type":
        st.title("Fit Feedback Breakdown by Body Type")
        fig = plot_fit_feedback_by_body_type(df)
        st.plotly_chart(fig, use_container_width=True)

    # 10 Size Range by Category
    elif page == "Size Range by Category":
        st.title("Size Range Distribution by Top 10 Categories")
        fig = plot_size_range_by_category(df)
        st.plotly_chart(fig, use_container_width=True)
    # 11 Sentiment Analysis Histogram
    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis of Review Texts")
        fig = plot_sentiment_analysis_histogram(df)
        st.plotly_chart(fig, use_container_width=True)
    # 12 Sentiment Rating Treemap
    elif page == "Sentiment Rating Treemap":
        st.title("Treemap of Positive and Negative Feelings by Rating")
        fig = plot_sentiment_rating_treemap(df)
        st.plotly_chart(fig, use_container_width=True)