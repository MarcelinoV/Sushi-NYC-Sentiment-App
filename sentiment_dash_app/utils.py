from io import BytesIO
from wordcloud import WordCloud
import base64
import ast
import numpy as np
import pandas as pd
import re
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# wordcloud function
def wordcloud_viz(df, gen_whole_wordcloud=False):

	loaded_tokens = [ast.literal_eval(i) for i in df['WORD_TOKENS']]

	if gen_whole_wordcloud:
		# Flatten the list of lists into a single list of words
		flattened_tokens = [word for word_tokens in loaded_tokens for word in word_tokens]
	else:
		# Use the first list of tokens (or any specific logic you need)
		flattened_tokens = loaded_tokens[0]  # Assuming you want the first set of tokens

	# Combine the tokens into a single string
	combined_text = ' '.join(flattened_tokens)

	# Generate the Wordcloud
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

	 # Convert the Wordcloud to a PIL image
	wordcloud_image = wordcloud.to_image()
    
    # Convert the PIL image to a base64-encoded string
	buffered = BytesIO()
	wordcloud_image.save(buffered, format="PNG")
	wordcloud_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

	return wordcloud_base64

# support function
def update_chart_glo(group_vals, flattened_metrics, avg_sent_scores):
    selected_columns = group_vals

    model_score_map = pd.DataFrame(flattened_metrics)
    
    # Reshape DataFrame to long format for Plotly Express
    df_melted = avg_sent_scores.melt(value_vars=selected_columns, var_name="Sentiment Metric", value_name="Score") \
        .merge(model_score_map, on='Sentiment Metric')

    # Create bar chart
    fig = px.bar(df_melted, x="Sentiment Metric", y="Score", 
                 text_auto=True, title=f"Average Citywide Sentiment Scores", color="Model", 
                 #height=second_row_visuals_height, width=second_row_visuals_width
                 )

    # Improve visualization
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig.update_layout(yaxis=dict(title="Average Score"), xaxis=dict(title="Metric"), showlegend=True)

    return fig

# weighted confidence score
def weighted_confidence_score(df, rating_col='OVERALL_RATING', count_col='USERRATINGCOUNT', threshold=100):
    """
    Calculate the Weighted Confidence Score for each location.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing rating and review count columns.
        rating_col (str): Column name for overall rating.
        count_col (str): Column name for number of user reviews.
        threshold (int): Number of reviews at which confidence plateaus.
        
    Returns:
        pd.Series: A series containing the weighted confidence scores.
    """
    confidence_factor = 1 - np.exp(-df[count_col] / threshold)
    weighted_score = round(df[rating_col] * confidence_factor, 2)
    return weighted_score

# Preprocessing: Remove special characters & lowercase
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation, lowercase
    return text

# generate LDA
def lda_builder(df:pd.DataFrame, num_of_topics:int, ngram_range:tuple, review_type:str):
    
    if review_type == 'Positive':
        reviews = df[(df['RATING'] > 3.2) | (df['BASE_POS'] >= .5)]['TEXT'].tolist()

    elif review_type == 'Negative':
        reviews = df[(df['RATING'] < 2.8) | (df['BASE_NEG'] >= .5)]['TEXT'].tolist()

    else:
        reviews = df[(df['RATING'] >=2.8) & (df['RATING'] <= 3.2) | (df['BASE_NEU'] >= .6)]['TEXT'].tolist()

    # declare ngrame_range

    min_ngram, max_ngram = ngram_range

    reviews_cleaned = [preprocess(review) for review in reviews]

    # Convert cleaned text to a document-term matrix using CountVectorizer (Bag of Words)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(min_ngram,max_ngram))
    X = vectorizer.fit_transform(reviews_cleaned)

    # Fit LDA model using scikit-learn's LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=num_of_topics, random_state=42)  # 5 topics as an example
    lda.fit(X)
  
    return lda, vectorizer, X

# convert lda object to df

def lda_to_df(lda, vectorizer, X):
    # Get topic-term matrix and normalize
    topic_term_distrib = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]  # shape: (n_topics, n_terms)
    vocab = np.array(vectorizer.get_feature_names_out())

    # Get top N words per topic
    top_n_words = 10
    top_words = []
    for topic_weights in topic_term_distrib:
        top_indices = topic_weights.argsort()[::-1][:top_n_words]
        top_words.append(", ".join(vocab[top_indices]))

    # Get topic sizes (how common each topic is across documents)
    doc_topic_distr = lda.transform(X)  # shape: (n_docs, n_topics)
    topic_freq = doc_topic_distr.sum(axis=0)

    # Reduce topic-term vectors to 2D for plotting
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    topic_coords = tsne.fit_transform(topic_term_distrib)

    # Create DataFrame for Plotly
    topic_df = pd.DataFrame({
        "x": topic_coords[:, 0],
        "y": topic_coords[:, 1],
        "topic": [f"Topic {i+1}" for i in range(lda.n_components)],
        "freq": topic_freq,
        "top_words": top_words
    })

    return topic_df

# build lda visual

def lda_visual(topic_df, place_name='Citywide', type_='Neutral'):
    fig = go.Figure()
    
    total_freq = topic_df["freq"].sum()  # for percentage calculation

    for i, topic_num in enumerate(topic_df['topic'].unique(), 1):
        topic_data = topic_df[topic_df['topic'] == topic_num]
        topic_freq = topic_data["freq"].iloc[0]  # assuming one row per topic
        prop = topic_freq / total_freq * 100
        top_words_raw = topic_data['top_words'].iloc[0]  # e.g. "sushi fish rice tuna"
        top_words_list = top_words_raw.split()           # Convert to list: ["sushi", "fish", "rice", "tuna"]
        top_words_vertical = "<br>".join(top_words_list) # Join with <br> for vertical stacking

        hover_text = (
            f"<b>Topic {i}</b><br>"
            f"Top Words: {topic_data['top_words'].iloc[0]}<br>"
            # f"<b>Top Words:</b><br>{top_words_vertical}<br>"
            f"<b>Proportion: {prop:.2f}%</b>"
        )

        fig.add_trace(go.Scatter(
            x=topic_data["x"],
            y=topic_data["y"],
            mode="markers+text",
            marker=dict(
                size=np.sqrt(topic_freq) * 7.5,
                sizemode="area",
                sizeref=2.*max(topic_df["freq"])/(60.**2),
                sizemin=10,
                line=dict(width=1, color="DarkSlateGrey")
            ),
            name=f"Topic {i}",
            text=f"Topic {i}",
            hovertext=hover_text,
            # hovertemplate=
            #     f"<b>Topic {i}</b><br>" +
            #     "<b>Top Words:</b><br>" +
            #     "<br>".join(top_words_list[:20]) + "<br>" +
            #     f"<b>Proportion: {prop:.2f}%</b><extra></extra>",
            hoverinfo="text",
            customdata=[[topic_data['top_words'].iloc[0]]]
        ))

    fig.update_layout(
        title=f"LDA Topic Visualization (t-SNE) - {place_name} - {type_} Sushi Reviews",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_dark",
        showlegend=True,
        height=500
    )

    return fig

# produce user friendly axes for 3d sentiment scatter
def produce_axes(sent_vector_labels, special=False):

    if special:
        return {i:i for i in sent_vector_labels}

    new_axes = []

    for i in sent_vector_labels:
        if 'POS' in i:
            new_axes.append('Positive Sentiment')
        elif 'NEG' in i:
            new_axes.append('Negative Sentiment')
        else:
            new_axes.append('Neutral Sentiment')

    col_map = {i:j for i,j in zip(sent_vector_labels, new_axes)}

    return col_map

# visualize treemap for LDA topic
def create_treemap_from_topic(topic_row):
    """
    Creates a treemap plot of top words for a given topic.

    Parameters:
        topic_row (pd.Series): A row from topic_df containing 'topic' and 'top_words'.

    Returns:
        plotly.graph_objs._figure.Figure: Treemap figure.
    """
    words = topic_row["top_words"].split(", ")
    topic_name = topic_row["topic"]

    df = pd.DataFrame({
        "labels": words,
        "parents": [topic_name] * len(words),
        "values": [len(words) - i for i in range(len(words))]  # Just assign decreasing dummy values
    })

    # Add root node for hierarchy
    root = pd.DataFrame({"labels": [topic_name], "parents": [""], "values": [df["values"].sum()]})
    df = pd.concat([root, df], ignore_index=True)

    fig = px.treemap(
        df,
        path=["parents", "labels"],
        values="values",
        title=f"Treemap of Top Words for {topic_name}",
        height=500
    )

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig
