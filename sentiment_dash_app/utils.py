from io import BytesIO
from wordcloud import WordCloud
import base64
import ast
import numpy as np
import pandas as pd
import plotly.express as px

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

