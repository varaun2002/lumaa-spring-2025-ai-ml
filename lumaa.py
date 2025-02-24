import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# 3. DATA LOADING AND PREPROCESSING
# ------------------------------

def load_and_preprocess_data(file_path):
    """Load movie data and perform initial preprocessing, keeping genre information."""
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Drop unnecessary columns except 'genres'
        columns_to_drop = ['Unnamed: 0', 'Reviews', 'Resenhas']
        existing_columns = [col for col in columns_to_drop if col in data.columns]
        if existing_columns:
            data.drop(existing_columns, axis=1, inplace=True)
        
        # Aggregate by movie name. For genres, take the first entry or you could combine if there are multiple.
        data = data.groupby('movie_name').agg({
            'Ratings': 'mean',
            'Description': 'first',
            'genres': 'first',    # keep genres
            'emotion': lambda x: ', '.join(set(x))
        }).reset_index()
        
        # Create a combined text field that merges description and genres.
        data['combined'] = data['Description'].fillna('') + " " + data['genres'].fillna('')
        
        return data
        
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        raise

# ------------------------------
# RECOMMENDATION FUNCTION (using CountVectorizer on combined text)
# ------------------------------

def recommend_movies(file_path, user_query):
    """
    Recommend top 5 movies based on similarity between user query and combined movie descriptions and genres.
    
    Parameters:
        file_path (str): Path to the CSV file.
        user_query (str): User's movie preference query.
    """
    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)
    
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    
    # Combine the movie combined texts with the query to ensure the same vocabulary is used
    combined_texts = data['combined'].tolist()
    all_texts = combined_texts + [user_query]
    
    # Fit the vectorizer and transform all texts
    vectors = vectorizer.fit_transform(all_texts)
    
    # Separate out the movie vectors and the query vector
    movie_vectors = vectors[:-1]
    query_vector = vectors[-1]
    
    # Compute cosine similarity between the user's query and all movie combined texts
    cosine_sim = cosine_similarity(query_vector, movie_vectors).flatten()
    
    # Get the indices of the top 5 movies (highest similarity scores)
    top_indices = cosine_sim.argsort()[-5:][::-1]
    
    # Display the recommendations
    print("\nTop 5 movie recommendations based on your query:")
    for idx in top_indices:
        movie = data.iloc[idx]
        print(f"Movie: {movie['movie_name']}")
        print(f"Description: {movie['Description']}")
        
# ------------------------------
# MAIN EXECUTION
# ------------------------------

if __name__ == "__main__":
    try:
        # Configuration
        DATA_FILE = 'Movies_Reviews_modified_version1.csv'
        
        # Get user input once
        print("\nMovie Recommendation System")
        user_query = input("\nEnter your movie preference (plot, genre, emotion): ")
        
        if user_query.lower() == 'quit':
            print("Thank you for using the Movie Recommender!")
        else:
            # Run the recommendation process
            recommend_movies(DATA_FILE, user_query)
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
