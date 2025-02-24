# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

# Content-Based Movie Recommendation System

A simple content-based recommendation system that suggests movies based on user text descriptions of their preferences. The system uses text similarity to match user queries with movie descriptions and genres.

## Dataset

The system uses a modified version of a movies dataset containing:
- Movie names
- Descriptions
- Genres
- Ratings
- Emotions

The dataset file (`Movies_Reviews_modified_version1.csv`) should be placed in the root directory of the project.

## Setup

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/varaun2002/lumaa-spring-2025-ai-ml.git
cd varaun2002/lumaa-spring-2025-ai-ml
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the System

You can run the recommendation system using Python:

```bash
python lumaa.py
```

The system will prompt you to enter your movie preferences. For example:
```
Enter your movie preference (plot, genre, emotion): I love thrilling action movies set in space, with a comedic twist
```

## Example Output

For the query "I love thrilling action movies set in space, with a comedic twist", the system returns:

```
Top 5 movie recommendations based on your query:
Movie: Guardians of the Galaxy
Description: A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe.

Movie: Thor: Ragnarok
Description: Thor must fight for survival and race against time to prevent the all-powerful Hela from destroying his home and the Asgardian civilization.

[Additional recommendations...]
```

## Implementation Details

The recommendation system uses:
- CountVectorizer for text feature extraction
- Cosine similarity for matching user queries with movie descriptions
- Combined text features from movie descriptions and genres
- Simple preprocessing to handle duplicate entries and missing values

## Code Structure

- `recommend.py`: Main script containing the recommendation system
  - `load_and_preprocess_data()`: Handles data loading and preprocessing
  - `recommend_movies()`: Core recommendation function using text similarity
  - Main execution block with user interface

## Note

This is a simple implementation focused on text similarity using CountVectorizer and cosine similarity. The system can be enhanced with more sophisticated NLP techniques or additional features, but the current implementation provides a solid foundation for content-based recommendations.

## Video Link

https://drive.google.com/file/d/17cWlDh4rztKZYBPauNwV5d_NTSW5kC7q/view?usp=sharing