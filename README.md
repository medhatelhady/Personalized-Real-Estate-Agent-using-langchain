# HomeMatch - Personalized Real Estate Agent

An AI-powered real estate recommendation system built with LangChain that matches users with their ideal homes based on personalized preferences.

## Overview

HomeMatch is an intelligent sales assistant that uses conversational AI to understand customer preferences and recommend suitable properties from a database of available homes. The system leverages OpenAI's GPT models, vector embeddings, and retrieval-augmented generation (RAG) to provide personalized, context-aware property recommendations.

## Features

- **Interactive Questionnaire**: Collects user preferences through 5 key questions about home requirements
- **Semantic Search**: Uses vector embeddings to find homes matching user criteria
- **Conversational Memory**: Maintains context throughout the conversation using conversation summary memory
- **Personalized Recommendations**: Generates attractive, tailored property descriptions based on user preferences
- **CSV Data Management**: Stores and queries property data from a structured CSV file

## How It Works

1. **Data Loading**: Loads property listings from `home.csv` containing neighborhood, location, bedrooms, bathrooms, size, and price information
2. **Vector Store Creation**: Converts property data into embeddings using OpenAI and stores them in a Chroma vector database
3. **User Interaction**: Asks users 5 personalized questions about their home preferences:
   - Desired house size
   - Top 3 priorities in property selection
   - Preferred amenities
   - Transportation requirements
   - Neighborhood urbanization level
4. **Conversation Summarization**: Uses ConversationSummaryMemory to extract key preferences from user responses
5. **Retrieval & Recommendation**: Queries the vector database and generates personalized property recommendations using a custom prompt template

## Tech Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI GPT-3.5-turbo**: Language model for conversation and recommendations
- **Chroma**: Vector database for semantic search
- **OpenAI Embeddings**: Text embedding model for property data
- **Python**: Core programming language

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key in the code:
```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Usage

### Running the Python Script
```bash
python HomeMatch.py
```

### Using the Jupyter Notebook
Open `HomeMatch.ipynb` in Jupyter and run the cells interactively to:
- Generate sample property data
- Test the recommendation system
- Experiment with different user preferences

## Data Structure

The `home.csv` file contains property listings with the following attributes:
- Neighborhood
- Location (City)
- Bedrooms
- Bathrooms
- House Size (sqft)
- Price (k$)

Sample properties include listings from major US cities like San Francisco, New York, Los Angeles, Boston, Chicago, Seattle, Miami, and Washington D.C.

## Requirements

See `requirements.txt` for full dependencies. Key packages include:
- langchain==0.0.305
- openai==0.28.1
- chromadb==0.4.18
- sentence-transformers>=2.2.0
- transformers>=4.31.0

## Project Structure

```
.
├── HomeMatch.py          # Main application script
├── HomeMatch.ipynb       # Jupyter notebook for interactive development
├── home.csv              # Property listings database
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

