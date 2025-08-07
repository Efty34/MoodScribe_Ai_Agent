import os
import json
import asyncio
import concurrent.futures
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from tools import fetch_movie_details, fetch_book_details, fetch_song_details

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.8, # A little more creativity for the initial list
)

# --- 2. Define Pydantic Models for Input, LLM Output, and Final Output ---

# Pydantic model for a single Exercise object (for structured output)
class Exercise(BaseModel):
    name: str = Field(description="The name of the exercise.")
    type: str = Field(description="The category of the exercise (e.g., Yoga, HIIT, Cardio).")
    duration: str = Field(description="Recommended duration in minutes.")
    intensity: str = Field(description="Intensity level (e.g., low, medium, high).")
    description: str = Field(description="A brief description of the exercise.")
    benefits: List[str] = Field(description="A list of key benefits.")
    equipment: List[str] = Field(description="List of required equipment. Can be empty.")
    instructions: List[str] = Field(description="Step-by-step instructions.")
    caloriesBurnedPerHour: str = Field(description="Estimated calories burned per hour.")
    suitable: List[str] = Field(description="Who this exercise is suitable for (e.g., beginner, intermediate).")
    location: str = Field(description="Recommended location (e.g., home, gym).")

# Pydantic model for the structured output from our first LLM call
class InitialRecommendations(BaseModel):
    recommended_movie_titles: List[str] = Field(description="List of 10 movie titles to recommend.")
    recommended_book_titles: List[str] = Field(description="List of 10 book titles to recommend.")
    recommended_song_titles_and_artists: List[str] = Field(description="List of 10 songs, each formatted as 'Title by Artist'.")
    generated_exercises: List[Exercise] = Field(description="A list of 10 fully detailed exercise objects.")

# Pydantic models for the API endpoint input (no changes here)
class PastLikings(BaseModel):
    favoriteMovies: Optional[List[str]] = Field(default_factory=list)
    favoriteBooks: Optional[List[str]] = Field(default_factory=list)
    favoriteSongs: Optional[List[str]] = Field(default_factory=list)
    favoriteExercises: Optional[List[str]] = Field(default_factory=list)

class UserInput(BaseModel):
    totalEntries: int
    totalStressEntries: int
    totalNoStressEntries: int
    predictedAspectCounts: Dict[str, int]
    pastLikings: Optional[PastLikings] = None


# --- 3. Create the Generator Chain ---
# This prompt tells the LLM to do ONLY ONE THING: generate the initial lists.
generator_prompt = ChatPromptTemplate.from_template(
    """
You are an expert content recommendation engine. Your task is to analyze the user's stress and no stress profile and past preferences to generate a list of recommendations.

**User Profile:**
{user_input}

**Analysis Instructions:**
1.  Analyze the user's stress levels (`totalStressEntries` vs `totalNoStressEntries`) and the specific stress aspects (`predictedAspectCounts`).
2.  If `pastLikings` are provided, use them to understand the user's taste in movies, books, songs, and exercises.
3.  Based on your complete analysis, generate a list of exactly 10 recommendations for each category.
4.  For songs, the format MUST be "Song Title by Artist".
5.  For exercises, you must generate the complete, detailed object for each recommendation.

**Your SOLE task is to output a JSON object that conforms to the required schema. Do not add any other text.**
"""
)

# This chain will take the user input, format it into the prompt, send it to the LLM,
# and parse the output into our `InitialRecommendations` Pydantic model.
generator_chain = generator_prompt | llm.with_structured_output(InitialRecommendations)


# --- 4. Define FastAPI Server and Endpoint ---
app = FastAPI(
    title="Wellness Recommender AI Chain",
    description="An API that provides personalized content recommendations using a deterministic chain.",
)

@app.on_event("startup")
async def startup_event():
    """Initialize any necessary resources on startup."""
    print("Starting up Wellness Recommender AI Chain...")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "message": "Wellness Recommender AI Chain is running"}

@app.post("/get-recommendations")
async def get_recommendations(user_input: UserInput):
    """
    Receives user stress data and returns personalized content recommendations.
    """
    input_str = user_input.model_dump_json(exclude_none=True)

    try:
        # STEP 1: Run the generator chain to get the initial list of titles and exercises
        print("--- Calling Generator Chain to get titles and exercises ---")
        initial_recs = await generator_chain.ainvoke({"user_input": input_str})

        # STEP 2: Asynchronously fetch details for movies, books, and songs in parallel
        print("--- Fetching details from external APIs in parallel ---")

        # Use ThreadPoolExecutor for better control over thread execution
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks to the thread pool
            movie_futures = [loop.run_in_executor(executor, fetch_movie_details, title) 
                           for title in initial_recs.recommended_movie_titles]
            book_futures = [loop.run_in_executor(executor, fetch_book_details, title) 
                          for title in initial_recs.recommended_book_titles]
            song_futures = [loop.run_in_executor(executor, fetch_song_details, title) 
                          for title in initial_recs.recommended_song_titles_and_artists]
            
            # Wait for all tasks to complete
            movie_results = await asyncio.gather(*movie_futures, return_exceptions=True)
            book_results = await asyncio.gather(*book_futures, return_exceptions=True)
            song_results = await asyncio.gather(*song_futures, return_exceptions=True)

        # STEP 3: Aggregate the results and format the final response
        print("--- Aggregating final response ---")

        # Filter out any results that had errors during fetching or exceptions
        def is_valid_result(result):
            return (
                result is not None and 
                not isinstance(result, Exception) and 
                not (isinstance(result, dict) and result.get("error"))
            )

        final_response = {
            "movies": [res for res in movie_results if is_valid_result(res)],
            "books": [res for res in book_results if is_valid_result(res)],
            "songs": [res for res in song_results if is_valid_result(res)],
            "exercises": [ex.model_dump() for ex in initial_recs.generated_exercises]
        }
        
        return final_response

    except Exception as e:
        print(f"An error occurred in the recommendation chain: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while processing your request: {str(e)}"
        )