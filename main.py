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

# Pydantic models for health suggestions
class HealthSuggestion(BaseModel):
    category: str = Field(description="Category of the suggestion (e.g., Mental Health, Physical Health, Lifestyle)")
    title: str = Field(description="Title of the suggestion")
    description: str = Field(description="Detailed description of the suggestion")
    priority: str = Field(description="Priority level: high, medium, low")
    timeframe: str = Field(description="When to implement this suggestion")
    difficulty: str = Field(description="Implementation difficulty: easy, moderate, challenging")
    benefits: List[str] = Field(description="Expected benefits from following this suggestion")

class MentalHealthAnalysis(BaseModel):
    stress_level_assessment: str = Field(description="Overall stress level assessment")
    key_stress_areas: List[str] = Field(description="Main areas causing stress based on predictedAspectCounts")
    mental_state_summary: str = Field(description="Summary of current mental state")
    risk_factors: List[str] = Field(description="Potential risk factors to monitor")
    positive_indicators: List[str] = Field(description="Positive aspects in the user's profile")

class HealthSuggestionsResponse(BaseModel):
    mental_health_analysis: MentalHealthAnalysis = Field(description="Analysis of user's mental health condition")
    immediate_suggestions: List[HealthSuggestion] = Field(description="Immediate actions to take (next 1-7 days)")
    short_term_suggestions: List[HealthSuggestion] = Field(description="Short-term goals (1-4 weeks)")
    long_term_suggestions: List[HealthSuggestion] = Field(description="Long-term lifestyle changes (1+ months)")
    emergency_resources: List[str] = Field(description="Emergency mental health resources if needed")


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

# Health suggestions prompt
# Health suggestions prompt
health_suggestions_prompt = ChatPromptTemplate.from_template(
    """
You are an expert mental health and wellness advisor. Your task is to analyze the user's stress profile and provide comprehensive, personalized health suggestions for leading a good and easy life.

**User Profile:**
{user_input}

**Analysis Guidelines:**
1. **Stress Assessment**: Analyze the ratio of stress entries to total entries and identify patterns
2. **Stress Sources**: Examine predictedAspectCounts to understand what's causing stress
3. **Mental State**: Determine current mental health condition based on the data
4. **Personalization**: Consider pastLikings if available to tailor suggestions

**Required Output Schema:**

For mental_health_analysis, provide:
- stress_level_assessment: Overall stress level (e.g., "High", "Moderate", "Low")
- key_stress_areas: List of main stress sources from predictedAspectCounts
- mental_state_summary: Brief summary of current mental state
- risk_factors: List of potential risks to monitor
- positive_indicators: List of positive aspects in the user's profile

For each suggestion (immediate_suggestions, short_term_suggestions, long_term_suggestions), EVERY suggestion MUST include ALL these fields:
- category: One of "Mental Health", "Physical Health", "Lifestyle", "Coping Strategies", "Preventive Care"
- title: Clear, concise title for the suggestion (REQUIRED)
- description: Detailed explanation of what to do (REQUIRED)
- priority: "high", "medium", or "low" (REQUIRED)
- timeframe: When to implement (e.g., "Daily", "Weekly", "1-2 weeks") (REQUIRED)
- difficulty: "easy", "moderate", or "challenging" (REQUIRED)
- benefits: Array of expected benefits (REQUIRED)

**Categories for Suggestions:**
- Mental Health: Stress management, mindfulness, therapy, emotional regulation
- Physical Health: Exercise, nutrition, sleep, medical checkups
- Lifestyle: Work-life balance, social connections, hobbies, environment
- Coping Strategies: Immediate stress relief, emergency techniques
- Preventive Care: Long-term habits, routine building, self-care

**Priority Levels:**
- High: Urgent issues that need immediate attention
- Medium: Important improvements for better wellbeing
- Low: Nice-to-have enhancements for optimal health

**Provide exactly:**
1. Comprehensive mental health analysis
2. 3-5 immediate actionable suggestions (1-7 days)
3. 3-5 short-term goals (1-4 weeks)
4. 3-5 long-term lifestyle improvements (1+ months)
5. Emergency resources list if stress levels are concerning

**Important**: 
- Be empathetic, practical, and focus on achievable steps
- Avoid medical diagnosis but provide supportive guidance
- ENSURE ALL REQUIRED FIELDS ARE INCLUDED for every suggestion
- Each suggestion must be complete with all 7 fields: category, title, description, priority, timeframe, difficulty, benefits

**Output the result as a JSON object following the exact required schema. Do not add any other text.**
"""
)

# This chain will take the user input, format it into the prompt, send it to the LLM,
# and parse the output into our `InitialRecommendations` Pydantic model.
generator_chain = generator_prompt | llm.with_structured_output(InitialRecommendations)

# Health suggestions chain
health_suggestions_chain = health_suggestions_prompt | llm.with_structured_output(HealthSuggestionsResponse)


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
    

@app.post("/health-suggestions")
async def get_health_suggestions(user_input: UserInput):
    """
    Analyzes user's mental condition and provides personalized health suggestions for a better life.
    """
    input_str = user_input.model_dump_json(exclude_none=True)

    try:
        # Run the health suggestions chain
        print("--- Analyzing mental condition and generating health suggestions ---")
        health_analysis = await health_suggestions_chain.ainvoke({"user_input": input_str})
        
        # Calculate stress percentage for additional context
        stress_percentage = (user_input.totalStressEntries / user_input.totalEntries * 100) if user_input.totalEntries > 0 else 0
        
        # Validate that we have suggestions
        if not health_analysis.immediate_suggestions:
            health_analysis.immediate_suggestions = [
                HealthSuggestion(
                    category="Mental Health",
                    title="Take Deep Breaths",
                    description="Practice 5-minute deep breathing exercises when feeling stressed",
                    priority="high",
                    timeframe="Immediately when stressed",
                    difficulty="easy",
                    benefits=["Reduces immediate stress", "Calms nervous system"]
                )
            ]
        
        return {
            "user_stress_percentage": round(stress_percentage, 2),
            "analysis_summary": {
                "total_entries_analyzed": user_input.totalEntries,
                "stress_entries": user_input.totalStressEntries,
                "non_stress_entries": user_input.totalNoStressEntries,
                "main_stress_sources": list(user_input.predictedAspectCounts.keys())
            },
            "mental_health_analysis": health_analysis.mental_health_analysis.model_dump(),
            "suggestions": {
                "immediate_actions": [suggestion.model_dump() for suggestion in health_analysis.immediate_suggestions],
                "short_term_goals": [suggestion.model_dump() for suggestion in health_analysis.short_term_suggestions],
                "long_term_changes": [suggestion.model_dump() for suggestion in health_analysis.long_term_suggestions]
            },
            "emergency_resources": health_analysis.emergency_resources or [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Emergency: 911"
            ],
            "disclaimer": "These suggestions are for informational purposes only and do not constitute medical advice. Please consult with healthcare professionals for persistent mental health concerns."
        }

    except Exception as e:
        print(f"An error occurred in health suggestions: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while generating health suggestions: {str(e)}"
        )