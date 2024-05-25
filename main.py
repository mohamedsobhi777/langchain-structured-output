# from langchain_community.chat_models import ChatOpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# The temperature impacts the randomness of the output,
# which in this case we don't want any randomness so we define it as 0.0
temperature = 0.0
model = "gpt-4"

llm = ChatOpenAI(model=model, temperature=temperature, api_key=os.environ['OPENAI_API_KEY_1'])


from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser

class SearchSchema(BaseModel):
    includedIngredients: list[str] = Field(description="the list of ingredients that should be included in the recipe")
    excludedIngredients: list[str] = Field(description="the list of ingredients that should be excluded from the recipe")
    allergies: list[Literal["peanut-free","soy-free","dairy-free","nut-free"]] = Field(description="the list of allergies specified in the request")
    diets: list[Literal["vegetarian","vegan","keto-friendly","pescatarian","mediterranean"]] = Field(description="the list of diets specified in the request")
    cuisines: list[str] = Field(description="the list of cuisines specified in the request")


pydantic_parser = PydanticOutputParser(pydantic_object=SearchSchema)
format_instructions = pydantic_parser.get_format_instructions()

# The Pydantic model creates the formatting instructions to be included in the prompt
# Here is the what those instructions look like
print(format_instructions)

RECIPE_SEARCH_PROMPT = """
Your goal is to understand and parse out the user's recipe search request based on their preferences.

{format_instructions}

Recipe Search Request:
{request}
"""

prompt = ChatPromptTemplate.from_template(
    template=RECIPE_SEARCH_PROMPT,
    partial_variables = {
        "format_instructions": format_instructions # passing in the formatting instructions created earlier in place of "format_instructions" placeholder
    }
)

full_chain = {"request": lambda x: x["request"]} | prompt | llm


request = "I am vegetarian and have a peanut allergy. Find me Mexican recipes that have tomatoes but no corn."
result = full_chain.invoke({"request": request})

print(result.content)
