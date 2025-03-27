import os
import requests
import json
import pandas as pd
import nest_asyncio
import asyncio
from crawl4ai import AsyncWebCrawler


# Now define your crawler function
async def crawler_func(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url
        )
    print(result.markdown)
    return result


# Parameters
data_dir = (
    "/home/harsh242/pc_data/academics/academics_bu_grs/CS598-Multi-modal-ML"
    "/Project/Prithvi-WxC/gdelt_data"
)
save_freq = 10

files = os.listdir(data_dir)
files.remove('GDELT.MASTERREDUCEDV2.TXT')

gdelt_columns_2013_present = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", 
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", 
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", 
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", 
    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", 
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", 
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", 
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", 
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", 
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", 
    "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", 
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

categories = [
    "Extreme Heat",
    "Severe Cold",
    "Drought Crisis",
    "Hurricane Impact",
    "Tornado Threat",
    "Cyclone Activity",
    "Thunderstorm Hazard",
    "Flooding Crisis",
    "Landslide Risk",
    "Wildfire Outbreak",
    "Snowstorm Hazard",
    "Hailstorm Damage",
    "Pandemic Spread",
    "Respiratory Outbreak",
    "Vector-Borne Threat",
    "Waterborne Disease",
    "Food Contamination",
    "Zoonotic Transmission",
    "Air Pollution Crisis",
    "Ecosystem Disruption"
]

# Define the Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define the model you want to use (e.g., mistral)
# MODEL_NAME = "deepseek-r1:32b"
# MODEL_NAME = "deepseek-r1:8b"
# MODEL_NAME = "llama3.2"
# MODEL_NAME = "llama3.2:3b-instruct-fp16"
# MODEL_NAME = "llama3.1:8b-instruct-fp16"
# MODEL_NAME = "gemma3"
MODEL_NAME = "gemma3:27b"

def ollama_api_call(prompt: str):

    # Try 8b-instruct-fp16 llama3.1 model 

    # Define your prompt
    # prompt = "What is the capital of UK? Tell me more about it"

    # Make the request to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Decode the JSON line
                json_response = json.loads(line)
                # Extract the response text
                response_text = json_response.get("response", "")
                full_response += response_text
                # Print without newline to show streaming
                # print(response_text, end="", flush=True)
    else:
        print("Error:", response.status_code, response.text)

    return full_response

def save_article(article: str, filename: str, url: str = None):
    with open(filename, 'w', encoding='utf-8') as f:
        if url:
            f.write(f"URL: {url}\n\n")
        f.write(article)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    raw_articles_dir = "raw_articles"
    os.makedirs(raw_articles_dir, exist_ok=True)

    cleaned_articles_dir = f"cleaned_articles/{MODEL_NAME}"
    os.makedirs(cleaned_articles_dir, exist_ok=True)

    df = pd.read_csv(
        os.path.join(data_dir, files[0]), 
        sep='\t',
        names=gdelt_columns_2013_present,
        index_col=False
    )
    df = df.iloc[200:220]
    df["CATEGORIES"] = None
    # print(df.head())

    url_list = df.SOURCEURL.to_list()

    # print(url_list[10])

    category_list = []
    for index, row in df.iterrows():
        url = row['SOURCEURL']
        result = asyncio.run(crawler_func(url))
        filename = f"{raw_articles_dir}/article_{index}.md"
        save_article(result.markdown, filename)

        # prompt = (
        #     f"Article markdown: {result.markdown}"
        #     "\n\n"
        #     f"Article url: {url}"
        #     "\n\n"
        #     "Given the above markdown of a crawled webpage and its url, your task is to "
        #     "extract the article out of all the noise. There will be a lot of "
        #     "urls, advertisements and link to other pages or other articles. Ignore them. "
        #     "Focus on just the article. You can use the url as a hint to identify the article. "
        #     "Just return the article without any modification. Do not include any other text. "
        #     "If you cannot find any article, return None."
        # )

        prompt = (
            f"Article markdown: {result.markdown}"
            "\n\n"
            f"Article url: {url}"
            "\n\n"
            "Given the above markdown of a crawled webpage and its url, your task is to "
            "extract the heading of the article out of all the noise. There will be a lot of "
            "urls, advertisements and link to other pages or other articles. Ignore them. "
            "Focus on just the heading. You can use the url as a hint to identify the heading. "
            "Just return the heading without any modification. Do not include any other text. "
            "If you cannot find any heading, return None."
        )

        # prompt = (
        #     "Given the markdown of a crawled webpage that will be posted below, your task is to "
        #     "extract the news article out of all the noise. There will be a lot of "
        #     "urls, advertisements and link to other pages or other articles. Ignore them. "
        #     "Focus on just the news article. You can use the url as a hint to identify the article. "
        #     "Just return the news article without any modification. Do not include any other text."
        #     "\n\n"
        #     f"Article url: {url}"
        #     "\n\n"
        #     "Here is the article: "
        #     f"Article markdown: {result.markdown}"
        # )

        article = ollama_api_call(prompt)
        
        # Save article to file
        filename = f"{cleaned_articles_dir}/article_{index}.txt"
        save_article(article, filename, url)

        prompt2 = (
            f"Article: {article}"
            "\n\n"
            "Given the above article, your task is to read through the article "
            "and categorize it into the following 20 categories. \n"
            f"Categories = {categories} \n"
            "One article can have multiple categories. "
            "Your output should be only the list of categories separated by commas. "
            "Example: Extreme Heat, Drought Crisis, Pandemic Spread \n"
            "Only refer to the categories in the list above. Do not include any other categories. "
            "Do not include any other text. Just the list of categories. \n"
            "If you cannot find any category, return None."
        )

        categories = ollama_api_call(prompt2)
        categories = categories.strip()
        df.loc[index, "CATEGORIES"] = categories
        category_list.append(categories)
        # print(categories)
        if (index) % save_freq == 0:
            # checkpoint_file = f"gdelt_data_with_categories_checkpoint_{index}.csv"
            filename = "gdelt_data_with_categories.csv"
            df.to_csv(filename, index=False)

        print("--------------------------------END--------------------------------\n\n")

    # print(category_list)
    for index, cat in enumerate(category_list):
        print(f"{index}: {cat}")
        print("\n")

    df.to_csv("gdelt_data_with_categories.csv", index=False)

        # Convert results to DataFrame
        # results_df = pd.DataFrame(results)

