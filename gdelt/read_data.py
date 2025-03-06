import os
import pandas as pd
import nest_asyncio
import asyncio
from crawl4ai import AsyncWebCrawler

data_dir = (
    "/home/harsh242/pc_data/academics/academics_bu_grs/CS598-Multi-modal-ML"
    "/Project/Prithvi-WxC/gdelt_data"
)


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

df = pd.read_csv(
    os.path.join(data_dir, files[0]), 
    sep='\t',
    names=gdelt_columns_2013_present,
    index_col=False
)
print(df.head())

url_list = df.SOURCEURL.to_list()

print(url_list[10])

# For Jupyter notebooks, use this instead of asyncio.run()
# nest_asyncio.apply()

# Now define your crawler function
async def crawler_func():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url_list[10]
        )
    print(result.markdown)
    return result

# Run the async function
# await crawler_func()

if __name__ == "__main__":
    asyncio.run(crawler_func())

