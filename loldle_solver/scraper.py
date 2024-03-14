import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://loldle.net/classic")
input("Type all champions in the box and press ENTER")

rows = []
champs = driver.find_elements(By.CLASS_NAME, "classic-answer")
for champ in champs:
    attrs = champ.find_elements(By.CLASS_NAME, "square-content")
    row = []
    for attr in attrs:
        try:
            name = attr.find_element(By.CLASS_NAME, "champion-icon-name")
            row.append(name.get_attribute("innerHTML"))
        except:
            row.append(attr.text)
    rows.append(row)
pd.DataFrame(
    rows,
    columns=[
        "Name",
        "Gender",
        "Position",
        "Species",
        "Resource",
        "Range type",
        "Region",
        "Release",
    ],
).to_csv("loldle.csv")
