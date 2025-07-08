from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import re
import venosch

s = Service('chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--blink-settings=imagesEnabled=false")
options.add_argument("--ignore-certificate-errors")
driver = webdriver.Chrome(service=s, options=options)
driver.get("https://www.sofascore.com/football/match/colorado-rapids-2-houston-dynamo-2/JUmdsuCod#id:13430529")

# squad_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "xYoqG")))

# teams = [el for el in squad_container.find_elements(By.CLASS_NAME, "hiWfit") if el.text.strip()]

# home = teams[0].text
# away = teams[2].text

# minute_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "diKTsJ")))
# inner_minute_element = minute_container.find_element(By.CSS_SELECTOR, ".fPSBzf.bYPznK")
# print(f"Home {home} - {away} Away at {inner_minute_element.text}")

details_cont = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".fPSBzf.gasSYy")))
details_divs = driver.find_elements(By.CSS_SELECTOR, ".fRBCCw.dkmVnc")
row_div = details_divs[2].find_element(By.CSS_SELECTOR, ".xYoiw") 

key_elements = row_div.find_elements(By.CSS_SELECTOR, ".ioWvhD")

events = {
    "substitutions": [],
    "yellow_cards": [],
    "red_cards": [],
    "goals": []
}

minute_pattern = re.compile(r"^\d+'\s*(\+\d+)?$")
score_pattern = re.compile(r"^\d+\s*-\s*\d+$")

def is_valid_name(text):
    if minute_pattern.match(text):
        return False
    if score_pattern.match(text):
        return False
    return True

def clean_part(text):
    text = text.strip()
    for prefix in ["yellow card", "red card", "goal"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    return text

for ke in key_elements:
    try:
        sub_elements = ke.find_elements(By.CSS_SELECTOR, '.fPSBzf.cMGtQw, .fPSBzf.etJpkR')
        for se in sub_elements:
            se_text = se.text.strip()
            parts = se_text.split("\n")
            cleaned_parts = [clean_part(p) for p in parts if p.strip() != ""]
            names = [p for p in cleaned_parts if is_valid_name(p)]
            
            title_text = None
            try:
                svg = se.find_element(By.TAG_NAME, 'svg')
                try:
                    title_text = driver.execute_script("return arguments[0].querySelector('title')?.textContent;", svg)
                except Exception:
                    title_text = None
            except Exception:
                title_text = None

            event_type = None
            if title_text:
                lower_title = title_text.lower()
                if "yellow card" in lower_title:
                    event_type = "yellow_cards"
                elif "red card" in lower_title:
                    event_type = "red_cards"
                elif "goal" in lower_title:
                    event_type = "goals"
            if not event_type:
                lower_text = se_text.lower()
                if "yellow card" in lower_text:
                    event_type = "yellow_cards"
                elif "red card" in lower_text:
                    event_type = "red_cards"
                elif "goal" in lower_text:
                    event_type = "goals"
                else:
                    event_type = "substitution"

            if event_type == "substitution":
                if len(names) >= 2:
                    substitution = {"out": names[0], "in": names[1]}
                    if substitution not in events["substitutions"]:
                        events["substitutions"].append(substitution)
            elif event_type == "yellow_cards":
                if names:
                    card_name = names[0]
                    if card_name not in events["yellow_cards"]:
                        events["yellow_cards"].append(card_name)
            elif event_type == "red_cards":
                if names:
                    card_name = names[0]
                    if card_name not in events["red_cards"]:
                        events["red_cards"].append(card_name)
            elif event_type == "goals":
                if len(names) == 1:
                    goal_name = names[0]
                    if goal_name not in events["goals"]:
                        events["goals"].append(goal_name)
                elif len(names) >= 2:
                    goal_event = {"scorer": names[0], "assist": names[1]}
                    if goal_event not in events["goals"]:
                        events["goals"].append(goal_event)
    except Exception:
        continue

for type in events:
    print(type)
    print(events[type])

driver.quit()

input("hi")