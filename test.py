from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable, Sequence
import pandas as pd
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import requests
import math
from fuzzywuzzy import process, fuzz
import re
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import json
from tqdm import tqdm
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QToolButton, QMenu

app = QApplication([])

window = QWidget()
layout = QHBoxLayout(window)

button = QToolButton()
button.setText("Execute")
button.setPopupMode(QToolButton.MenuButtonPopup)  # Action + menu icon

# Optional: primary click action
button.clicked.connect(lambda: print("Primary Action"))

# Add dropdown menu
menu = QMenu()
menu.addAction("Alternative 1", lambda: print("Alt 1"))
menu.addAction("Alternative 2", lambda: print("Alt 2"))
button.setMenu(menu)

layout.addWidget(button)
window.show()
app.exec_()
