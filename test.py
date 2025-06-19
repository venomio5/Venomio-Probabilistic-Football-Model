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
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QDateEdit
from PyQt5.QtCore import QDate

app = QApplication([])

window = QWidget()
layout = QVBoxLayout()

date_edit = QDateEdit()
date_edit.setCalendarPopup(True)
date_edit.setDate(QDate.currentDate())  # Optional: sets current date
date_edit.setStyleSheet("""
    QCalendarWidget QToolButton {
        background-color: #2c2f33;
        color: white;
        font-weight: bold;
        border: none;
    }
    QCalendarWidget QSpinBox {
        color: white;
    }
    QCalendarWidget QWidget {
        background-color: #23272a;
        color: white;
    }
    QCalendarWidget QAbstractItemView:enabled {
        background-color: #2c2f33;
        color: white;
        selection-background-color: #7289da;
    }
""")


layout.addWidget(date_edit)
window.setLayout(layout)
window.show()

app.exec_()
