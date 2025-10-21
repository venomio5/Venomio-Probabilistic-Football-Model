import core
import pandas as pd
from datetime import datetime
import ast

current_date = "2023-07-05"
datetime_obj = pd.to_datetime(current_date)

#core.MonteCarloSim(match_id=4772)
#core.FillTeamsData(3)
core.ScrapeMatchesData(upto_date=current_date)