from datetime import datetime, timedelta
import sys
from PyQt5.QtCore import Qt, QRunnable, QObject, pyqtSignal, QThreadPool, QDate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                            QComboBox, QScrollArea, QLabel, QPushButton, QCheckBox, QDoubleSpinBox,
                            QDialog, QListWidget, QListWidgetItem, QTabWidget, QFormLayout, QPlainTextEdit, QHeaderView,
                            QGridLayout, QLineEdit, QGroupBox, QSpinBox, QDateEdit, QTableWidget, QTableWidgetItem)
from functools import partial
import core
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

class WorkerSignals(QObject):
    finished = pyqtSignal() 
    error = pyqtSignal(tuple) 
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class UpdateWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(UpdateWorker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            import traceback
            self.signals.error.emit((e.__class__, e, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Venomio Probabilistic Football Model v7")
        self.setGeometry(100, 100, 1920, 1080)
        self.showMaximized()
        self.threadpool = QThreadPool()
        self.open_windows = []
        
        self.vpfm_db = core.DB
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left section
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #1a1a1a;")
        main_layout.addWidget(left_widget, stretch=7)
        self.setup_left_section(left_widget)

        # Right section
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #1a1a1a;")
        self.setup_right_section(right_widget)
        main_layout.addWidget(right_widget, stretch=3)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# ----------------------- GAMES SECTION  -----------------------
    def setup_left_section(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        filters_layout = QHBoxLayout()
        self.date_filter = QComboBox()
        self.date_filter.addItem("Today", "today")
        self.date_filter.addItem("Past", "past")
        self.date_filter.addItem("Upcoming", "upcoming")
        self.date_filter.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: white;
                selection-background-color: #138585;
                selection-color: white;
            }
        """)

        self.league_filter = QComboBox()
        self.league_filter.addItem("All", "all")
        self.league_filter.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: white;
                selection-background-color: #138585;
                selection-color: white;
            }
        """)

        filters_layout.addWidget(self.date_filter)
        filters_layout.addWidget(self.league_filter)
        layout.addLayout(filters_layout)

        # Matches scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")

        self.matches_container = QWidget()
        self.matches_container.setLayout(QVBoxLayout())
        self.matches_container.layout().setAlignment(Qt.AlignTop)
        self.matches_container.layout().setSpacing(10)
        self.matches_container.layout().setContentsMargins(20, 0, 20, 0)
        
        self.scroll_area.setWidget(self.matches_container)
        layout.addWidget(self.scroll_area)

        # Connect signals
        self.date_filter.currentIndexChanged.connect(self.filter_matches)
        self.league_filter.currentIndexChanged.connect(self.filter_matches)

        # Initialize data
        self.load_fixtures()

    def load_fixtures(self):
        self.all_matches = []
        fixtures_query = "SELECT * FROM schedule_data"
        self.all_matches = self.vpfm_db.select(fixtures_query)

        self.all_matches['datetime'] = self.all_matches.apply(
            lambda row: datetime.combine(row['date'], datetime.min.time()) + row['local_time'],
            axis=1
        )

        self.all_matches["league_name"] = self.all_matches["league_id"].apply(
            lambda lid: core.get_league_name_by_id(lid)
        )

        self.all_matches["home_team"] = self.all_matches["home_team_id"].apply(
            lambda tid: core.get_team_name_by_id(tid)
        )

        self.all_matches["away_team"] = self.all_matches["away_team_id"].apply(
            lambda tid: core.get_team_name_by_id(tid)
        )

        league_ids = {row["league_id"] for _, row in self.all_matches.iterrows()}
        league_names = [core.get_league_name_by_id(league_id) for league_id in league_ids if core.get_league_name_by_id(league_id) is not None]
        league_names.sort()
        self.league_filter.clear()
        self.league_filter.addItem("All", "all")
        for league in league_names:
            self.league_filter.addItem(league, league)

        self.filter_matches()
    
    def filter_matches(self):
        current_time = datetime.now()
        two_hours_ago = current_time - timedelta(hours=2.1)

        today_matches = {"inPlay": [], "later": []}
        upcoming_matches = []
        past_matches = []

        for idx, row in self.all_matches.iterrows():
            if row['date'] == current_time.date():
                if two_hours_ago <= row['datetime'] <= current_time:
                    today_matches["inPlay"].append(row)
                elif row['datetime'] > current_time:
                    today_matches["later"].append(row)
                else:
                    past_matches.append(row)
            else:
                if row['date'] > current_time.date():
                    upcoming_matches.append(row)
                else:
                    past_matches.append(row)

        today_matches["inPlay"].sort(key=lambda match: match["datetime"])
        today_matches["later"].sort(key=lambda match: match["datetime"])
        upcoming_matches.sort(key=lambda match: match["date"])
        past_matches.sort(key=lambda match: match["date"])

        # Apply filters
        selected_date = self.date_filter.currentData()
        selected_league = self.league_filter.currentData()

        if selected_date == "today":
            matches_to_display = today_matches["inPlay"] + today_matches["later"]
        elif selected_date == "past":
            matches_to_display = past_matches
        else:
            matches_to_display = upcoming_matches

        if selected_league != "all":
            matches_to_display = [m for m in matches_to_display if m["league_name"] == selected_league]

        self.display_matches(matches_to_display, today_matches)

    def display_matches(self, matches, today_matches):
        while self.matches_container.layout().count():
            child = self.matches_container.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not matches:
            self.matches_container.layout().addWidget(QLabel("No games", styleSheet="color: white;"))
            return

        selected_date = self.date_filter.currentData()
        selected_league = self.league_filter.currentData()

        if selected_date == "today":
            in_play = [m for m in today_matches["inPlay"] if selected_league in ("all", m["league_name"])]
            later = [m for m in today_matches["later"] if selected_league in ("all", m["league_name"])]

            if in_play:
                self.add_section_header("In-Play")
                for match in in_play:
                    self.create_match_box(match)

            if later:
                self.add_section_header("Later")
                for match in later:
                    self.create_match_box(match)
        elif selected_date == "past":
            self.add_section_header("Past")
            for match in matches:
                self.create_match_box(match)
        else:
            grouped = {}
            for match in matches:
                date_str = str(match["date"])
                grouped.setdefault(date_str, []).append(match)

            for date_str, group in sorted(grouped.items()):
                self.add_section_header(date_str)
                for match in group:
                    self.create_match_box(match)

        self.matches_container.layout().addStretch()

    def add_section_header(self, text):
        header = QLabel(text)
        header.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 16px;
            margin-top: 15px;
            margin-bottom: 5px;
        """)
        self.matches_container.layout().addWidget(header)

    def create_match_box(self, match):
        box = QWidget()
        box.setStyleSheet("""
            padding: 15px;
        """)
        box.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(box)

        league = QLabel(match["league_name"])
        league.setStyleSheet("color: white; font-size: 14px;")
        league.setFixedWidth(200)

        teams = QLabel(f"{match['home_team']} - {match['away_team']}")
        teams.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: bold;
            qproperty-alignment: AlignCenter;
        """)

        time = QLabel()
        time.setText(str(match["datetime"]))
        time.setStyleSheet("color: white; font-size: 14px;")
        time.setFixedWidth(100)
        time.setAlignment(Qt.AlignRight)

        layout.addWidget(league)
        layout.addWidget(teams)
        layout.addWidget(time)

        box.mousePressEvent = lambda event: self.on_match_clicked(match)
        self.matches_container.layout().addWidget(box)

# ----------------------- MATCH SECTION  -----------------------
    def on_match_clicked(self, match):
        print(f"{match['home_team']} vs {match['away_team']} {match['schedule_id']}")
        match_window = QMainWindow()
        match_window.setWindowTitle(f"{match['home_team']} vs {match['away_team']}")
        match_window.setStyleSheet("background-color: #1a1a1a; color: white;")
        match_window.setWindowState(Qt.WindowMaximized)
        
        central_widget = QWidget()
        match_window.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        tabs = QTabWidget()
        tabs.setStyleSheet("background-color: #1a1a1a; color: white;")
        main_layout.addWidget(tabs)

        # ------------------- Build Game Tab (default) -------------------
        build_game_tab = QWidget()
        build_game_tab.setStyleSheet("background-color: #1a1a1a; color: white;")
        build_layout = QVBoxLayout(build_game_tab)
        
        form_layout = QFormLayout()

        referee_label = QLabel("Referee Name:")
        referee_label.setStyleSheet("color: white;")
        self.referee_input = QLineEdit()
        self.referee_input.setStyleSheet("background-color: #2a2a2a; color: white;")
        self.referee_input.setPlaceholderText("Type referee name...")
        form_layout.addRow(referee_label, self.referee_input)

        saved_referee = core.get_referee_name(match['schedule_id'])
        if saved_referee:
            self.referee_input.setText(saved_referee)
        
        home_players_label = QLabel(f"{match['home_team']} Players:")
        home_players_label.setStyleSheet("color: white;")
        home_players_input = QPlainTextEdit()
        home_players_input.setStyleSheet("background-color: #2a2a2a; color: white;")
        home_players_input.setPlaceholderText(f"Paste {match['home_team']} players here...")
        form_layout.addRow(home_players_label, home_players_input)
        
        away_players_label = QLabel(f"{match['away_team']} Players:")
        away_players_label.setStyleSheet("color: white;")
        away_players_input = QPlainTextEdit()
        away_players_input.setStyleSheet("background-color: #2a2a2a; color: white;")
        away_players_input.setPlaceholderText(f"Paste {match['away_team']} players here...")
        form_layout.addRow(away_players_label, away_players_input)

        initial_minute_label = QLabel("Match Initial Minute:")
        initial_minute_label.setStyleSheet("color: white;")
        self.initial_minute_spin = QSpinBox()
        self.initial_minute_spin.setMinimum(0)
        self.initial_minute_spin.setMaximum(89)
        self.initial_minute_spin.setValue(0)
        self.initial_minute_spin.setStyleSheet("background-color: #2a2a2a; color: white;")
        form_layout.addRow(initial_minute_label, self.initial_minute_spin)

        home_goals_label = QLabel("Home Initial Goals:")
        home_goals_label.setStyleSheet("color: white;")
        self.home_goals_spin = QSpinBox()
        self.home_goals_spin.setMinimum(0)
        self.home_goals_spin.setMaximum(10)
        self.home_goals_spin.setValue(0)
        self.home_goals_spin.setStyleSheet("background-color: #2a2a2a; color: white;")
        form_layout.addRow(home_goals_label, self.home_goals_spin)
        
        away_goals_label = QLabel("Away Initial Goals:")
        away_goals_label.setStyleSheet("color: white;")
        self.away_goals_spin = QSpinBox()
        self.away_goals_spin.setMinimum(0)
        self.away_goals_spin.setMaximum(10)
        self.away_goals_spin.setValue(0)
        self.away_goals_spin.setStyleSheet("background-color: #2a2a2a; color: white;")
        form_layout.addRow(away_goals_label, self.away_goals_spin)
        
        home_subs_label = QLabel("Home Initial # Subs:")
        home_subs_label.setStyleSheet("color: white;")
        self.home_subs_spin = QSpinBox()
        self.home_subs_spin.setMinimum(0)
        self.home_subs_spin.setMaximum(5)
        self.home_subs_spin.setValue(5)
        self.home_subs_spin.setStyleSheet("background-color: #2a2a2a; color: white;")
        form_layout.addRow(home_subs_label, self.home_subs_spin)
        
        away_subs_label = QLabel("Away Initial # Subs:")
        away_subs_label.setStyleSheet("color: white;")
        self.away_subs_spin = QSpinBox()
        self.away_subs_spin.setMinimum(0)
        self.away_subs_spin.setMaximum(5)
        self.away_subs_spin.setValue(5)
        self.away_subs_spin.setStyleSheet("background-color: #2a2a2a; color: white;")
        form_layout.addRow(away_subs_label, self.away_subs_spin)
        
        build_layout.addLayout(form_layout)

        home_saved_players = core.get_saved_lineup(match['schedule_id'], "home")
        away_saved_players = core.get_saved_lineup(match['schedule_id'], "away")

        if home_saved_players:
            home_players_text = "\n".join(player["player_id"] for player in home_saved_players)
            home_players_input.setPlainText(home_players_text)
        
        if away_saved_players:
            away_players_text = "\n".join(player["player_id"] for player in away_saved_players)
            away_players_input.setPlainText(away_players_text)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        convert_button = QPushButton("Create Line-ups")
        convert_button.setStyleSheet("background-color: #404040; color: white; padding: 10px;")
        button_layout.addWidget(convert_button)

        self.submit_button = QPushButton("Run Game Simulations")
        self.submit_button.setStyleSheet("background-color: #138585; color: white; padding: 10px;")
        self.submit_button.setEnabled(False)
        button_layout.addWidget(self.submit_button)

        button_layout.addStretch()
        build_layout.addLayout(button_layout)

        def create_players_table(players):
            table = QTableWidget(len(players), 5)
            table.setHorizontalHeaderLabels(["Player", "YC", "RC", "On Field", "Bench"])
            table.horizontalHeader().setStretchLastSection(False)
            table.verticalHeader().setVisible(False)
            table.setStyleSheet(
                "QTableWidget { background-color: #2a2a2a; color: white; }"
                "QHeaderView::section { background-color: #138585; color: white; }"
            )

            table.setColumnWidth(0, 160)
            for i in range(1, 5):
                table.setColumnWidth(i, 60)

            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

            for row, player in enumerate(players):
                if isinstance(player, dict):
                    name_text        = player.get("player_id", "")
                    yc_checked       = player.get("yellow_card", False)
                    rc_checked       = player.get("red_card",   False)
                    on_field_checked = player.get("on_field",   False)
                    bench_checked    = player.get("bench",      False)
                else:  
                    name_text        = str(player)
                    yc_checked = rc_checked = False
                    on_field_checked = row < 11         
                    bench_checked    = not on_field_checked

                name_item = QTableWidgetItem(name_text)
                name_item.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, 0, name_item)

                yc_cb = QCheckBox()
                yc_cb.setChecked(yc_checked)
                rc_cb = QCheckBox()
                rc_cb.setChecked(rc_checked)
                table.setCellWidget(row, 1, yc_cb)
                table.setCellWidget(row, 2, rc_cb)

                on_field_cb = QCheckBox()
                bench_cb    = QCheckBox()
                on_field_cb.setFixedWidth(55)
                bench_cb.setFixedWidth(55)
                on_field_cb.setChecked(on_field_checked)
                bench_cb.setChecked(bench_checked)

                table.setCellWidget(row, 3, on_field_cb)
                table.setCellWidget(row, 4, bench_cb)

            return table

        def on_create_lineups():
            referee_name = self.referee_input.text().strip()
            if referee_name:
                core.send_referee_name_to_db(referee_name, match['schedule_id'])

            if home_saved_players:
                home_final_players = home_saved_players
            else:
                home_starters, home_benchers = core.match_players(match['home_team_id'], home_players_input)
                home_final_players = home_starters + home_benchers

            if away_saved_players:
                away_final_players = away_saved_players
            else:
                away_starters, away_benchers = core.match_players(match['away_team_id'], away_players_input)
                away_final_players = away_starters + away_benchers

            form_layout.labelForField(home_players_input).hide()
            form_layout.labelForField(away_players_input).hide()
            home_players_input.hide()
            away_players_input.hide()
            convert_button.hide()

            tables_layout = QHBoxLayout()
            tables_layout.setSpacing(20)

            self.home_players_table = create_players_table(home_final_players)
            self.away_players_table = create_players_table(away_final_players)
            tables_layout.addWidget(self.home_players_table)
            tables_layout.addWidget(self.away_players_table)

            build_layout.insertLayout(1, tables_layout)
            self.submit_button.setEnabled(True)

        convert_button.clicked.connect(on_create_lineups)

        def extract_players_data(table: QTableWidget) -> list[dict]:
            players_data = []
            for row in range(table.rowCount()):
                player_id = table.item(row, 0).text()

                yc = table.cellWidget(row, 1).isChecked()
                rc = table.cellWidget(row, 2).isChecked()
                on_field = table.cellWidget(row, 3).isChecked()
                bench = table.cellWidget(row, 4).isChecked()

                players_data.append({
                    "player_id": player_id,
                    "yellow_card": yc,
                    "red_card": rc,
                    "on_field": on_field,
                    "bench": bench
                })

            return players_data

        def run_build_game():        
            home_initial_goals = self.home_goals_spin.value()
            away_initial_goals = self.away_goals_spin.value()
            match_initial_time = self.initial_minute_spin.value()
            home_initial_n_subs = self.home_subs_spin.value()
            away_initial_n_subs = self.away_subs_spin.value()
            referee_name = self.referee_input.text()

            home_players_data = extract_players_data(self.home_players_table)
            away_players_data = extract_players_data(self.away_players_table)

            core.send_lineup_to_db(home_players_data, match['schedule_id'], "home")
            core.send_lineup_to_db(away_players_data, match['schedule_id'], "away")

            task_description = f"Simulating: {match['home_team']} vs {match['away_team']}"
            list_item = self.add_task_to_queue(task_description)
        
            worker = UpdateWorker(
                core.Alg,
                schedule_id=int(match['schedule_id']),
                home_initial_goals=home_initial_goals,
                away_initial_goals=away_initial_goals,
                match_initial_time=match_initial_time,
                home_n_subs_avail=home_initial_n_subs,
                away_n_subs_avail=away_initial_n_subs
            )
            worker.signals.finished.connect(lambda: self.remove_task_from_queue(list_item))
            worker.signals.error.connect(lambda err: print("Simulation error:", err))
            worker.signals.result.connect(lambda res: print("Simulation finished with result:", res))
            self.threadpool.start(worker)
        
        self.submit_button.clicked.connect(run_build_game)

        # ------------------- Odds Tab -------------------
        odds_tab = QWidget()
        odds_tab.setStyleSheet("background-color: #1a1a1a; color: white;")
        odds_layout = QVBoxLayout(odds_tab)

        # ----- Simulation Parameters -----
        simulation_params_group = QGroupBox("Simulation Parameters")
        simulation_params_group.setStyleSheet("background-color: #1a1a1a; color: white; font-weight: bold;")
        sim_params_layout = QVBoxLayout(simulation_params_group)

        minute_frame = QHBoxLayout()
        current_minute_spin = QSpinBox()
        current_minute_spin.setRange(0, 90)
        current_minute_spin.setValue(0)

        current_minute_spin.valueChanged.connect(lambda _: update_odds())

        dash_label = QLabel("-")
        dash_label.setStyleSheet("color: white;")

        max_minute_spin = QSpinBox()
        max_minute_spin.setRange(1, 90)
        max_minute_spin.setValue(90)
        max_minute_spin.valueChanged.connect(lambda _: update_odds())

        minute_frame.addWidget(current_minute_spin)
        minute_frame.addWidget(dash_label)
        minute_frame.addWidget(max_minute_spin)
        sim_params_layout.addLayout(minute_frame)

        goals_frame = QHBoxLayout()

        home_goals_layout = QVBoxLayout()
        home_goals_label = QLabel(f"{match['home_team']} Goals:")
        home_goals_label.setStyleSheet("color: white;")
        home_goals_spin = QSpinBox()
        home_goals_spin.setRange(0, 10)
        home_goals_spin.setValue(0)
        home_goals_spin.valueChanged.connect(lambda _: update_odds())
        home_goals_layout.addWidget(home_goals_label)
        home_goals_layout.addWidget(home_goals_spin)

        away_goals_layout = QVBoxLayout()
        away_goals_label = QLabel(f"{match['away_team']} Goals:")
        away_goals_label.setStyleSheet("color: white;")
        away_goals_spin = QSpinBox()
        away_goals_spin.setRange(0, 10)
        away_goals_spin.setValue(0)
        away_goals_spin.valueChanged.connect(lambda _: update_odds())
        away_goals_layout.addWidget(away_goals_label)
        away_goals_layout.addWidget(away_goals_spin)

        goals_frame.addLayout(home_goals_layout)
        goals_frame.addLayout(away_goals_layout)
        sim_params_layout.addLayout(goals_frame)

        # xG
        xg_frame = QHBoxLayout()
        home_xg_label = QLabel(f"<span style='color:#FFFFFF;'>xG: </span> <span style='color:#138585;'>N/A</span>")
        home_xg_label.setStyleSheet("color: white;")
        away_xg_label = QLabel(f"<span style='color:#FFFFFF;'>xG: </span> <span style='color:#138585;'>N/A</span>")
        away_xg_label.setStyleSheet("color: white;")
        xg_frame.addWidget(home_xg_label)
        xg_frame.addWidget(away_xg_label)
        sim_params_layout.addLayout(xg_frame)

        odds_layout.addWidget(simulation_params_group)

        # --- Match Odds Section
        match_odds_group = QGroupBox("Match")
        match_odds_group.setStyleSheet("font-weight: bold;")
        match_odds_layout = QFormLayout(match_odds_group)
        home_odds_label = QLabel('0')
        match_odds_layout.addRow("Home:", home_odds_label)
        away_odds_label = QLabel('0')
        match_odds_layout.addRow("Away:", away_odds_label)
        draw_odds_label = QLabel('0')
        match_odds_layout.addRow("Draw:", draw_odds_label)
        odds_layout.addWidget(match_odds_group)
        home_odds_label.setStyleSheet("color: #138585;")
        away_odds_label.setStyleSheet("color: #138585;")
        draw_odds_label.setStyleSheet("color: #138585;")

        # --- Asian Handicap Section ---
        asian_group = QGroupBox("Asian Handicap")
        asian_group.setStyleSheet("font-weight: bold;") 
        asian_layout = QFormLayout(asian_group)
        asian_dropdown = QComboBox()
        asian_dropdown.addItems(["+1.5", "+2.5", "-1.5", "-2.5"])
        asian_layout.addRow("Handicap", asian_dropdown)
        home_asian_odds_label = QLabel('0')
        asian_layout.addRow("Home Odds:", home_asian_odds_label)
        away_asian_odds_label = QLabel('0')
        asian_layout.addRow("Away Odds:", away_asian_odds_label)
        odds_layout.addWidget(asian_group)
        home_asian_odds_label.setStyleSheet("color: #138585;")
        away_asian_odds_label.setStyleSheet("color: #138585;")

        # --- Total Over/Under Section ---
        totals_group = QGroupBox("Total Over/Under")
        totals_group.setStyleSheet("font-weight: bold;")
        totals_layout = QFormLayout(totals_group)
        totals_dropdown = QComboBox()

        totals_dropdown.addItems(["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"])
        totals_layout.addRow("Total", totals_dropdown)
        under_odds_label = QLabel('0')
        totals_layout.addRow("Under:", under_odds_label)
        over_odds_label = QLabel('0')
        totals_layout.addRow("Over:", over_odds_label)
        odds_layout.addWidget(totals_group)
        under_odds_label.setStyleSheet("color: #138585;")
        over_odds_label.setStyleSheet("color: #138585;")

        # --- Correct Score Section ---
        score_group = QGroupBox("Correct Score") 
        score_group.setStyleSheet("font-weight: bold;") 
        score_layout = QVBoxLayout(score_group) 

        correct_score_labels = {}
        score_odds_dict = {} 

        correct_scores = [
            "0-0", "0-1", "0-2", "0-3",
            "1-0", "1-1", "1-2", "1-3",
            "2-0", "2-1", "2-2", "2-3",
            "3-0", "3-1", "3-2", "3-3",
            "Any Other Home Win", "Any Other Away Win", "Any Other Draw"
        ]

        for score in correct_scores:
            label = QLabel(f"<span style='color:#FFFFFF;'>{score}:</span> <span style='color:#138585;'>N/A</span>") 
            correct_score_labels[score] = label
            score_layout.addWidget(label)

        odds_layout.addWidget(score_group)

        # --- Team Totals Section ---
        team_totals_group = QGroupBox("Team Totals")
        team_totals_group.setStyleSheet("font-weight: bold;")
        team_totals_layout = QHBoxLayout(team_totals_group)

        # -- Home Subsection --
        home_team_totals_box = QGroupBox("Home Totals")
        home_team_totals_box.setStyleSheet("font-weight: bold;")
        home_team_totals_layout = QFormLayout(home_team_totals_box)
        home_totals_dropdown = QComboBox()
        home_totals_dropdown.addItems(["0.5", "1.5", "2.5"])
        home_team_totals_layout.addRow("Total", home_totals_dropdown)
        home_team_totals_under_odds = QLabel('0')
        home_team_totals_layout.addRow("Under:", home_team_totals_under_odds)
        home_team_totals_over_odds = QLabel('0')
        home_team_totals_layout.addRow("Over:", home_team_totals_over_odds)
        team_totals_layout.addWidget(home_team_totals_box)
        home_team_totals_under_odds.setStyleSheet("color: #138585;")
        home_team_totals_over_odds.setStyleSheet("color: #138585;")

        # -- Away Subsection --
        away_team_totals_box = QGroupBox("Away Totals")
        away_team_totals_box.setStyleSheet("font-weight: bold;")
        away_team_totals_layout = QFormLayout(away_team_totals_box)
        away_totals_dropdown = QComboBox()
        away_totals_dropdown.addItems(["0.5", "1.5", "2.5"])
        away_team_totals_layout.addRow("Total", away_totals_dropdown)
        away_team_totals_under_odds = QLabel('0')
        away_team_totals_layout.addRow("Under:", away_team_totals_under_odds)
        away_team_totals_over_odds = QLabel('0')
        away_team_totals_layout.addRow("Over:", away_team_totals_over_odds)
        team_totals_layout.addWidget(away_team_totals_box)
        away_team_totals_under_odds.setStyleSheet("color: #138585;")
        away_team_totals_over_odds.setStyleSheet("color: #138585;")

        odds_layout.addWidget(team_totals_group)

        def get_aggregated_goals(shots_df, home_team_id, start_minute, start_home_goals, start_away_goals):
            if shots_df is None or shots_df.empty:
                return pd.DataFrame(columns=['sim_id', 'minute', 'home_goals', 'away_goals'])
            
            df = shots_df.copy()
            df = df[df['minute'] >= start_minute]
            
            df = df.sort_values(['sim_id', 'minute']).reset_index(drop=True)
            
            df['squad']   = pd.to_numeric(df['squad'],   errors='coerce').astype('Int64')
            df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce').fillna(0).astype(int)
            
            df['is_home']   = df['squad'] == int(home_team_id)
            df['home_goal'] = ((df['outcome'] == 1) &  df['is_home']).astype(int)
            df['away_goal'] = ((df['outcome'] == 1) & ~df['is_home']).astype(int)
            
            df['home_goal_cum'] = df.groupby('sim_id')['home_goal'].cumsum() + start_home_goals
            df['away_goal_cum'] = df.groupby('sim_id')['away_goal'].cumsum() + start_away_goals
            
            agg = (
                df.groupby(['sim_id', 'minute'])
                .agg(home_goals=('home_goal_cum', 'last'),
                    away_goals=('away_goal_cum', 'last'))
                .reset_index()
            )
            
            max_minute = 90
            full_index = pd.MultiIndex.from_product(
                [agg['sim_id'].unique(), range(max_minute + 1)],
                names=['sim_id', 'minute']
            )
            
            agg = (
                agg.set_index(['sim_id', 'minute'])
                .reindex(full_index)
                .groupby(level=0)
                .ffill()
                .fillna({'home_goals': start_home_goals, 'away_goals': start_away_goals})
                .reset_index()
            )
            
            return agg

        simulation_data = None
        aggregated_df   = pd.DataFrame() 
        def load_simulation_data():
            nonlocal simulation_data, aggregated_df
            schedule_id = int(match['schedule_id']) 
            sql_query = "SELECT * FROM simulation_data WHERE schedule_id = %s"
            simulation_data = self.vpfm_db.select(sql_query, (schedule_id,))

            aggregated_df = get_aggregated_goals(
                simulation_data,
                int(match['home_team_id']),
                int(current_minute_spin.value()),
                int(home_goals_spin.value()),
                int(away_goals_spin.value())
            )
        load_simulation_data()

        def update_odds():
            nonlocal aggregated_df
            if aggregated_df.empty:
                return
            
            aggregated_df = get_aggregated_goals(
                simulation_data,
                int(match['home_team_id']),
                int(current_minute_spin.value()),
                int(home_goals_spin.value()),
                int(away_goals_spin.value())
            ) 

            current_minute = current_minute_spin.value()
            max_minute = max_minute_spin.value()
            home_goals = home_goals_spin.value()
            away_goals = away_goals_spin.value()

            filtered_data = aggregated_df[
                (aggregated_df['minute'] == current_minute) & 
                (aggregated_df['home_goals'] == home_goals) & 
                (aggregated_df['away_goals'] == away_goals)
            ]

            relevant_sim_ids = filtered_data['sim_id'].unique()

            final_data = aggregated_df[
                (aggregated_df['minute'] == max_minute) &
                (aggregated_df['sim_id'].isin(relevant_sim_ids))
            ]

            total_final_data = len(final_data)

            # --- xG ---
            total_home_goals = final_data["home_goals"].sum() / total_final_data if total_final_data != 0 else 0
            total_away_goals = final_data["away_goals"].sum() / total_final_data if total_final_data != 0 else 0
            home_xg_label.setText(f"<span style='color:#FFFFFF;'>xG: </span> <span style='color:#138585;'>{round(total_home_goals, 3)}</span>")
            away_xg_label.setText(f"<span style='color:#FFFFFF;'>xG: </span> <span style='color:#138585;'>{round(total_away_goals, 3)}</span>")

            # --- Match Odds ---
            home_wins = len(final_data[final_data['home_goals'] > final_data['away_goals']])
            away_wins = len(final_data[final_data['home_goals'] < final_data['away_goals']])
            draws = len(final_data[final_data['home_goals'] == final_data['away_goals']])

            home_odds = round(1 / (home_wins / total_final_data ), 3) if total_final_data != 0 and home_wins != 0 else 0
            draw_odds = round(1 / (draws / total_final_data ), 3)  if total_final_data != 0 and draws != 0 else 0
            away_odds = round(1 / (away_wins / total_final_data ), 3) if total_final_data != 0 and away_wins != 0 else 0
            home_odds_label.setText(str(home_odds))
            away_odds_label.setText(str(away_odds))
            draw_odds_label.setText(str(draw_odds))

            # --- Totals (Over/Under) ---
            teams_totals = float(totals_dropdown.currentText())
            count_teams_under = len(final_data[final_data['home_goals'] + final_data['away_goals'] < teams_totals])
            count_teams_over = len(final_data[final_data['home_goals'] + final_data['away_goals'] > teams_totals])
            under_teams_odds = round(1 / (count_teams_under / total_final_data), 3)   if total_final_data != 0 and count_teams_under != 0 else 0
            over_teams_odds = round(1 / (count_teams_over / total_final_data), 3)  if total_final_data != 0 and count_teams_over != 0 else 0
            under_odds_label.setText(str(under_teams_odds))
            over_odds_label.setText(str(over_teams_odds))

            # --- Correct Score ---
            score_counts = final_data.groupby(['home_goals', 'away_goals']).size()

            for score in correct_score_labels.keys():
                if "Any Other" in score:
                    continue

                home_goals, away_goals = map(int, score.replace(" ", "").split("-"))
                count = score_counts.get((home_goals, away_goals), 0)
                odds = round(1 / (count / total_final_data), 3)  if total_final_data != 0 and count != 0 else 0

                if odds == 0:
                    odds_str = f"<span style='color:#138585;'>N/A</span>"
                    score_odds_dict[score] = 10**30
                else:
                    odds_str = f"<span style='color:#138585;'>{odds}</span>" 
                    score_odds_dict[score] = odds

                correct_score_labels[score].setText(f"<span style='color:#FFFFFF;'>{score}:</span> {odds_str}")

            any_other_home_win = sum(score_counts[(h, a)] for h in range(4, 11) for a in range(0, 4) if (h, a) in score_counts)
            any_other_away_win = sum(score_counts[(h, a)] for h in range(0, 4) for a in range(4, 11) if (h, a) in score_counts)
            any_other_draw = sum(score_counts[(h, h)] for h in range(4, 11) if (h, h) in score_counts)

            aggregated_home_odds = round(1 / (any_other_home_win / total_final_data), 3) if total_final_data != 0 and any_other_home_win != 0 else 10**30
            aggregated_away_odds = round(1 / (any_other_away_win / total_final_data), 3) if total_final_data != 0 and any_other_away_win != 0 else 10**30
            aggregated_draw_odds = round(1 / (any_other_draw / total_final_data), 3) if total_final_data != 0 and any_other_draw != 0 else 10**30

            score_odds_dict["Home Win 4+"] = aggregated_home_odds
            score_odds_dict["Away Win 4+"] = aggregated_away_odds
            score_odds_dict["Draw +4"] = aggregated_draw_odds
            
            correct_score_labels["Any Other Home Win"].setText(
                f"<span style='color:#FFFFFF;'>Any Other Home Win: </span> <span style='color:#138585;'>{aggregated_home_odds if aggregated_home_odds != 10**30 else 'N/A'}</span>"
            )
            correct_score_labels["Any Other Away Win"].setText(
                f"<span style='color:#FFFFFF;'>Any Other Away Win: </span> <span style='color:#138585;'>{aggregated_away_odds if aggregated_away_odds != 10**30 else 'N/A'}</span>"
            )
            correct_score_labels["Any Other Draw"].setText(
                f"<span style='color:#FFFFFF;'>Any Other Draw: </span> <span style='color:#138585;'>{aggregated_draw_odds if aggregated_draw_odds != 10**30 else 'N/A'}</span>"
            )
            # --- Asian Handicap ---
            handicap_str = asian_dropdown.currentText() 
            handicap_val = float(handicap_str.replace("+", "").replace("-", ""))
            if handicap_str.startswith("+"):
                home_asian_wins = len(final_data[final_data['home_goals'] + handicap_val > final_data['away_goals']])
                away_asian_wins = len(final_data[final_data['away_goals'] - handicap_val > final_data['home_goals']])
            elif handicap_str.startswith("-"):
                home_asian_wins = len(final_data[final_data['home_goals'] - handicap_val > final_data['away_goals']])
                away_asian_wins = len(final_data[final_data['away_goals'] + handicap_val > final_data['home_goals']])
            home_asian_odds = round(1 / (home_asian_wins / total_final_data), 3) if total_final_data != 0 and home_asian_wins != 0 else 0
            away_asian_odds = round(1 / (away_asian_wins / total_final_data), 3) if total_final_data != 0 and away_asian_wins != 0 else 0
            home_asian_odds_label.setText(str(home_asian_odds))
            away_asian_odds_label.setText(str(away_asian_odds))

            # -- Home Totals --
            home_threshold = float(home_totals_dropdown.currentText())
            count_home_under = len(final_data[final_data['home_goals'] < home_threshold])
            count_home_over = len(final_data[final_data['home_goals']  > home_threshold])
            under_home_odds = round(1 / (count_home_under / total_final_data), 3) if total_final_data != 0 and count_home_under != 0 else 0
            over_home_odds = round(1 / (count_home_over / total_final_data), 3) if total_final_data != 0 and count_home_over != 0 else 0
            home_team_totals_under_odds.setText(str(under_home_odds))
            home_team_totals_over_odds.setText(str(over_home_odds))

            # -- Away Totals --
            away_threshold = float(away_totals_dropdown.currentText())
            count_away_under = len(final_data[final_data['away_goals'] < away_threshold])
            count_away_over = len(final_data[final_data['away_goals']  > away_threshold])
            under_away_odds = round(1 / (count_away_under / total_final_data), 3) if total_final_data != 0 and count_away_under != 0 else 0
            over_away_odds = round(1 / (count_away_over / total_final_data), 3) if total_final_data != 0 and count_away_over != 0 else 0
            away_team_totals_under_odds.setText(str(under_away_odds))
            away_team_totals_over_odds.setText(str(over_away_odds))     

        totals_dropdown.currentIndexChanged.connect(update_odds)
        asian_dropdown.currentIndexChanged.connect(update_odds)
        home_totals_dropdown.currentIndexChanged.connect(update_odds)
        away_totals_dropdown.currentIndexChanged.connect(update_odds)

        update_odds()

        # ------------------- Simulator -------------------
        simulator_tab = QWidget()
        simulator_tab.setStyleSheet("background-color: #1a1a1a; color: white;")
        sim_layout = QVBoxLayout(simulator_tab)

        inner_tabs = QTabWidget()
        sim_layout.addWidget(inner_tabs)
        
        match_tab = QWidget()
        tw_tab = QWidget()
        score_tab = QWidget()
        
        inner_tabs.addTab(match_tab, "Match")
        inner_tabs.addTab(tw_tab, "2-Way")
        inner_tabs.addTab(score_tab, "Score")
        
        inner_tabs.setStyleSheet("""
            QTabBar::tab {
                background: #1a1a1a;
                color: white;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background: #138585;
            }
        """)

        # MATCH TAB
        match_layout = QVBoxLayout(match_tab)
        main_horizontal = QHBoxLayout()
        match_layout.addLayout(main_horizontal)
        results_container = QVBoxLayout()
        main_horizontal.addLayout(results_container, stretch=1)

        results_grid = QGridLayout()
        results_container.addLayout(results_grid)
        match_results_headers = ["Selection", "Profit/Loss", "%", "EV"]
        for col, header in enumerate(match_results_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(label, 0, col)

        outcomes = ["Home", "Away", "Draw"]
        match_profit_labels = {}
        match_probability_labels = {}
        match_ev_labels = {}
        for i, outcome in enumerate(outcomes):
            row = i + 1
            lbl = QLabel(outcome)
            results_grid.addWidget(lbl, row, 0)

            profit_lbl = QLabel("$0.00")
            profit_lbl.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(profit_lbl, row, 1)
            match_profit_labels[outcome] = profit_lbl

            probability_lbl = QLabel("0.00%")
            probability_lbl.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(probability_lbl, row, 2)
            match_probability_labels[outcome] = probability_lbl

            ev_lbl = QLabel("$0.00")
            ev_lbl.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(ev_lbl, row, 3)
            match_ev_labels[outcome] = ev_lbl

        total_row = len(outcomes) + 1
        total_ev_text = QLabel("Total EV:")
        total_ev_text.setStyleSheet("font-weight: bold;")
        results_grid.addWidget(total_ev_text, total_row, 0, 1, 3)
        match_total_ev_label = QLabel("$0.00")
        match_total_ev_label.setStyleSheet("font-weight: bold;")
        results_grid.addWidget(match_total_ev_label, total_row, 3)

        bets_container = QVBoxLayout()
        main_horizontal.addLayout(bets_container, stretch=3)

        match_bets_grid = QGridLayout()
        bets_container.addLayout(match_bets_grid)
        bet_headers = ["Bet Selection", "Bet Type", "Odds", "Amount", "Action"]
        for col, header in enumerate(bet_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            match_bets_grid.addWidget(label, 0, col)

        match_bet_widgets = []

        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Bet")
        add_btn.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        button_layout.addWidget(add_btn)
        update_btn = QPushButton("🔄")
        update_btn.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        button_layout.addWidget(update_btn)
        button_layout.addStretch()
        match_layout.addLayout(button_layout)
        
        def add_match_bet_row():
            row = 1 + len(match_bet_widgets)
            combo = QComboBox()
            combo.addItems(["Home", "Away", "Draw"])
            match_bets_grid.addWidget(combo, row, 0)
            
            type_combo = QComboBox()
            type_combo.addItems(["Back", "Lay"])
            match_bets_grid.addWidget(type_combo, row, 1)
            
            odds_entry = QLineEdit()
            match_bets_grid.addWidget(odds_entry, row, 2)
            
            amount_entry = QLineEdit()
            match_bets_grid.addWidget(amount_entry, row, 3)
            
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("background-color: red; color: white;")
            delete_btn.clicked.connect(lambda _, r=row: delete_match_bet_row(r))
            match_bets_grid.addWidget(delete_btn, row, 4)
            
            match_bet_widgets.append({
                "row": row,
                "combo": combo,
                "type_combo": type_combo,
                "odds_entry": odds_entry,
                "amount_entry": amount_entry,
                "delete_btn": delete_btn
            })
        
        def delete_match_bet_row(row):
            for bet in match_bet_widgets:
                if bet["row"] == row:
                    for widget in [bet["combo"], bet["type_combo"], bet["odds_entry"], bet["amount_entry"], bet["delete_btn"]]:
                        widget.deleteLater()
                    match_bet_widgets.remove(bet)
                    break
        
        def update_match_trading():
            matched_bets = []
            for bet in match_bet_widgets:
                selection = bet["combo"].currentText()
                bet_type = bet["type_combo"].currentText()
                try:
                    odds = float(bet["odds_entry"].text())
                    amount = float(bet["amount_entry"].text())
                except ValueError:
                    continue
                matched_bets.append({
                    "Selection": selection,
                    "Type": bet_type,
                    "Odds": odds,
                    "Amount": amount
                })
            trading_data = core.MatchTrade(matched_bets)
            total_ev = 0.0
            for outcome, profit_label in match_profit_labels.items():
                pl = trading_data.selections_pl.get(outcome, 0)
                color = "#4df9a2" if pl >= 0 else "#ce1d2e"
                profit_label.setText(f"${pl:.2f}")
                profit_label.setStyleSheet("font-weight: bold; color:" + color + ";")
                
                if outcome == "Home":
                    odds_text = home_odds_label.text()
                elif outcome == "Away":
                    odds_text = away_odds_label.text()
                elif outcome == "Draw":
                    odds_text = draw_odds_label.text()
                
                try:
                    odds_float = float(odds_text)
                    percentage = (1 / odds_float) * 100 if odds_float != 0 else 0.0
                except ValueError:
                    percentage = 0.0
                match_probability_labels[outcome].setText(f"{percentage:.2f}%")
                
                ev = pl * (percentage / 100)
                match_ev_labels[outcome].setText(f"${ev:.2f}")
                total_ev += ev
            match_total_ev_label.setText(f"${total_ev:.2f}")
        
        add_btn.clicked.connect(add_match_bet_row)
        update_btn.clicked.connect(update_match_trading)
    
        # 2-Way TAB  
        tw_layout = QVBoxLayout(tw_tab)
        tw_tab.setLayout(tw_layout)

        tw_market_selector = QComboBox()
        tw_market_selector.addItems(["Totals", "Asian Handicap"])
        tw_layout.addWidget(tw_market_selector)

        main_horizontal = QHBoxLayout()
        tw_layout.addLayout(main_horizontal)

        results_container = QVBoxLayout()
        main_horizontal.addLayout(results_container, stretch=1)

        results_grid = QGridLayout()
        results_container.addLayout(results_grid)

        def update_tw_results_grid():
            for i in reversed(range(results_grid.count())):
                widget = results_grid.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            headers = ["Selection", "Profit/Loss", "%", "EV"]
            for col, header in enumerate(headers):
                label = QLabel(header)
                label.setStyleSheet("font-weight: bold;")
                results_grid.addWidget(label, 0, col)
            if tw_market_selector.currentText() == "Totals":
                outcomes = ["Under", "Over"]
            else:
                outcomes = ["Home AH", "Away AH"]
            global tw_profit_labels, tw_probability_labels, tw_ev_labels, tw_total_ev_label
            tw_profit_labels = {}
            tw_probability_labels = {}
            tw_ev_labels = {}
            for i, outcome in enumerate(outcomes):
                row = i + 1
                outcome_lbl = QLabel(outcome)
                results_grid.addWidget(outcome_lbl, row, 0)
                profit_lbl = QLabel("$0.00")
                profit_lbl.setStyleSheet("font-weight: bold;")
                results_grid.addWidget(profit_lbl, row, 1)
                tw_profit_labels[outcome] = profit_lbl
                prob_lbl = QLabel("0.00%")
                prob_lbl.setStyleSheet("font-weight: bold;")
                results_grid.addWidget(prob_lbl, row, 2)
                tw_probability_labels[outcome] = prob_lbl
                ev_lbl = QLabel("$0.00")
                ev_lbl.setStyleSheet("font-weight: bold;")
                results_grid.addWidget(ev_lbl, row, 3)
                tw_ev_labels[outcome] = ev_lbl
            total_row = len(outcomes) + 1
            total_ev_text = QLabel("Total EV:")
            total_ev_text.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(total_ev_text, total_row, 0, 1, 3)
            tw_total_ev_label = QLabel("$0.00")
            tw_total_ev_label.setStyleSheet("font-weight: bold;")
            results_grid.addWidget(tw_total_ev_label, total_row, 3)

        update_tw_results_grid()

        bets_container = QVBoxLayout()
        main_horizontal.addLayout(bets_container, stretch=3)

        tw_bets_grid = QGridLayout()
        bets_container.addLayout(tw_bets_grid)

        bets_headers = ["Bet Selection", "Bet Type", "Odds", "Amount", "Action"]
        for col, header in enumerate(bets_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            tw_bets_grid.addWidget(label, 0, col)

        btn_layout_ou = QHBoxLayout()
        add_btn_ou = QPushButton("Add Bet")
        add_btn_ou.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        btn_layout_ou.addWidget(add_btn_ou)
        update_btn_ou = QPushButton("🔄")
        update_btn_ou.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        btn_layout_ou.addWidget(update_btn_ou)
        btn_layout_ou.addStretch()
        tw_layout.addLayout(btn_layout_ou)

        ou_bet_widgets = []

        def add_ou_bet_row():
            row = 1 + len(ou_bet_widgets)
            combo = QComboBox()
            if tw_market_selector.currentText() == "Totals":
                combo.addItems(["Over", "Under"])
            else:
                combo.addItems(["Home AH", "Away AH"])
            tw_bets_grid.addWidget(combo, row, 0)
            
            type_combo = QComboBox()
            type_combo.addItems(["Back", "Lay"])
            tw_bets_grid.addWidget(type_combo, row, 1)
            
            odds_entry = QLineEdit()
            tw_bets_grid.addWidget(odds_entry, row, 2)
            
            amount_entry = QLineEdit()
            tw_bets_grid.addWidget(amount_entry, row, 3)
            
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("background-color: red; color: white;")
            delete_btn.clicked.connect(lambda _, r=row: delete_ou_bet_row(r))
            tw_bets_grid.addWidget(delete_btn, row, 4)
            
            ou_bet_widgets.append({
                "row": row,
                "combo": combo,
                "type_combo": type_combo,
                "odds_entry": odds_entry,
                "amount_entry": amount_entry,
                "delete_btn": delete_btn
            })

        def delete_ou_bet_row(row):
            for bet in ou_bet_widgets:
                if bet["row"] == row:
                    for widget in [bet["combo"], bet["type_combo"], bet["odds_entry"], bet["amount_entry"], bet["delete_btn"]]:
                        widget.deleteLater()
                    ou_bet_widgets.remove(bet)
                    break

        def update_tw_trading():
            market_type = tw_market_selector.currentText()
            matched_bets = []
            for bet in ou_bet_widgets:
                selection = bet["combo"].currentText()
                bet_type = bet["type_combo"].currentText()
                try:
                    odds = float(bet["odds_entry"].text())
                    amount = float(bet["amount_entry"].text())
                except ValueError:
                    continue
                matched_bets.append({
                    "Selection": selection,
                    "Type": bet_type,
                    "Odds": odds,
                    "Amount": amount
                })
            trading_data = core.TWTrade(matched_bets)
            total_ev = 0.0
            if market_type == "Totals":
                try:
                    under_odds = float(under_odds_label.text())
                    p_under = (1 / under_odds) * 100 if under_odds != 0 else 0.0
                except ValueError:
                    p_under = 0.0
                p_over = 100 - p_under
                outcomes = ["Over", "Under"]
                percentages = {"Over": p_over, "Under": p_under}
            else:
                try:
                    home_ah_odds = float(home_asian_odds_label.text())
                    p_home_ah = (1 / home_ah_odds) * 100 if home_ah_odds != 0 else 0.0
                except ValueError:
                    p_home_ah = 0.0
                p_away_ah = 100 - p_home_ah
                outcomes = ["Home AH", "Away AH"]
                percentages = {"Home AH": p_home_ah, "Away AH": p_away_ah}
            for outcome in outcomes:
                pl = trading_data.selections_pl.get(outcome, 0)
                color = "#4df9a2" if pl >= 0 else "#ce1d2e"
                tw_profit_labels[outcome].setText(f"${pl:.2f}")
                tw_profit_labels[outcome].setStyleSheet("font-weight: bold; color:" + color + ";")
                probability = percentages[outcome]
                tw_probability_labels[outcome].setText(f"{probability:.2f}%")
                ev = pl * (probability / 100)
                tw_ev_labels[outcome].setText(f"${ev:.2f}")
                total_ev += ev
            tw_total_ev_label.setText(f"${total_ev:.2f}")

        def update_tw_bet_rows():
            for bet in ou_bet_widgets:
                combo = bet["combo"]
                current = combo.currentText()
                combo.clear()
                if tw_market_selector.currentText() == "Totals":
                    combo.addItems(["Over", "Under"])
                    if current in ["Over", "Under"]:
                        combo.setCurrentText(current)
                else:
                    combo.addItems(["Home AH", "Away AH"])
                    if current in ["Home AH", "Away AH"]:
                        combo.setCurrentText(current)

        tw_market_selector.currentIndexChanged.connect(update_tw_results_grid)
        tw_market_selector.currentIndexChanged.connect(update_tw_bet_rows)

        add_btn_ou.clicked.connect(add_ou_bet_row)
        update_btn_ou.clicked.connect(update_tw_trading)
        
        # SCORE TAB
        score_layout = QVBoxLayout(score_tab)
        score_tab.setLayout(score_layout)

        total_layout = QHBoxLayout()
        total_label = QLabel("Total Bet:")
        total_layout.addWidget(total_label)
        score_total_bet_entry = QLineEdit()
        total_layout.addWidget(score_total_bet_entry)
        total_layout.addStretch()
        score_layout.addLayout(total_layout)

        main_horizontal_score = QHBoxLayout()
        score_layout.addLayout(main_horizontal_score)

        results_container_score = QVBoxLayout()
        main_horizontal_score.addLayout(results_container_score, stretch=1)
        score_results_grid = QGridLayout()
        results_container_score.addLayout(score_results_grid)
        score_results_headers = ["Selection", "Profit/Loss", "%", "EV"]
        for col, header in enumerate(score_results_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            score_results_grid.addWidget(label, 0, col)
        score_outcomes = [
            "0-0", "0-1", "0-2", "0-3",
            "1-0", "1-1", "1-2", "1-3",
            "2-0", "2-1", "2-2", "2-3",
            "3-0", "3-1", "3-2", "3-3",
            "Any Other Home Win", "Any Other Away Win", "Any Other Draw"
        ]
        score_profit_labels = {}
        score_probability_labels = {}
        score_ev_labels = {}
        for i, outcome in enumerate(score_outcomes):
            row = i + 1
            lbl_outcome = QLabel(outcome)
            score_results_grid.addWidget(lbl_outcome, row, 0)

            profit_lbl = QLabel("$0.00")
            score_results_grid.addWidget(profit_lbl, row, 1)
            score_profit_labels[outcome] = profit_lbl

            probability_lbl = QLabel("0.00%")
            probability_lbl.setStyleSheet("font-weight: bold;")
            score_results_grid.addWidget(probability_lbl, row, 2)
            score_probability_labels[outcome] = probability_lbl

            ev_lbl = QLabel("$0.00")
            ev_lbl.setStyleSheet("font-weight: bold;")
            score_results_grid.addWidget(ev_lbl, row, 3)
            score_ev_labels[outcome] = ev_lbl

        total_row = len(score_outcomes) + 1
        total_ev_text = QLabel("Total EV:")
        total_ev_text.setStyleSheet("font-weight: bold;")
        score_results_grid.addWidget(total_ev_text, total_row, 0, 1, 3)
        score_total_ev_label = QLabel("$0.00")
        score_total_ev_label.setStyleSheet("font-weight: bold;")
        score_results_grid.addWidget(score_total_ev_label, total_row, 3)

        bets_container_score = QVBoxLayout()
        main_horizontal_score.addLayout(bets_container_score, stretch=3)
        score_bets_grid = QGridLayout()
        bets_container_score.addLayout(score_bets_grid)
        bets_headers_score = ["Bet Selection", "Bet Type", "Odds", "Amount", "Action"]
        for col, header in enumerate(bets_headers_score):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            score_bets_grid.addWidget(label, 0, col)

        score_bet_widgets = []

        btn_layout_score = QHBoxLayout()
        add_btn_score = QPushButton("Add Bet")
        add_btn_score.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        btn_layout_score.addWidget(add_btn_score)
        update_btn_score = QPushButton("🔄")
        update_btn_score.setStyleSheet("background-color: #138585; color: white; padding: 5px;")
        btn_layout_score.addWidget(update_btn_score)
        btn_layout_score.addStretch()
        score_layout.addLayout(btn_layout_score)
        
        def add_score_bet_row():
            row = 1 + len(score_bet_widgets)
            combo = QComboBox()
            combo.addItems(score_outcomes)
            score_bets_grid.addWidget(combo, row, 0)
            type_combo = QComboBox()
            type_combo.addItems(["Back", "Lay"])
            score_bets_grid.addWidget(type_combo, row, 1)
            odds_entry = QLineEdit()
            score_bets_grid.addWidget(odds_entry, row, 2)
            amount_entry = QLineEdit()
            score_bets_grid.addWidget(amount_entry, row, 3)
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("background-color: red; color: white;")
            delete_btn.clicked.connect(lambda _, r=row: delete_score_bet_row(r))
            score_bets_grid.addWidget(delete_btn, row, 4)
            score_bet_widgets.append({
                "row": row,
                "combo": combo,
                "type_combo": type_combo,
                "odds_entry": odds_entry,
                "amount_entry": amount_entry,
                "delete_btn": delete_btn
            })
        
        def delete_score_bet_row(row):
            for bet in score_bet_widgets:
                if bet["row"] == row:
                    for widget in [bet["combo"], bet["type_combo"], bet["odds_entry"], bet["amount_entry"], bet["delete_btn"]]:
                        widget.deleteLater()
                    score_bet_widgets.remove(bet)
                    break
        
        def update_score_trading():
            outcome_key_map = {
                "Any Other Home Win": "Home Win 4+",
                "Any Other Away Win": "Away Win 4+",
                "Any Other Draw": "Draw +4"
            }
            matched_bets = []
            for bet in score_bet_widgets:
                raw_selection = bet["combo"].currentText()
                selection = outcome_key_map.get(raw_selection, raw_selection)
                bet_type = bet["type_combo"].currentText()
                try:
                    odds = float(bet["odds_entry"].text())
                except ValueError:
                    continue
                try:
                    amount = float(bet["amount_entry"].text()) if bet["amount_entry"].text() else 0
                except ValueError:
                    amount = 0
                matched_bets.append({
                    "Selection": selection,
                    "Type": bet_type,
                    "Odds": odds,
                    "Amount": amount
                })
            try:
                total_bet = float(score_total_bet_entry.text())
            except ValueError:
                total_bet = 0
        
            trading_data = core.ScoreTrade(matched_bets)
        
            if total_bet > 0:
                dutching_stakes = trading_data.dutching(total_bet, {bet["Selection"]: bet["Odds"] for bet in matched_bets})
                for bet in score_bet_widgets:
                    sel = bet["combo"].currentText()
                    if sel in dutching_stakes:
                        bet["amount_entry"].setText(f"{dutching_stakes[sel]:.2f}")
        
            total_ev = 0.0
            for outcome in score_profit_labels.keys():
                trading_key = outcome_key_map.get(outcome, outcome)
                pl = trading_data.selections_pl.get(trading_key, 0)
                color = "#4df9a2" if pl >= 0 else "#ce1d2e"
                score_profit_labels[outcome].setText(f"${pl:.2f}")
                score_profit_labels[outcome].setStyleSheet("color:" + color + ";")

                odds_float = score_odds_dict.get(trading_key, 0) if outcome in outcome_key_map else score_odds_dict.get(outcome, 0)

                percentage = (100 / odds_float) if odds_float != 0 else 0.0
                score_probability_labels[outcome].setText(f"{percentage:.2f}%")
                ev = pl * (percentage / 100)
                score_ev_labels[outcome].setText(f"${ev:.2f}")
                total_ev += ev
            score_total_ev_label.setText(f"${total_ev:.2f}")
        
        add_btn_score.clicked.connect(add_score_bet_row)
        update_btn_score.clicked.connect(update_score_trading)

        # -------------------
        tabs.addTab(build_game_tab, "Build Game")
        tabs.addTab(odds_tab, "Odds")
        tabs.addTab(simulator_tab, "Simulator")

        tabs.setStyleSheet("""
        QTabBar::tab {
        background: #1a1a1a;
        color: white;
        padding: 10px;
        }
        QTabBar::tab:selected {
        background: #138585;
        }
        """)

        self.open_windows.append(match_window)
        match_window.show()

# ----------------------- UPDATE SECTION  -----------------------
    def setup_right_section(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setStyleSheet("""
            QDateEdit {
                color: white;
                qproperty-alignment: 'AlignCenter';
                background-color: #1a1a1a;
                border: 1px solid #444;
                padding: 5px;
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #444;
            }
            QDateEdit QAbstractItemView {
                background-color: #1a1a1a;
                color: white;
                selection-background-color: #7289da;
            }
            QCalendarWidget QToolButton {
                background-color: #1a1a1a;
                color: white;
                font-weight: bold;
                border: none;
            }
            QCalendarWidget QSpinBox {
                color: white;
            }
            QCalendarWidget QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QCalendarWidget QAbstractItemView:enabled {
                background-color: #1a1a1a;
                color: white;
                selection-background-color: #7289da;
            }
        """)
        layout.addWidget(self.date_edit)
        
        self.league_scroll = QScrollArea()
        self.league_scroll.setWidgetResizable(True)
        self.league_scroll.setStyleSheet("border: none;")
        layout.addWidget(self.league_scroll)

        self.league_container = QWidget()
        league_layout = QVBoxLayout(self.league_container)
        league_layout.setAlignment(Qt.AlignTop)
        league_layout.setSpacing(10)
        league_layout.setContentsMargins(5, 5, 5, 5)
        self.league_scroll.setWidget(self.league_container)
        
        self.load_leagues()

        compare_evs_button = QPushButton("Compare Profit")
        compare_evs_button.setStyleSheet("background-color: #138585; color: white; padding: 10px;")
        compare_evs_button.clicked.connect(self.open_compare_profit)
        layout.addWidget(compare_evs_button)

        trade_amount_button = QPushButton("TradeAmount")
        trade_amount_button.setStyleSheet("background-color: #138585; color: white; padding: 10px;")
        trade_amount_button.clicked.connect(self.open_trade_amount)
        layout.addWidget(trade_amount_button)

        self.setup_queue_section(layout)

    def load_leagues(self):
        leagues_query = "SELECT * FROM league_data"
        leagues_df = self.vpfm_db.select(leagues_query)
        
        while self.league_container.layout().count():
            child = self.league_container.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["League", "Last Updated"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setRowCount(len(leagues_df))
        table.setStyleSheet(
            "QTableWidget { background-color: #1a1a1a; color: white; }"
            "QHeaderView::section { background-color: #1a1a1a; color: white; }"
        )

        for row_idx, row in leagues_df.iterrows():
            league_id = row["league_id"]

            league_btn = QPushButton(row["league_name"])
            league_btn.setStyleSheet(
                "QPushButton { background-color: transparent; color: white; font-size: 14px; text-align: left; border: none; }"
                "QPushButton:hover { color: #138585; }"
            )
            league_btn.clicked.connect(
                lambda _, lid=league_id, ln=row["league_name"]: self.open_league_window(lid, ln)
            )
            table.setCellWidget(row_idx, 0, league_btn)

            date_edit = QDateEdit()
            date_edit.setCalendarPopup(True)
            ld = row["last_updated_date"]
            date_edit.setDate(QDate(ld.year, ld.month, ld.day))
            date_edit.setStyleSheet(
                "QDateEdit { color: white; background-color:#1a1a1a; qproperty-alignment: 'AlignCenter'; border:1px solid #444; padding:5px; }"
                "QDateEdit::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width:20px; border-left:1px solid #444; }"
            )
            date_edit.dateChanged.connect(
                lambda qd, lid=league_id: self.update_last_updated_date(lid, qd)
            )
            table.setCellWidget(row_idx, 1, date_edit)

        self.league_container.layout().addWidget(table)

        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)

        update_btn = QPushButton("Extract & Process")
        update_btn.setStyleSheet(
            "QPushButton { background-color:#138585; color:white; font-size:14px; padding:10px; border-radius:5px; }"
            "QPushButton:hover { background-color:#1a1a1a; }"
        )

        schedule_btn = QPushButton("Update Schedule")
        schedule_btn.setStyleSheet(
            "QPushButton { background-color:#138585; color:white; font-size:14px; padding:10px; border-radius:5px; }"
            "QPushButton:hover { background-color:#1a1a1a; }"
        )

        add_btn = QPushButton("+")
        add_btn.setFixedSize(40, 40)
        add_btn.setStyleSheet(
            "QPushButton { background-color:#138585; color:white; font-size:20px; border-radius:20px; }"
            "QPushButton:hover { background-color:#1a1a1a; }"
        )

        btn_layout.addWidget(update_btn)
        btn_layout.addWidget(schedule_btn)
        btn_layout.addWidget(add_btn)
        self.league_container.layout().addWidget(btn_container)
        self.league_container.layout().addStretch()

        add_btn.clicked.connect(self.open_new_league_window)

        def run_extract_and_process():
            upto_date = datetime.strptime(self.date_edit.date().toString('yyyy-MM-dd'), '%Y-%m-%d').date()
            list_item = self.add_task_to_queue(f"Extract & Process up to {upto_date}")

            def task():
                core.Extract_Data(upto_date)
                core.Process_Data(upto_date)

            worker = UpdateWorker(task)
            worker.signals.finished.connect(lambda li=list_item: (self.remove_task_from_queue(li), self.load_leagues()))
            worker.signals.error.connect(lambda error_info: print(f"Error extracting/processing data: {error_info}"))
            self.threadpool.start(worker)

        def run_update_schedule():
            upto_date = datetime.strptime(self.date_edit.date().toString('yyyy-MM-dd'), '%Y-%m-%d').date()
            list_item = self.add_task_to_queue(f"Update Schedule")

            def task():
                schedule_updater = core.UpdateSchedule(upto_date)
                schedule_updater.update_all_leagues()

            worker = UpdateWorker(task)
            worker.signals.finished.connect(lambda li=list_item: (self.remove_task_from_queue(li), self.load_fixtures()))
            worker.signals.error.connect(lambda error_info: print(f"Error updating schedule: {error_info}"))
            self.threadpool.start(worker)

        update_btn.clicked.connect(run_extract_and_process)
        schedule_btn.clicked.connect(run_update_schedule)

    def update_last_updated_date(self, league_id, qdate):
        date_str = qdate.toString("yyyy-MM-dd")
        self.vpfm_db.execute(
            "UPDATE league_data SET last_updated_date = %s WHERE league_id = %s",
            (date_str, league_id)
        )

    def update_url_text(self, league_id, field_name, widget):
        self.vpfm_db.execute(
            f"UPDATE league_data SET {field_name} = %s WHERE league_id = %s",
            (widget.text(), league_id)
        )

    def update_active(self, league_id, is_active):
        self.vpfm_db.execute(
            "UPDATE league_data SET is_active = %s WHERE league_id = %s",
            (is_active, league_id)
        )

    def open_league_window(self, league_id, ln):
        ld_row = self.vpfm_db.select(
            "SELECT is_active, ss_url, fbref_fixtures_url FROM league_data WHERE league_id = %s",
            (league_id,)
        ).iloc[0]

        league_dialog = QDialog(self)
        league_dialog.setWindowTitle(ln)
        league_dialog.setWindowFlags(league_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        league_dialog.setStyleSheet("background-color: #1a1a1a; color: white;")
        layout = QVBoxLayout(league_dialog)

        active_chk = QCheckBox("Active")
        active_chk.setChecked(bool(ld_row["is_active"]))
        active_chk.stateChanged.connect(
            lambda state: self.update_active(league_id, state == Qt.Checked)
        )
        layout.addWidget(active_chk)

        sg_edit = QLineEdit(ld_row["ss_url"] or "")
        sg_edit.setPlaceholderText("Statsbomb / SG url")
        sg_edit.setStyleSheet("background-color:#1a1a1a; color:white;")
        sg_edit.editingFinished.connect(
            partial(self.update_url_text, league_id, "ss_url", sg_edit)
        )
        layout.addWidget(sg_edit)

        fb_edit = QLineEdit(ld_row["fbref_fixtures_url"] or "")
        fb_edit.setPlaceholderText("FBRef fixtures url")
        fb_edit.setStyleSheet("background-color:#1a1a1a; color:white;")
        fb_edit.editingFinished.connect(
            partial(self.update_url_text, league_id, "fbref_fixtures_url", fb_edit)
        )
        layout.addWidget(fb_edit)

        update_teams_btn = QPushButton("Update teams")
        layout.addWidget(update_teams_btn)

        def run_fill_teams():
            list_item = self.add_task_to_queue(f"{ln}: Update teams")
            worker = UpdateWorker(core.Fill_Teams_Data, league_id)
            worker.signals.finished.connect(lambda li=list_item: self.remove_task_from_queue(li))
            worker.signals.error.connect(lambda error_info: print(f"Error updating teams: {error_info}"))
            self.threadpool.start(worker)
            league_dialog.accept()

        update_teams_btn.clicked.connect(run_fill_teams)

        league_dialog.exec_()

    def open_new_league_window(self):
        league_dialog = QDialog(self)
        league_dialog.setWindowTitle("Add new league")
        league_dialog.setWindowFlags(league_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        league_dialog.setStyleSheet("background-color: #1a1a1a; color: white;")
        layout = QVBoxLayout(league_dialog)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("League name")
        name_edit.setStyleSheet("background-color:#1a1a1a; color:white;")
        layout.addWidget(name_edit)

        active_chk = QCheckBox("Active")
        active_chk.setChecked(True)
        layout.addWidget(active_chk)

        sg_edit = QLineEdit()
        sg_edit.setPlaceholderText("Statsbomb / SG url")
        sg_edit.setStyleSheet("background-color:#1a1a1a; color:white;")
        layout.addWidget(sg_edit)

        fb_edit = QLineEdit()
        fb_edit.setPlaceholderText("FBRef fixtures url")
        fb_edit.setStyleSheet("background-color:#1a1a1a; color:white;")
        layout.addWidget(fb_edit)

        create_btn = QPushButton("Create")
        layout.addWidget(create_btn)

        def create_league():
            ln = name_edit.text().strip()
            if not ln:
                return
            self.vpfm_db.execute(
                "INSERT INTO league_data (league_name, is_active, ss_url, fbref_fixtures_url, last_updated_date) "
                "VALUES (%s, %s, %s, %s, NOW())",
                (ln, active_chk.isChecked(), sg_edit.text().strip(), fb_edit.text().strip())
            )
            league_id = self.vpfm_db.select(
                "SELECT league_id FROM league_data WHERE league_name = %s ORDER BY league_id DESC LIMIT 1",
                (ln,)
            ).iloc[0]["league_id"]

            list_item = self.add_task_to_queue(f"{ln}: Create league / Update teams")
            worker = UpdateWorker(core.Fill_Teams_Data, league_id)
            worker.signals.finished.connect(lambda li=list_item: self.remove_task_from_queue(li))
            worker.signals.error.connect(lambda error_info: print(f"Error creating league: {error_info}"))
            self.threadpool.start(worker)

            self.load_leagues()
            league_dialog.accept()

        create_btn.clicked.connect(create_league)

        league_dialog.exec_()

# ----------------------- QUEUE SECTION  -----------------------
    def setup_queue_section(self, parent_layout): 
        self.update_queue_list = QListWidget()
        self.update_queue_list.setStyleSheet("color: white; background-color: #333;")
        self.update_queue_list.setMaximumHeight(50)
        parent_layout.addWidget(self.update_queue_list)

    def add_task_to_queue(self, task_description):
        item = QListWidgetItem(task_description)
        self.update_queue_list.addItem(item)
        return item

    def remove_task_from_queue(self, list_item):
        row = self.update_queue_list.row(list_item)
        self.update_queue_list.takeItem(row)

# -------------------- Open Buttons --------------------
    def open_compare_profit(self):
        ce_dialog = QDialog()
        ce_dialog.setWindowTitle("Compare Profit")
        ce_dialog.setStyleSheet("background-color: #1a1a1a; color: white;")
        ce_dialog.setWindowFlags(ce_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        back_label = QLabel("Back:")
        back_spin = QDoubleSpinBox()
        back_spin.setDecimals(2)
        back_spin.setRange(1.01, 1000)
        back_spin.setSingleStep(0.01)
        back_spin.setValue(2.02)
        
        lay_label = QLabel("Lay:")
        lay_spin = QDoubleSpinBox()
        lay_spin.setDecimals(2)
        lay_spin.setRange(1.01, 1000)
        lay_spin.setSingleStep(0.01)
        lay_spin.setValue(1.98)
        
        result_label = QLabel("Enter odds")
  
        main_layout = QVBoxLayout(ce_dialog)
        
        back_layout = QHBoxLayout()
        back_layout.addWidget(back_label)
        back_layout.addWidget(back_spin)

        lay_layout = QHBoxLayout()
        lay_layout.addWidget(lay_label)
        lay_layout.addWidget(lay_spin)
        
        main_layout.addLayout(back_layout)
        main_layout.addLayout(lay_layout)
        main_layout.addWidget(result_label)
        
        def on_calculate():
            back_odds = back_spin.value()
            lay_odds = lay_spin.value()
            
            try:
                back_risk = 100 / (back_odds - 1)
            except ZeroDivisionError:
                back_risk = float('inf')
            
            lay_risk = (lay_odds - 1) * 100
            
            if abs(back_risk - lay_risk) < 1e-6:
                result_text = f"Both options are equivalent in risk. Risk: {back_risk:.2f}"
            elif back_risk < lay_risk:
                result_text = f"Backing is more profitable.\nBack Risk: {back_risk:.2f} vs Lay Risk: {lay_risk:.2f}"
            else:
                result_text = f"Laying is more profitable.\nLay Risk: {lay_risk:.2f} vs Back Risk: {back_risk:.2f}"
            
            result_label.setText(result_text)
        
        back_spin.valueChanged.connect(on_calculate)
        lay_spin.valueChanged.connect(on_calculate)

        self.ce_dialog = ce_dialog
        ce_dialog.show()

    def open_trade_amount(self):
        ta_dialog = QDialog()
        ta_dialog.setWindowTitle("Trade Amount Calculation")
        ta_dialog.setStyleSheet("background-color: #1a1a1a; color: white;")
        ta_dialog.setWindowFlags(ta_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(ta_dialog)

        betTypeLayout = QHBoxLayout()
        betTypeLabel = QLabel("Bet Type:")
        betTypeCombo = QComboBox()
        betTypeCombo.addItems(["Backing", "Laying"])
        betTypeLayout.addWidget(betTypeLabel)
        betTypeLayout.addWidget(betTypeCombo)
        layout.addLayout(betTypeLayout)

        myOddsLayout = QHBoxLayout()              
        myOddsLabel = QLabel("My Odds:")
        myOddsSpinBox = QDoubleSpinBox() 
        myOddsSpinBox.setDecimals(3) 
        myOddsSpinBox.setRange(1.01, 1000)
        myOddsSpinBox.setSingleStep(0.001)
        myOddsSpinBox.setValue(2.000)
        myOddsLayout.addWidget(myOddsLabel)
        myOddsLayout.addWidget(myOddsSpinBox)
        layout.addLayout(myOddsLayout)

        exchangeOddsLayout = QHBoxLayout()
        exchangeOddsLabel = QLabel("Exchange Odds:")
        exchangeOddsSpinBox = QDoubleSpinBox()
        exchangeOddsSpinBox.setDecimals(2)
        exchangeOddsSpinBox.setRange(1.01, 1000)
        exchangeOddsSpinBox.setSingleStep(0.01)
        exchangeOddsSpinBox.setValue(2.500)
        exchangeOddsLayout.addWidget(exchangeOddsLabel)
        exchangeOddsLayout.addWidget(exchangeOddsSpinBox)
        layout.addLayout(exchangeOddsLayout)

        bankrollLayout = QHBoxLayout()
        bankrollLabel = QLabel("Bankroll:")
        bankrollSpinBox = QDoubleSpinBox()
        bankrollSpinBox.setDecimals(2)
        bankrollSpinBox.setRange(0, 10000000)
        bankrollSpinBox.setSingleStep(1)
        bankrollSpinBox.setValue(1500.00)
        bankrollLayout.addWidget(bankrollLabel)
        bankrollLayout.addWidget(bankrollSpinBox)
        layout.addLayout(bankrollLayout)

        resultLabel = QLabel("")
        layout.addWidget(resultLabel)

        def calculate_trade():
            bet_type = betTypeCombo.currentText()
            my_odds = myOddsSpinBox.value()
            exchange_odds = exchangeOddsSpinBox.value()
            bankroll = bankrollSpinBox.value()         

            if bet_type == "Backing":
                p = 1 / my_odds
                edge = exchange_odds * p - 1
                kelly_fraction = max(edge / (exchange_odds - 1), 0)
                trade_amount = bankroll * (kelly_fraction / 3)
                resultLabel.setText(f"Bet Type: Backing\n"
                                    f"Edge: {edge*100:.2f}%\n"
                                    f"Trade Amount: ${trade_amount:.2f}")
            else:
                p = 1 / my_odds
                edge = 1 - exchange_odds * p
                kelly_fraction = max(edge / (exchange_odds - 1), 0)
                stake = bankroll * (kelly_fraction / 3)
                liability = stake * (exchange_odds - 1)
                resultLabel.setText(f"Bet Type: Laying\n"
                                    f"Edge: {edge*100:.2f}%\n"
                                    f"Trade Stake: ${stake:.2f}\n"
                                    f"Liability: ${liability:.2f}")

        myOddsSpinBox.valueChanged.connect(calculate_trade)
        exchangeOddsSpinBox.valueChanged.connect(calculate_trade)
        bankrollSpinBox.valueChanged.connect(calculate_trade)
        betTypeCombo.currentIndexChanged.connect(calculate_trade)

        self.ta_dialog = ta_dialog
        ta_dialog.show()

    def open_arbitrage(self):
        print("Arbitrage opened.")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())