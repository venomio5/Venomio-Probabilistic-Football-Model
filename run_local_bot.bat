@echo off
title Bot + Ngrok

start cmd /k python compute_service.py
timeout /t 3 > nul
start cmd /k ngrok http 8001