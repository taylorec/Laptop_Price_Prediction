# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 08:14:03 2021

@author: evan
"""

  
mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml