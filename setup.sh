#!/bin/bash
streamlit run app.py --server.port $PORT --server.enableCORS false
python -m spacy download en_core_web_sm
