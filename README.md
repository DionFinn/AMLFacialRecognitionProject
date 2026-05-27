## Project: Facial Recognition with Emotion and Liveness
Team Member Names and Student ID:
- Rothpitou Poeung - 105217834
- Dion Finnerty - 103545669
- Kushagra Suryawanshi - 104809447
- Jason Tjahjono - 104656753
- Sovithyea Prach - 105270743

## System Overview

Data source:
- [11-785-Fall-20-Homework-2: Part 2](https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/overview): for metric learning and supervised learning models training
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013): for emotion detection model training
- [](): for anti-spoofing training
```bash

```

## Instructions
1. Create environment and install all necessary packages
```bash
# On Windows
python python -m venv env
.env\Scripts\activate.bat # command prompt
.env\Scripts\Activate.ps1 # powershell
pip install -r requirements.txt

# On MacOS / Linux
python -m venv env
source env/bin activate
pip install -r requirements.txt

# Using Conda environment
conda activate [your-environment-name]
pip install -r requirements.txt
```
2. Create a models/ folder inside of model/, download all the artifacts from this [link](https://liveswinburneeduau-my.sharepoint.com/:f:/g/personal/105217834_student_swin_edu_au/IgBe0NPGz9cKT4OJi7HVW6rqAVwPOpWKT9YChlpGFOHw5gQ?e=dmEEX8) and place it in model/models

3. Run StreamLit GUI
```bash
cd gui/
streamlit run gui.py
```