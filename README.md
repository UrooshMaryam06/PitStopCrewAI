# 🏎️ Pit Stop Crew AI: F1 Performance & Strategy Predictor

**Course:** AI 201 - Programming for Artificial Intelligence  
**Institution:** GIK Institute of Engineering Sciences and Technology  
**Prepared by:** Pit Stop Crew (Uroosh Maryam)

---

## 📌 Project Overview
Pit Stop Crew AI is a comprehensive data analytics and machine learning system designed to predict Formula 1 race outcomes. This project bridges the gap between raw telemetry data and actionable race strategy by using **Object-Oriented Programming (OOP)** and **Gradient Boosting AI models**.

### 🎯 Key Objectives
- **Data Engineering:** Cleaned and merged 2022 F1 season datasets (Drivers, Teams, Races, Sprint results).
- **OOP Architecture:** Implemented a robust system using Inheritance and Polymorphism to model the F1 ecosystem.
- **AI Prediction:** Built a Multi-Class LightGBM Classifier to predict finishing positions and podium shares.

---

## 🧠 Technical Implementation

### 1. Object-Oriented Programming (OOP)
The system is built on a foundation of clean, reusable code using:
* **Inheritance:** A base `F1Entity` class serves as the parent for `Driver` and `Team` classes, ensuring shared attributes like names are managed efficiently.
* **Polymorphism:** Method overriding via `get_info()` allows specific entities to report data uniquely.
* **Encapsulation:** Driver stats, team lineups, and circuit telemetry are bundled into secure, managed objects.
* **Exception Handling:** Robust `try-except` blocks ensure the system handles missing data files without crashing.

### 2. Machine Learning Pipeline
* **Model:** LightGBM (Multi-Class Classifier).
* **Feature Engineering:** Created a "Team Form" feature using a **Rolling Average** of constructor points from the previous 5 races.
* **Evaluation:** Measured performance using Accuracy and Cohen’s Kappa to ensure predictions exceed random chance.

---

## 🛠️ Tools & Libraries
- **Language:** Python 3.x
- **Data:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure
- `code_sem_project.ipynb`: The main Python Notebook containing cleaning, OOP, and ML.
- `cleaned_f1_...csv`: Processed datasets used for training.
- `Formula_1_Performance_Analysis_Report.docx`: Detailed technical documentation.
- `Formula-1-Performance-Analysis-and-Race-Outcome-Prediction-System.pptx`: Presentation slides.

---

## 🏁 Results & Insights
The model successfully identifies podium contenders by analyzing the intersection of qualifying performance and constructor momentum. Results are visualized through trend lines and podium share pie charts, providing a "Race Engineer's view" of the 2022 season.
