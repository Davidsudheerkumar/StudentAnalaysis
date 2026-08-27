# ============================================================
# PROMPT 6 - FASTAPI SETUP AND DATA LOADING
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_data():
    # Load the Maths dataset
    df = pd.read_csv("data/Maths.csv")

    # Create Result based on G3
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout"
        if x == 0
        else ("Fail" if 1 <= x <= 9 else "Pass")
    )

    # Convert G3 into percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average parent education level
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate grade trend
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count support services received
    support_columns = ["schoolsup", "famsup", "paid"]

    df["total_support"] = (
        df[support_columns]
        .apply(lambda row: (row == "yes").sum(), axis=1)
    )

    # Calculate academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return prepared DataFrame
    return df


# Load data when the application starts
df = load_data()


# ============================================================
# PROMPT 7 - GET ENDPOINTS
# ============================================================

# ------------------------------------------------------------
# GET /summary
# ------------------------------------------------------------

@app.get("/summary")
def get_summary():
    # Exclude dropout students from performance calculations
    non_dropout = df[df["G3"] != 0]

    # Calculate class average G3
    if len(non_dropout) > 0:
        class_average_g3 = round(
            float(np.mean(non_dropout["G3"])),
            2
        )
    else:
        class_average_g3 = 0.0

    # Calculate pass rate
    if len(non_dropout) > 0:
        pass_count = np.sum(non_dropout["G3"] >= 10)

        pass_rate = round(
            float((pass_count / len(non_dropout)) * 100),
            2
        )
    else:
        pass_rate = 0.0

    # Count at-risk students
    at_risk_count = np.sum(
        (df["G3"] >= 1) &
        (df["G3"] <= 9)
    )

    # Count dropout students
    dropout_count = np.sum(df["G3"] == 0)

    # Return summary as JSON
    return {
        "total_students": int(len(df)),
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate,
        "at_risk_count": int(at_risk_count),
        "dropout_count": int(dropout_count)
    }


# ------------------------------------------------------------
# GET /at-risk
# ------------------------------------------------------------

@app.get("/at-risk")
def get_at_risk():
    # Select students whose G3 is between 1 and 9
    at_risk = df[
        (df["G3"] >= 1) &
        (df["G3"] <= 9)
    ].copy()

    # Sort worst-performing students first
    at_risk = at_risk.sort_values(
        "G3",
        ascending=True
    )

    # Create response list
    result = []

    for index, row in at_risk.iterrows():
        result.append({
            "student_index": int(index),
            "G1": float(row["G1"]),
            "G2": float(row["G2"]),
            "G3": float(row["G3"]),
            "absences": int(row["absences"])
        })

    return result


# ------------------------------------------------------------
# GET /top-students
# ------------------------------------------------------------

@app.get("/top-students")
def get_top_students():
    # Exclude dropout students
    non_dropout = df[df["G3"] != 0].copy()

    # Sort students by G3 in descending order
    top_students = non_dropout.sort_values(
        "G3",
        ascending=False
    ).head(5)

    # Create response list
    result = []

    for index, row in top_students.iterrows():
        result.append({
            "student_index": int(index),
            "G1": float(row["G1"]),
            "G2": float(row["G2"]),
            "G3": float(row["G3"])
        })

    return result


# ============================================================
# PROMPT 8 - PYDANTIC MODEL AND POST ENDPOINT
# ============================================================

class StudentInput(BaseModel):
    # First internal assessment grade
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="G1 must be between 0 and 20"
    )

    # Second internal assessment grade
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="G2 must be between 0 and 20"
    )

    # Weekly study time category
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Studytime must be between 1 and 4"
    )

    # Number of absences
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Absences must be between 0 and 100"
    )

    # Number of previous failures
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Failures must be between 0 and 4"
    )


# ------------------------------------------------------------
# POST /predict-result
# ------------------------------------------------------------

@app.post("/predict-result")
def predict_result(student: StudentInput):
    # Calculate estimated final grade using the specified formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp estimated G3 between 0 and 20
    estimated_g3 = max(
        0,
        min(20, estimated_g3)
    )

    # Determine predicted result
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # Determine confidence level
    if (
        student.G1 > 12 and
        student.G2 > 12
    ):
        confidence = "High"

    elif (
        student.G1 < 8 and
        student.G2 < 8
    ):
        confidence = "High"

    else:
        confidence = "Medium"

    # Return prediction details
    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence
    }


# ============================================================
# PROMPT 9 - ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    # Return basic API information
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# ============================================================
# UVICORN RUNNER
# ============================================================

if __name__ == "__main__":
    # Start the FastAPI application using Uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )