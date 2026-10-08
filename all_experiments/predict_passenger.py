"""Predict Titanic survival for one passenger with the model saved by Lab 3.

The notebook trained the model and saved two files in models/:
  titanic_survival.joblib      the fitted pipeline (preprocessing + classifier)
  titanic_survival_card.json   what the model expects and how well it tested

This script loads both and prints a prediction. No training code is needed.

Example:
    python predict_passenger.py --pclass 2 --sex female --age 30 --fare 20
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Predict survival for one passenger.")
    parser.add_argument("--pclass", type=int, required=True, choices=[1, 2, 3], help="ticket class")
    parser.add_argument("--sex", required=True, choices=["male", "female"])
    parser.add_argument("--age", type=float, required=True, help="age in years")
    parser.add_argument("--fare", type=float, required=True, help="ticket fare")
    parser.add_argument("--sibsp", type=int, default=0, help="siblings/spouses aboard")
    parser.add_argument("--parch", type=int, default=0, help="parents/children aboard")
    parser.add_argument("--embarked", default="S", choices=["C", "Q", "S"], help="port of embarkation")
    args = parser.parse_args()

    model = joblib.load(MODELS / "titanic_survival.joblib")
    card = json.loads((MODELS / "titanic_survival_card.json").read_text())

    # Build one row with exactly the columns the model was trained on.
    passenger = pd.DataFrame([{
        "Pclass": args.pclass, "Sex": args.sex, "Age": args.age, "SibSp": args.sibsp,
        "Parch": args.parch, "Fare": args.fare, "Embarked": args.embarked,
    }])[card["features"]]

    probability = float(model.predict_proba(passenger)[0, 1])
    decision = "survived" if probability >= card["decision_threshold"] else "did not survive"
    print(f"survival probability: {probability:.3f}  ->  {decision}")
    print(f"(model: {card['model']}, test ROC AUC {card['final_test_roc_auc']})")


if __name__ == "__main__":
    main()
