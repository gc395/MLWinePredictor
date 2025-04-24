import sys
import subprocess
import os

def validate_args(args):
    if len(args) != 3:
        print("Usage: spark-submit run_model.py <training_data.csv> <validation_data.csv>")
        sys.exit(1)

def run_and_capture(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout, result.returncode

def main():
    validate_args(sys.argv)

    raw_train = sys.argv[1]
    raw_val = sys.argv[2]

    cleaned_train = "cleaned_train.csv"
    cleaned_val = "cleaned_validation.csv"

    if not os.path.isfile(raw_train):
        print(f"Error: Training file '{raw_train}' not found.")
        sys.exit(1)
    if not os.path.isfile(raw_val):
        print(f"Error: Validation file '{raw_val}' not found.")
        sys.exit(1)

    logs = ""

    # Clean datasets
    out, code = run_and_capture(["spark-submit", "debug_columns.py", raw_train, cleaned_train])
    logs += "=== Cleaning Training Dataset ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Failed to clean training dataset.\n"

    out, code = run_and_capture(["spark-submit", "debug_columns.py", raw_val, cleaned_val])
    logs += "=== Cleaning Validation Dataset ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Failed to clean validation dataset.\n"

    # Train
    out, code = run_and_capture(["spark-submit", "train_model.py", cleaned_train, cleaned_val])
    logs += "=== Training Output ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Training failed.\n"

    # Predict
    out, code = run_and_capture(["spark-submit", "predict_model.py", cleaned_val, "trained_model"])
    logs += "=== Prediction Output ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Prediction failed.\n"

    # Save logs
    with open("output.txt", "w") as f:
        f.write(logs)

    print("✔️ Run complete. See output.txt for full logs.")

if __name__ == "__main__":
    main()
