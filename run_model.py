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

    training_file = sys.argv[1]
    validation_file = sys.argv[2]

    if not os.path.isfile(training_file):
        print(f"Error: Training file '{training_file}' not found.")
        sys.exit(1)
    if not os.path.isfile(validation_file):
        print(f"Error: Validation file '{validation_file}' not found.")
        sys.exit(1)

    logs = ""

    out, code = run_and_capture(["spark-submit", "train_model.py", training_file, validation_file])
    logs += "=== Training Output ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Training failed.\n"

    out, code = run_and_capture(["spark-submit", "predict_model.py", validation_file, "trained_model"])
    logs += "=== Prediction Output ===\n" + out + "\n"
    if code != 0:
        logs += "[Error] Prediction failed.\n"

    with open("output.txt", "w") as f:
        f.write(logs)

    print("Run completed. Output saved to output.txt")

if __name__ == "__main__":
    main()
