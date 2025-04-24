import sys
import subprocess
import os

def validate_args(args):
    if len(args) != 3:
        print("Usage: spark-submit run_model.py <training_data.csv> <validation_data.csv>")
        sys.exit(1)

def main():
    validate_args(sys.argv)

    training_file = sys.argv[1]
    validation_file = sys.argv[2]

    if not os.path.exists(training_file):
        print(f"Error: Training file '{training_file}' not found.")
        sys.exit(1)
    if not os.path.exists(validation_file):
        print(f"Error: Validation file '{validation_file}' not found.")
        sys.exit(1)

    result = subprocess.run([
        "spark-submit", "train_model.py", training_file, validation_file
    ])

    if result.returncode != 0:
        print("Training failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
