#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <statistic|machine_learning|deep_learning> <regression|classification>"
  exit 1
fi

# Check if an argument is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <regression|classification>"
  exit 1
fi

# Convert input to lowercase for flexibility
arg1=$(echo "$1" | tr '[:upper:]' '[:lower:]')
arg2=$(echo "$2" | tr '[:upper:]' '[:lower:]')

# Moving to the imputation folder because all paths within the code files are mapped from there (will be handeled later)
cd ./imputation/

# Match argument to executables
if [ "$arg1" = "statistic" ]&& ["$arg2" = "regression"] ; then
  echo "Launching statistical approche..."
  python statistical.py

elif [ "$arg1" = "machine_learning" ]&& ["$arg2" = "regression"] ; then
  echo "Launching machine learning regression approche..."
  python machine_learning_regression.py


elif [ "$arg1" = "machine_learning" ] || [ "$arg2" = "classification" ]; then
  echo "Launching machine learning classification approche..."
  python machine_learning_classification.py


elif [ "$arg1" = "deep_learning" ]&& ["$arg2" = "regression"] ; then
  echo "Launching deep learning regression approche..."
  python deep_learning_regression.py



elif [ "$arg1" = "deep_learning" ] || [ "$arg2" = "classification" ]; then
  echo "Launching deep learning classification approche..."
  python deep_learning_classification.py


else
  echo "Error: Unknown argument '$1' or '$2'"
  exit 1
fi
