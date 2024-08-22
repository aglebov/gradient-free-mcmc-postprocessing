#! /bin/bash

# Save the current working directory in an environment variable.
INITIAL_WORKING_DIRECTORY=$(pwd)

# This line changes to current working directory to where
# the analysis.sh file is.
cd "$(dirname "$0")"

pdftotext ./build/main.pdf - | egrep -e '\w+' | iconv -f ISO-8859-15 -t UTF-8 | wc -w

# Go back to where we were before changing into the
# scripts directory.
cd "$INITIAL_WORKING_DIRECTORY"

