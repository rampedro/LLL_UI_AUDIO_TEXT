#!/bin/bash

# Search for Python processes with a specific string in the CMD column
python_pid=$(ps aux | grep "/usr/local/Cellar/python@3.11/3.11.4_1/Frameworks/Pyth" | grep -v grep | awk '{print $2}')

# Check if the Python PID is found and terminate the process if it exists
if [ -n "$python_pid" ]; then
    echo "Found Python process with PID: $python_pid. Killing the process..."
    kill -9 $python_pid
    echo "Python process terminated."
else
    echo "No Python process found."
fi

