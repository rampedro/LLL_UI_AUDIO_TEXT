#!/bin/bash

# Run the curl command to get the token and capture the result
token_response=$(curl -s -X POST 'https://iam.cloud.ibm.com/identity/token' -H 'Content-Type: application/x-www-form-urlencoded' -d 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=')
# Check if the curl command was successful
if [ $? -eq 0 ]; then
    echo "Curl command successful. Response:"
    echo "$token_response"

    # Extract the access_token using jq
    access_token=$(echo "$token_response" | jq -r '.access_token')

    # Check if access_token is not empty
    if [ -n "$access_token" ]; then
        # Set the API_KEY environment variable with the access token
        export API_KEY="$access_token"
        echo "API_KEY environment variable set successfully."
        echo $API_KEY
    else
        echo "Failed to extract the access token from the response: access_token is empty or not found."
    fi
else
    echo "Failed to execute the curl command to get the token: curl command failed."
fi

