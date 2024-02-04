from flask import Flask, request, render_template, redirect, url_for
from drug_named_entity_recognition import find_drugs
import json
import requests
from transformers import BertTokenizer
import nltk
nltk.download('punkt')  # Download the sentence tokenizer
from nltk import sent_tokenize



header = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer eyJraWQiOiIyMDI0MDEwNjA4MzciLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC01NTAwMDM1QUs1IiwiaWQiOiJJQk1pZC01NTAwMDM1QUs1IiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiOTJmMTExMGItZTBlYS00NDIwLWExNWItM2RjZjlhOWExNGVlIiwiaWRlbnRpZmllciI6IjU1MDAwMzVBSzUiLCJnaXZlbl9uYW1lIjoicGVkcmFtIiwiZmFtaWx5X25hbWUiOiJhaGFkaW5lamFkIiwibmFtZSI6InBlZHJhbSBhaGFkaW5lamFkIiwiZW1haWwiOiJwZWRyYW1AdGNlZ3JvdXAuY29tIiwic3ViIjoicGVkcmFtQHRjZWdyb3VwLmNvbSIsImF1dGhuIjp7InN1YiI6InBlZHJhbUB0Y2Vncm91cC5jb20iLCJpYW1faWQiOiJJQk1pZC01NTAwMDM1QUs1IiwibmFtZSI6InBlZHJhbSBhaGFkaW5lamFkIiwiZ2l2ZW5fbmFtZSI6InBlZHJhbSIsImZhbWlseV9uYW1lIjoiYWhhZGluZWphZCIsImVtYWlsIjoicGVkcmFtQHRjZWdyb3VwLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiJmMmE0N2JjM2ViYWQ0MWZhOTA3YWIzNzM2YWZmMDMzMiIsImltc191c2VyX2lkIjoiODk4MDEzMiIsImZyb3plbiI6dHJ1ZSwiaW1zIjoiMTczMzkxMSJ9LCJpYXQiOjE3MDcwNzk1NzAsImV4cCI6MTcwNzA4MzE3MCwiaXNzIjoiaHR0cHM6Ly9pYW0uY2xvdWQuaWJtLmNvbS9pZGVudGl0eSIsImdyYW50X3R5cGUiOiJ1cm46aWJtOnBhcmFtczpvYXV0aDpncmFudC10eXBlOmFwaWtleSIsInNjb3BlIjoiaWJtIG9wZW5pZCIsImNsaWVudF9pZCI6ImRlZmF1bHQiLCJhY3IiOjEsImFtciI6WyJwd2QiXX0.H1s9k7WNgRh-m1OR2vhe9DcaaGAZjVMJLFxHuB5ylN7_6jPrvBkU7IUtD01ylZV23pChXIfTx0WjA862iazjfcFeAzgACaVxfFbUWQyOuo6O92U9NhoUMdf8nDE3SIc8mdAjTyO4_Mz0LNdlsddGXnyJn0pcluFIgPAz9dG-T9ovS3DZLxERxyco0hgdKSGaRt7QBJhuTA4Nsykl-D6H3K3Qn_EZ7X5kpB2J8JcDQ_kFyVKimNm3q_ly1rVVpxNOz9_GCP893nyjjJBBrlUxe6N8lsf_c0EoO4rwCja70q7DJQkxwUc9n9sf7wL3Iz2RCmWzLxbSe_8u7b0v6144-w'}




### Best parameters are 
## portion 2000
## over lap 250 
## max token for api 900
app = Flask(__name__,template_folder='templates')

# Tokenize the text into sentences
def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Split text into smaller portions while keeping sentences intact
def split_text_with_equal_size(text, desired_portion_length=2000, overlap_tokens=250, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    sentences = split_text_into_sentences(text)
    total_tokens = tokenizer(text)['input_ids']

    # Check if the text is shorter than 900 tokens
    if len(total_tokens) <= 20:
        return [text]

    portions = []
    current_portion = []
    current_portion_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        if current_portion_length + sentence_length <= desired_portion_length:
            current_portion.extend(sentence_tokens)
            current_portion_length += sentence_length
        else:
            if current_portion:
                portion_text = tokenizer.convert_tokens_to_string(current_portion)
                portions.append(portion_text)

            current_portion = sentence_tokens
            current_portion_length = sentence_length

    if current_portion:
        portion_text = tokenizer.convert_tokens_to_string(current_portion)
        portions.append(portion_text)

    return portions

  
  
  



@app.route('/', methods=['GET', 'POST'])
def input_text():
    if request.method == 'POST':
        user_input = request.form['user_input']
        if not user_input:
            return render_template('error.html', error_message='Input text is required.')
        else:
            return redirect(url_for('generate_text', user_input=user_input))
    else:
        return render_template('input.html')



################## G generate_text T ################


@app.route('/generate_text', methods=['GET'])
def generate_text():
    try:
        user_input = request.args.get('user_input')

        if not user_input:
            return render_template('error.html', error_message='Input text is required.')

        if not user_input:
            return render_template_string('error.html')

        pieces = split_text_with_equal_size(user_input)
        
        url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
        headers = header
        
        generated_texts = []

        for piece in pieces:
            payload = {
                "model_id": "meta-llama/llama-2-70b-chat",
                "input": "instruction : Extract medical information from the conversation specific to entities mentioned in the text. Categorize the information into demographics, Chief Complaint, HPI, PMHx, Social History, Family History, ROS, state and types of Physical Exam and Test, Medications and dosages, Plans and Other mentioned high-value information, and corresponding billing and icd codes. Only include explicitly mentioned information and notes when needed. refrain from inferring or adding details not present in the text. input text : " + piece ,
                "parameters": {"decoding_method": "greedy", "max_new_tokens": 900, "min_new_tokens": 50, "stop_sequences": [], "repetition_penalty": 1},
                "project_id": "beaf6470-c5bc-4695-b204-29d09c8bf7fb",
                "moderations": {
                    "hap": {"input": True, "output": True, "threshold": 0.5, "mask": {"remove_entity_value": False}}
                }
            }

            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            generated_text = response_data['results'][0]['generated_text']
            generated_texts.append(generated_text)


      
        combined_result = "".join(generated_texts)






        #########################################
        ### use combined result as a prompt again
        #########################################


        # # Make the API call with the combined text
        # combined_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
        # combined_payload = {
        #     "model_id": "meta-llama/llama-2-70b-chat",
        #     "input": "instruction : Categorize the input_text into demographics,  Chief Complaint, HPI, PMHx, Social History, Family #History, ROS, state and types of Physical Exam and Test, Medications and dosages, Plans and Other mentioned high value information, and #corresponding billing and icd codes. Only include explicitly mentioned information and notes when needed. refrain from inferring or adding #details not present in the text. input_text : " + combined_result,
        #     "parameters": {"decoding_method": "greedy", "max_new_tokens": 1500, "min_new_tokens": 0, "stop_sequences": [], #"repetition_penalty": 1},
        #     "project_id": "beaf6470-c5bc-4695-b204-29d09c8bf7fb",
        #     "moderations": {
        #         "hap": {"input": True, "output": True, "threshold": 0.5, "mask": {"remove_entity_value": True}}
        #     }
        # }
        # 
        # # Make the combined text API call
        # combined_response = requests.post(combined_url, headers=headers, json=combined_payload)
        # combined_response_data = combined_response.json()
        # 
        # # Extract the generated text from the combined text response
        # combined_generated_text = combined_response_data['results'][0]['generated_text']


        #########################################        #########################################
        ### end of added functionality, pass combined_generated_text instead of combined_result
        #########################################        #########################################
        
        
        
        #return render_template_string('result.html', combined_result=combined_result)
        return render_template('result.html', combined_result=combined_result , original_text=user_input)


    except Exception as e:

        print(e)
        print(response_data)
        return render_template('error.html', error_message=str(response_data))


from urllib.parse import quote, unquote



@app.route('/setup_n_generate_text', methods=['GET'])
def setup_n_generate_text():
    try:
        user_config = request.args.get('user_config_area', default="100 99 Extract medical information from the conversation specific to entities mentioned in the text. Categorize the information into demographics, Chief Complaint, HPI, PMHx, Social History, Family History, ROS, state and types of Physical Exam and Test, Medications and dosages, Plans and Other mentioned high-value information, and corresponding billing and icd codes. Only include explicitly mentioned information and notes when needed. refrain from inferring or adding details not present in the text.")
        user_input = request.args.get('user_input', default="I am patient with diabetic, taking advil and night quels with men medicaitons such as viagra")


        print("Got the variables")    
        if not user_input:
            return render_template('error.html', error_message='Input text is required.')

        if not user_input:
            return render_template_string('error.html',error_message='Input text is required2.')
          
        if not user_config:
            return render_template('error.html', error_message='congif setting is required.')
        
        
        configs = user_config.split("99")
        
        maxTok, instruction = configs
        
  
        drug_names = find_drugs(user_input.split(" "),is_ignore_case=True)
        

        print("------Drugs-----")
        print(drug_names)

        drug_names2 = [entry[0]['name'] for entry in drug_names]


        print("------Drugs name-----")

        print(drug_names2)

        drug_names2 = " ".join(drug_names2)



        pieces = split_text_with_equal_size(user_input)
        
        url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
        headers = header
            #'Authorization': 'Bearer eyJraWQiOiIyMDI0MDEwNjA4MzciLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC01NTAwMDM1QUs1IiwiaWQiOiJJQk1pZC01NTAwMDM1QUs1IiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiNWQyYTIxNGMtODA2Ny00ZDY4LWEwNmItMzM3NDMzNTVlMzczIiwiaWRlbnRpZmllciI6IjU1MDAwMzVBSzUiLCJnaXZlbl9uYW1lIjoicGVkcmFtIiwiZmFtaWx5X25hbWUiOiJhaGFkaW5lamFkIiwibmFtZSI6InBlZHJhbSBhaGFkaW5lamFkIiwiZW1haWwiOiJwZWRyYW1AdGNlZ3JvdXAuY29tIiwic3ViIjoicGVkcmFtQHRjZWdyb3VwLmNvbSIsImF1dGhuIjp7InN1YiI6InBlZHJhbUB0Y2Vncm91cC5jb20iLCJpYW1faWQiOiJJQk1pZC01NTAwMDM1QUs1IiwibmFtZSI6InBlZHJhbSBhaGFkaW5lamFkIiwiZ2l2ZW5fbmFtZSI6InBlZHJhbSIsImZhbWlseV9uYW1lIjoiYWhhZGluZWphZCIsImVtYWlsIjoicGVkcmFtQHRjZWdyb3VwLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiJmMmE0N2JjM2ViYWQ0MWZhOTA3YWIzNzM2YWZmMDMzMiIsImltc191c2VyX2lkIjoiODk4MDEzMiIsImZyb3plbiI6dHJ1ZSwiaW1zIjoiMTczMzkxMSJ9LCJpYXQiOjE3MDcwMjQ3OTgsImV4cCI6MTcwNzAyODM5OCwiaXNzIjoiaHR0cHM6Ly9pYW0uY2xvdWQuaWJtLmNvbS9pZGVudGl0eSIsImdyYW50X3R5cGUiOiJ1cm46aWJtOnBhcmFtczpvYXV0aDpncmFudC10eXBlOmFwaWtleSIsInNjb3BlIjoiaWJtIG9wZW5pZCIsImNsaWVudF9pZCI6ImRlZmF1bHQiLCJhY3IiOjEsImFtciI6WyJwd2QiXX0.r8HyUcnbqRqGSHOYM1S4buS43qBwC0B9zDtoEJIP_koCi7IryAxyQ8hvTaZ4-to9lhDQDWI9N67Hx5II6A_jSwQD_gszsbD_BHzZ29hmJa6bDj9d7OI3ikkLPSymHf9UWfPQMfwsWTqXNCCy5lqEk-vBE0zfFQddPqoq99SUmiRvTrasBYTOVjaRL3AavZS_DTeaHAkT0tVZTB4HA9_E_C2rskMw3S3vCdquudWyYco81WZPeEXyzWohXfYr_NHCzsHTf0MBpm7aKZJITY3o_Ji1x6-jokGnhbZOZCB5eO55Xfha4_3_mi-TKXE54_AN00V1KasAmQJYqbUHYNinEg'}

        generated_texts = []
#"instruction : Extract medical information from the conversation specific to entities mentioned in the text. Categorize the information into demographics, Chief Complaint, HPI, PMHx, Social History, Family History, ROS, state and types of Physical Exam and Test, Medications and dosages, Plans and Other mentioned high-value information, and corresponding billing and icd codes. Only include explicitly mentioned information and notes when needed. refrain from inferring or adding details not present in the text. input text : "
        for piece in pieces:
            payload = {
                "model_id": "meta-llama/llama-2-70b-chat",
                "input": instruction + " input text : " + piece + ". medications: " + drug_names2 ,
                "parameters": {"decoding_method": "greedy", "max_new_tokens": int(maxTok), "min_new_tokens": 10, "stop_sequences": [], "repetition_penalty": 1},
                "project_id": "beaf6470-c5bc-4695-b204-29d09c8bf7fb",
                "moderations": {
                    "hap": {"input": True, "output": True, "threshold": 0.5, "mask": {"remove_entity_value": False}}
                }
            }
            print()
            print()
            print()


            print(">>>>> payload <<<<<")
            print(payload)
       
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
         
            generated_text = response_data['results'][0]['generated_text']
            generated_texts.append(generated_text)
            
            print()
            print()
            print()
            
            print(">>>>> RESULTS <<<<<")
            print(generated_texts)



        combined_result = "".join(generated_texts)





# Assuming drug_names is initialized somewhere above this snippet
        unique_drug_tuples = []  # List to hold unique drugs
        processed_drug_names = set()  # Set to track processed drug names and avoid duplicates

        for drug_tuple in drug_names:
            drug_info = drug_tuple[0]  # Access the drug info dictionary from the tuple
            if 'name' in drug_info:  # Ensure 'name' key exists in the dictionary
                drug_name = drug_info['name']  # Get the drug's name

        # Check if the drug has already been processed
                if drug_name not in processed_drug_names:
                    if 'synonyms' in drug_info:  # Ensure 'synonyms' key exists
                        drug_info['synonyms'] = list(drug_info['synonyms'])  # Convert set to list
                    unique_drug_tuples.append((drug_info, drug_tuple[1], drug_tuple[2]))  # Add the updated tuple
                    processed_drug_names.add(drug_name)  # Mark this drug as processed

# Serialize the modified data to JSON, ensuring no duplicates
        serialized_drugs_corrected = json.dumps(unique_drug_tuples)  # Optional: , indent=4 for pretty print


        #print(generated_texts)
        print(serialized_drugs_corrected)
        return render_template('result.html', combined_result=combined_result , original_text=user_input, drugs=serialized_drugs_corrected)


    except Exception as e:
        print(e)
        return render_template('error.html', error_message='issue is here')





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8099)
