from sentence_transformers import SentenceTransformer
import numpy as np
from connect_gemini import Setup_Envir  


from pprint import pprint

# pprint(Setup_Envir())

API_KEY = Setup_Envir()

# measuring the similarity by two vectors

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))



model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# pip install senstence_transformers

from lucknowllm import UnstructuredDataLoader, split_into_segments

loader = UnstructuredDataLoader()
# instead of vector DB
external_database = loader.get_data(folder_name="d:/source/RAG-LLM/html-dump")

# to be implemented 
# - search for the docs in html-dump that matches the best according query content
chunks = split_into_segments(external_database[0]['data'])
embedded_data = model.encode(chunks)


queries = ["Who is Alfred Nobel?"]
embedded_queries = model.encode(queries)

pprint(chunks)
# you may set up the model from lucknowllm - instead of from scratch
# from lucknowllm import GeminiModel
# Gemini = GeminiModel(api_key = Setup_Envir(), model_name = "gemini-1.0-pro")

from lucknowllm import GeminiModel

Gemini = GeminiModel(api_key = API_KEY, model_name = "gemini-1.0-pro")

import google.ai.generativelanguage_v1 as glm

client = glm.GenerativeServiceClient(
    client_options=dict(api_key=API_KEY))

response = client.generate_content({
"model": "models/gemini-1.5-flash",
"contents": [ {"parts": [ {"text": "Explain how AI works"}]}]
})

print(type(response).to_dict(response))

# ---------------------------------------------
# similar but using google generative AI
# ---------------------------------------------

# print('--------------------------------------------------------')

# import google.generativeai as gen_ai

# # # pip install google_generativeai

# gen_ai.configure(api_key=API_KEY)

# # Set up the model
# generation_config = {
#   "temperature": 0.9,
#   "top_p": 1,
#   "top_k": 1,
#   "max_output_tokens": 2048,
# }

# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
# ]


# Gemini = gen_ai.GenerativeModel(model_name="gemini-1.0-pro",
#                               generation_config=generation_config,
#                               safety_settings=safety_settings)


# # # Doing the pull from gemini

# for i, query_vec in enumerate(embedded_queries):
#     # Compute similarities
#     similarities = cosine_similarity(query_vec[np.newaxis, :], embedded_data)

#     # Get top 3 indices based on similarities
#     top_indices = np.argsort(similarities[0])[::-1][:3]
#     top_doct = [chunks[index] for index in top_indices]

#     # Print the top 3 similar sentences
#     argumented_prompt = f"You are an expert question answering system, I'll give you question and context and you'll return the answer. Query : {queries[i]} Contexts : {top_doct[0]}"
    
    
#     model_output = Gemini.generate_content(argumented_prompt)
#     #print(model_output)
#     # If you're not importing the Gemini model from the lucknowllm package
#     # Use the following code to get the output
#     print(model_output.text)