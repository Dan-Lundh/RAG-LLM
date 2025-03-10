from dotenv import load_dotenv
import os
import requests
from pprint import pprint


def Setup_Envir():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    return API_KEY

print(Setup_Envir)

