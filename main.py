import sys
sys.path.append("..")

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine,text
from dotenv import load_dotenv
import os
from utility.utils import get_response, chat_completion_request, get_table_details
from utility.utils import num_tokens_from_string
from utility.prompts import tools,get_main_prompt
import json
import time
load_dotenv()
# from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_chat():
    """_summary_
    Main function to run the chat
    """
    
    engine = create_engine(os.getenv("DATABASE_URL"),pool_pre_ping=True)
    table_name = "sample_table2"
    path = "data/big_sample.xlsx"
    # upload_excel_to_psql(path = path,table_name=table_name,engine=engine)
    table_details = get_table_details(table_name,engine)
    system_prompt= get_main_prompt(table_details)
    messages = []
    
    messages.append({"role": "system", "content": system_prompt})
    print("Hi, how can I help you?")
    
    while(True):
        question = input()
        if(question == "exit"):
            break
        messages.append({"role": "user", "content": question})
        chat_response = chat_completion_request(
            messages, tools=tools
        )
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)
        tool_calls = assistant_message.tool_calls
        if(tool_calls):
            prompt = json.loads(assistant_message.tool_calls[0].function.arguments)["prompt"]
            function_response = get_response(question = prompt,table_name=table_name,engine=engine)
            token = num_tokens_from_string(str(function_response))
            if(token > 8000):
                function_response = "The database response is larger than the token limit. Please try Giving a shorter query."
            messages.append(
                    {
                        "tool_call_id": tool_calls[0].id,
                        "role": "tool",
                        "name": tool_calls[0].function.name,
                        "content": function_response,
                    }
            )
            messages.append({
                "role" : "assistant",
                "content" : function_response
            })
            print()
            print(function_response)
            print()
            continue
        print()
        print(assistant_message.content)
        print()
        time.sleep(0.3)
        
if __name__ == "__main__":
    run_chat()
