from dotenv import load_dotenv
import together
from utility.prompts import get_system_prompt, correct_error_prompt, tools
import tiktoken
import os
import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine,text
import sqlalchemy
import time
import openai

load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Wrapper functions to calculate time and handle rate limit errors
def calculate_time(func):
    def inner1(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
    return inner1

def retry(wait: int = 1):
    def retry_inner(func):
        def inner2(*args, **kwargs):
            try:
                output = func(*args,**kwargs)
            except:
                time.sleep(1)
                output = func(*args,**kwargs)
            return output
        return inner2
    return retry_inner

@retry()
def get_response_together(prompt : str,model :str = "mistralai/Mistral-7B-Instruct-v0.2",temperature :int = 0) -> str:
    """_summary_
    Get response from together ai 
    Args:
        prompt (str): prompt for inference
        model (str, optional): Model to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        temperature (int, optional): temperature of the model. Defaults to 0.

    Returns:
        str: Model output
    """
    output = together.Complete.create(
        prompt = prompt,
        model = model, 
        max_tokens = 1536,
        stop = ["</s>", "[INST]", ".\n\n", "Paragraph","AI","Question","\n\n"], 
        temperature = temperature,
        top_k = 50,
        top_p = 0.7
    )
    print(output['output']['choices'][0]['text'])
  # print generated text
    return output['output']['choices'][0]['text']

def get_response_openai(message : list)->str:
    """_summary_
    Get response from openai GPT
    Args:
        message (list): Sequence of messages in GPT format

    Returns:
        str: Model response
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,
        max_tokens=512
    )
    return completion.choices[0].message.content

def num_tokens_from_string(string: str) -> int:
    """_summary_
    Count the number of tokens for a string
    Args:
        string (str): String to count token

    Returns:
        int: Number of tokens in the string
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def upload_excel_to_psql(path : str,engine,table_name : str = "sample_table2") -> bool:
    """_summary_
    Uploads the excel file to postgreSQL
    Args:
        path (str): path of excel file
        engine (_type_): sqlalchemy engine for postgre database
        table_name (str, optional): Table name to insert in database. Defaults to "sample_table2".

    Returns:
        bool: Status of upsert
    """
    try:
      df = pd.read_excel(path)
      df.to_sql(table_name,engine)
      return True
    except Exception as e:
      return False
    
def get_table_details(table_name : str,engine : sqlalchemy.engine.base.Engine) -> str:
    """_summary_
    Fetches the column details from table in database
    Todo:
        Check if table exists
    Args:
        table_name (str): Name of the table in database
        engine (sqlalchemy.engine.base.Engine): sqlalchemy engine for postgre database

    Returns:
        str: Formatted Column details
    """
    with engine.connect() as conn:
      response = conn.execute(text(f"""select column_name, data_type from INFORMATION_SCHEMA.COLUMNS where table_name = '{table_name}';"""))
    data = []
    for i in response:
      data.append((i[0].strip(" "),i[1].strip(" ")))
    table_details = [f"{str(i).strip(' ')}\n" for i in data]
    table_details = "".join(table_details)
    return table_details

def response_to_sql(response:str) -> str:
    """_summary_
    Converts the llm response to sql_query
    Args:
        response (str): Response from the LLM

    Returns:
        str: Sql query to execute in database
    """
    try:
        query_str = response.split("```")[1]
        query_str = " ".join(query_str.split("\n")).strip("\n").strip(" ")
        sql_query = query_str[3:] if query_str.startswith("sql") else query_str
    except Exception as e:
        sql_query = response
    return sql_query

def fetch_results_from_psql(sql_query : str,engine: sqlalchemy.engine.base.Engine) -> str:
    """_summary_
    Runs the SQL query in the database
    Args:
        sql_query (str): Query to run
        engine (sqlalchemy.engine.base.Engine): sqlalchemy engine for postgre database

    Returns:
        str: _description_
    """
    try:
        with engine.connect() as curs:
            result = curs.execute(text(sql_query))
        results = [i for i in result]
        return True,results
    except Exception as e:
        error_name = e.__class__.__name__
        error_msg = str(e)
        return False, {"error_msg" : error_msg, "error_name" : error_name}
    

def get_response(question : str,engine : sqlalchemy.engine.base.Engine,table_name : str,max_retries : int = 0) -> str:
    """_summary_
    Main function to fetch results of the question asked
    Args:
        question (str): Question to be asked from the database
        engine (sqlalchemy.engine.base.Engine): sqlaclhemy engine for postgre
        table_name (str): name of the table in database
        max_retries (int, optional): No.of tries to correct the syntax and retry in case of syntax error. Defaults to 0.

    Returns:
        str: Response to the question
    """
    table_details = get_table_details(table_name = table_name,engine = engine)
    details = get_system_prompt(table_name = table_name,table_details = table_details)
    if(max_retries < 0):
        print("max_retries must be >= 0. Defaulting to 0")
        max_retries = 0
    elif(max_retries > 4):
        print("Warning - Max retires > 4 will incur massive gpt charges")
    user_query = question
    messages = [
        {"role":"system","content":details},
        {"role":"user","content":user_query}
    ]

    response = get_response_openai(messages)
    sql_query = response_to_sql(response)
    ret,results = fetch_results_from_psql(sql_query , engine)

    for i in range(0,max_retries):
        if(ret):
            break
        print(f"Query Failed to execute Retrying attempt : {i+1}")
        error_msg = results["error_msg"]
        error_name = results["error_name"]
        error_prompt = correct_error_prompt(sql_query,error_msg)
        error_prompt = [
            {"role" : "assistant", "content" : response},
            {"role" : "user", "content" : error_prompt}
            ]
        error_prompt = messages + error_prompt
        response = get_response_openai(error_prompt)
        sql_query = response_to_sql(response)
        ret,results = fetch_results_from_psql(sql_query , engine)

    if(not ret):
        print("Failed to execute query with given max retries. Either increase max retries or alter the query.")
        return False

    if(len(results) == 0):
        results = "0 matches found"

    message = [
        {"role" : "assistant", "content" : response},
        {"role" : "system", "content" : f"The query was run and following is its output : {results}. Answer the user question using the sql query output in a normal way. Do not output the query."},
        {"role" : 'user', "content" : user_query}
    ]

    messages+=message
    token = num_tokens_from_string(str(results))
    if(token > 8000):
        print("WARNING - Token limit reached in response from database. Showing raw query output instead.")
        return results
    response = get_response_openai(messages)
    return response



def chat_completion_request(messages : list, tools : dict=tools, tool_choice : list=None, model : str="gpt-3.5-turbo") -> openai.types.chat.chat_completion.ChatCompletion:
    """_summary_
    Function to get GPT response with function calling
    Args:
        messages (list): Message history
        tools (dict, optional): Description of tools/functions. Defaults to tools.
        tool_choice (list, optional): List of tools to choose from. Defaults to None.
        model (str, optional): Name of the model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        openai.types.chat.chat_completion.ChatCompletion: assistants response to the messages
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    return response



