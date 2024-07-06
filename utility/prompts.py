def get_system_prompt(table_name,table_details):
    """_summary_
    Initial prompt for utility/utils.py -> get_response
    Args:
        table_name (str): Name of table in database.
        table_details (str): Column details of the table.

    Returns:
        str: Prompt
    """
    details = f"""
Your task is to take in the user question and convert it into postgreSQL query
Make sure to enclose all column names in " " to avoid errors and output the sql query

The table details is as follows:
table_name = {table_name}
columns = 
{table_details}
"""
    return details


def correct_error_prompt(sql_generated : str, error_message : str) -> str:
    """_summary_
    Prompt for correct syntax errors in output sql query
    Args:
        sql_generated (str): Generated sql query that needs to be corrected.
        error_message (str): The Syntax Error message thrown by psql engine

    Returns:
        str: Prompt
    """
    prompt = f"""Evecuting the above query gave this is syntax error: {error_message}.
To correct this, please generate an alternative SQL query which will correct the syntax error. The updated query should take care of all the syntax issues encountered. Follow the instructions mentioned above to remediate the error.
Update the below SQL query to resolve the issue:
{sql_generated}
Make sure the updated SQL query aligns with the requirements provided in the initial question."""
    return prompt

###for tool_calls in main.py
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_response_from_database",
            "description": "Fetches results from the postgre database assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The result you want from the database in natural language",
                    }
                },
                "required": ["prompt"],
            },
        }
    }
]

system_prompt = """
You are a helpful assistant and have access to another database assistant.
Your job is to read the users questions and call the database assistant if needed.
Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

Here are the database fields present. The database assistant can only answer queries that revolve around these column names. If a user asks question outside of these fields do not call the database assistant.

"""

def get_main_prompt(table_details:str) -> str:
    """_summary_
    Prompt for main.py file
    Args:
        table_details (str): _description_

    Returns:
        str: Prompt
    """
    return system_prompt + table_details