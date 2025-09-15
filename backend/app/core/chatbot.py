# /app/core/chatbot.py

import re
import os
import httpx
import json
import asyncio
import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi import WebSocket, WebSocketDisconnect
import json
import uuid
import inspect
from datetime import datetime # 1. IMPORT DATETIME

# --- Pydantic Models & Router (Unchanged) ---
class ChatMessage(BaseModel):
    role: str
    parts: List[str]

class ChatHistory(BaseModel):
    history: List[ChatMessage]

# WebSocket manager for plan updates
class PlanUpdateManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_plan_update(self, data: dict):
        for connection in self.active_connections:
            await connection.send_json(data)

router = APIRouter(prefix="/chat", tags=["Chatbot Core"])
http_client = httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1")
plan_manager = PlanUpdateManager()

# --- NEW: Code Session Management ---
class CodeManager:
    def __init__(self):
        self.code_sessions = {}
    
    def create_session(self, initial_code):
        session_id = str(uuid.uuid4())
        self.code_sessions[session_id] = {
            'current_code': initial_code,
            'modification_history': []
        }
        return session_id
    
    def update_code(self, session_id, modification, new_code):
        if session_id in self.code_sessions:
            self.code_sessions[session_id]['current_code'] = new_code
            self.code_sessions[session_id]['modification_history'].append(modification)
            return True
        return False
    
    def get_code(self, session_id):
        return self.code_sessions.get(session_id, {}).get('current_code', '')
    
    def cleanup_session(self, session_id):
        if session_id in self.code_sessions:
            del self.code_sessions[session_id]

code_manager = CodeManager()

# --- MAJOR ARCHITECTURAL UPGRADE ---

# 1. Tool Chest with 'create_plan' for the Architect Engine
# --- MODIFIED: Added commit_github_file tool ---
TOOLS_AVAILABLE = {
    "create_plan": {
        "description": "Analyzes the user's request and the available code to create a detailed, step-by-step plan (To-Do list) for how to achieve a complex goal. This should be the very first step for any complex code modification or creation task.",
        "params": {}
    },
    "create_google_task": {
        "description": "Creates a new task in a specific Google Tasks list by its name. Use for requests like 'add a task to my shopping list'.",
        "params": {"list_title": "string (the name of the list, e.g., 'Shopping')", "task_title": "string (the content of the task, e.g., 'Buy milk')"}
    },
    "get_google_tasks": {
        "description": "Fetches tasks from the user's Google Tasks lists. Use for queries like 'what are my tasks' or 'fetch my to-dos'.",
        "params": {}
    },
    "list_repos": {
        "description": "Fetches a list of the user's GitHub repositories.",
        "params": {}
    },
    "list_github_files": {
        "description": "Fetches a list of all files and directories in a GitHub repository.",
        "params": {"repo_full_name": "string (e.g., 'user/repo')"}
    },
    "get_github_file": {
        "description": "Fetches the content of a specific file from a GitHub repository.",
        "params": {"repo_full_name": "string (e.g., 'user/repo')", "file_path": "string (e.g., 'src/main.py')"}
    },
    "commit_github_file": {
        "description": "Updates a file in a GitHub repository and commits the change. Use this after modifying code.",
        "params": {
            "repo_full_name": "string (e.g., 'user/repo')", 
            "file_path": "string (e.g., 'src/main.py')",
            "new_content": "string (the full, new content of the file)",
            "commit_message": "string (a descriptive message for the commit)"
        }
    },
    "web_search": {
        "description": "Performs a web search for a given query.",
        "params": {"query": "string"}
    },
    "modify_code": {
        "description": "Use this action ONLY AFTER creating a plan. It is used to execute a single, specific step from your plan.",
        "params": {"modification_request": "string (a clear, single instruction from your plan, e.g., 'change the --hero-gradient to use only two colors')"}
    },
    "Finish": {
        "description": "Use this action when you have successfully completed all steps in your plan and are ready to respond to the user.",
        "params": {"answer": "string (your final, comprehensive answer to the user)"}
    }
}

# 2. Upgraded Cognitive Router with Clear Complexity Tiers
COGNITIVE_ROUTER_PROMPT = f"""
You are Virax Jr., an AI assistant with a modular cognitive architecture. Your primary function is to analyze a user's request and select the most appropriate execution mode based on its complexity.

--- CONTEXT ---
Current Date: {datetime.now().strftime("%B %d, %Y")}

--- COGNITIVE MODES ---

1.  **MODE_1_DIRECT_DISPATCH:**
    - **Complexity:** Trivial. A single, direct command.
    - **Use Case:** - "What are my repos?", 
       - "Add 'buy milk' to my shopping list", 
       - "Read the README.md in 'virax/main' repo.",
       - "Fetch index.html from user/repo",
       - "Get the content of file.txt from owner/repository"
    - **Output:**
        MODE: MODE_1_DIRECT_DISPATCH
        TOOL: [tool_name]
        PARAMS: [JSON object of parameters]

2.  **MODE_2_AGILE_AGENT (ReAct Framework):**
    - **Complexity:** Medium. Requires a few steps but no complex planning.
    - **Use Case:** "Search for the latest FastAPI version and then find its documentation."
    - **Output (First Step):**
        MODE: MODE_2_AGILE_AGENT
        GOAL: [A clear statement of the user's goal]
        Thought: [Your initial reasoning]
        Action: [The first tool call, e.g., web_search()]

3.  **MODE_3_ARCHITECT_ENGINE (Plan-and-Execute):**
    - **Complexity:** High. Requires understanding existing code, planning multiple changes, and executing them sequentially.
    - **Use Case:** Any request to modify or refactor code and commit it, like "Refactor this function and commit the changes," "change the UI of this page and push an update."
    - **MANDATORY WORKFLOW:** Your first action MUST be `create_plan()`.
    - **Output (First Step):**
        MODE: MODE_3_ARCHITECT_ENGINE
        GOAL: [A clear statement of the user's ultimate goal]
        Thought: The user is asking for a complex code modification. I must first create a detailed plan before making any changes.
        Action: create_plan()

--- GITHUB OPERATIONS GUIDANCE ---
- If the user asks to fetch/read/get a specific file from a repository: USE MODE_1_DIRECT_DISPATCH with get_github_file
- If the user asks to list files or see the contents of a repository: USE MODE_1_DIRECT_DISPATCH with list_github_files
- If the user asks to modify/update/change a file in a repository: USE MODE_3_ARCHITECT_ENGINE
- If the user asks to create a new file in a repository: USE MODE_3_ARCHITECT_ENGINE

If the user is just having a conversation (greetings, questions about capabilities, etc.), output: MODE: CONVERSATIONAL

--- AVAILABLE TOOLS ---
{json.dumps(TOOLS_AVAILABLE, indent=2)}
---

Analyze the user's message and conversation history. Select the appropriate mode and provide the required information.

User's message:
"""



# 3. Parsers and an Upgraded Tool Executor
def parse_router_decision(decision_text: str):
    """
    Parses the output from the cognitive router.
    Handles both JSON and line-by-line formats robustly.
    """
    decision = {"MODE": "CONVERSATIONAL", "TOOL": None, "PARAMS": None, "GOAL": None, "Thought": None, "Action": None}
    
    # Clean the text - remove markdown code blocks and extra whitespace
    clean_text = re.sub(r'```(?:json)?\s*', '', decision_text).strip()
    # print(f"[DEBUG] Cleaned Router Output:\n{clean_text}") # Uncomment for debugging

    # --- Strategy 1: Parse line-by-line for key-value pairs (Robust for the router's format) ---
    lines = clean_text.split('\n')
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            key_upper = key.upper()
            if key_upper in ['MODE']:
                decision["MODE"] = value
            elif key_upper in ['TOOL']:
                decision["TOOL"] = value
            elif key_upper in ['PARAMS', 'PARAMETERS']:
                # If value starts with '{', try to grab all lines until closing '}'
                if value.startswith('{'):
                    param_lines = [value]
                    for next_line in lines[idx+1:]:
                        param_lines.append(next_line)
                        if next_line.strip().endswith('}'):
                            break
                    value_stripped = '\n'.join(param_lines)
                    try:
                        decision["PARAMS"] = json.loads(value_stripped)
                    except Exception as e:
                        decision["PARAMS"] = value_stripped
                else:
                    # This regex finds key='value' or key="value" patterns.
                    kv_pairs = re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", value)
                    if kv_pairs:
                        decision["PARAMS"] = {k: v for k, v in kv_pairs}
                    else:
                        decision["PARAMS"] = value
            elif key_upper in ['GOAL', 'OBJECTIVE']:
                decision["GOAL"] = value
            elif key_upper in ['THOUGHT', 'REASONING']:
                decision["Thought"] = value
            elif key_upper in ['ACTION', 'NEXT STEP']:
                decision["Action"] = value

    # --- Strategy 2: Fallback to full JSON parsing (if the entire output is a JSON object) ---
    # This is less likely for the router but good to have.
    if decision["MODE"] == "CONVERSATIONAL" and decision["TOOL"] is None:
         json_match = re.search(r'\{[\s\S]*\}', clean_text)
         if json_match:
             try:
                 json_decision = json.loads(json_match.group())
                 for key in decision:
                     if key in json_decision:
                         decision[key] = json_decision[key]
             except json.JSONDecodeError:
                 pass # Stick with line-parsed or default values

    # --- Final Check: If MODE is still CONVERSATIONAL but a TOOL was identified, 
    # it's likely a direct dispatch.
    if decision["MODE"] == "CONVERSATIONAL" and decision["TOOL"]:
        decision["MODE"] = "MODE_1_DIRECT_DISPATCH"
        
    return decision

def parse_react_action(llm_output: str) -> Dict:
    thought_match = re.search(r"Thought:\s*(.*)", llm_output, re.DOTALL)
    action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action_str = action_match.group(1).strip() if action_match else ""
    return {"thought": thought, "action": action_str}

# --- MODIFIED: Function to get the appropriate model based on the .env file ---
def get_model_for_task(task_type: str):
    """Selects the Gemini model and API key based on the task type."""
    if task_type == "max_reasoning": # For very high-level reasoning or large coding tasks
        api_key = os.getenv("GEMINI_API_KEY_PRIMARY")
        model_name = os.getenv("GEMINI_MODEL_MAX", "gemini-2.5-pro")
    elif task_type == "high_reasoning": # For multi-step reasoning in complex agent loops
        api_key = os.getenv("GEMINI_API_KEY_PRIMARY")
        model_name = os.getenv("GEMINI_MODEL_HIGH", "gemini-2.5-flash")
    elif task_type == "planning": # For creating plans as per user request
        api_key = os.getenv("GEMINI_API_KEY_PRIMARY")
        model_name = os.getenv("GEMINI_MODEL_MEDIUM", "gemini-2.5-flash-lite")
    elif task_type == "standard": # For standard agentic tasks, synthesis, and normal conversation
        api_key = os.getenv("GEMINI_API_KEY_SECONDARY")
        model_name = os.getenv("GEMINI_MODEL_STANDARD", "gemini-2.0-flash")
    else: # "lite" for routing, simple classifications, or default
        api_key = os.getenv("GEMINI_API_KEY_SECONDARY")
        model_name = os.getenv("GEMINI_MODEL_LITE", "gemini-2.0-flash-lite")

    if not api_key:
        raise HTTPException(status_code=500, detail=f"GEMINI_API_KEY for task type '{task_type}' not found.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# --- UPDATED execute_tool function ---
async def execute_tool(action_str: str, chat_data: ChatHistory, goal: str, session_id=None) -> str:
    action_str = action_str.strip()

    # Finish is a virtual tool that stops the agent loop
    if action_str.startswith("Finish("):
        return action_str
    
    try:
        if action_str == "create_plan()":
            # --- MODIFIED LOGIC START ---
            # Check if the goal is to create/generate a new file in a repo.
            creation_keywords = ['create', 'generate', 'write', 'make']
            is_creation_goal = any(keyword in goal.lower() for keyword in creation_keywords) and "repo" in goal.lower() and "file" in goal.lower()

            if is_creation_goal:
                plan_text = (
                    "1. Verify if the file already exists in the repository to decide whether to create or modify.\n"
                    "2. If the file does not exist, explore the repository to understand its structure and contents.\n"
                    "3. Generate the content for the new file based on the goal and repository analysis.\n"
                    "4. Commit the new file to the repository with a descriptive message."
                )
                session_id = code_manager.create_session("") # Create an empty session
                return f"""I have analyzed the goal and created a plan to create the new file. Session ID: {session_id}
```plan
{plan_text}
```"""
            # --- MODIFIED LOGIC END ---

            code_context = ""
            for msg in reversed(chat_data.history[:-1]):
                code_block_match = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", msg.parts[0], re.DOTALL)
                if code_block_match:
                    code_context = code_block_match.group(1)
                    break
            
            if not code_context:
                return "STOP. I cannot create a plan without the code. Please provide the relevant code file in a markdown block."

            plan_prompt = f"""You are a senior software architect. Your goal is: "{goal}".
Analyze the following code that was provided by the user:
{code_context}
Based on the goal, create a concise, numbered list of the specific steps required to achieve it. This is your To-Do list. Each step should be a single, clear action that can be executed. Format your response EXACTLY like this:
1. First step description
2. Second step description
..."""
            model = get_model_for_task("planning")
            plan_response = model.generate_content(plan_prompt)
            
            plan_text = plan_response.text
            lines = plan_text.split('\n')
            numbered_steps = [line.strip() for line in lines if re.match(r'^\d+\.', line.strip())]
            
            if numbered_steps:
                plan_text = '\n'.join(numbered_steps)
            else: # Fallback
                plan_text = "1. Analyze the current implementation\n2. Apply the requested changes\n3. Test the changes for consistency"
            
            session_id = code_manager.create_session(code_context)
            
            return f"""I have analyzed the provided code and created a step-by-step plan. Session ID: {session_id}
```plan
{plan_text}
```"""

        if action_str.startswith("modify_code"):
            if not session_id:
                return "Error: No active code session. A plan must be created first to start a session."
            
            current_code = code_manager.get_code(session_id)
            if not current_code:
                return "Error: Code not found in the current session."
            
            params_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
            params_str = params_match.group(1) if params_match else ""
            req_match = re.search(r"modification_request\s*=\s*['\"]([^'\"]*)['\"]", params_str)
            if not req_match:
                return "Error: Missing 'modification_request' parameter in modify_code()."
            
            modification_request = req_match.group(1)
            
            modification_prompt = f"""As an expert developer, generate a complete README.md for the project based on the following code and files.
Modification Request: "{modification_request}"

Project Code Context:
{code_for_readme}

Return ONLY the complete README.md markdown code inside a single markdown block. Do not add any explanation.
"""
            model = get_model_for_task("max_reasoning")
            modification_response = model.generate_content(modification_prompt)
            
            modified_code = modification_response.text.strip()
            code_block_match = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", modified_code, re.DOTALL)
            if code_block_match:
                modified_code = code_block_match.group(1).strip()

            code_manager.update_code(session_id, modification_request, modified_code)
            return f"Successfully applied modification: {modification_request}"

        # --- Other Tools ---
        tool_name_match = re.match(r"(\w+)\(", action_str)
        if not tool_name_match: return f"Error: Could not parse tool name from action '{action_str}'"
        tool_name = tool_name_match.group(1).strip()
        params_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
        params_str = params_match.group(1) if params_match else ""
        if tool_name not in TOOLS_AVAILABLE: return f"Error: Tool '{tool_name}' not found."
        
        try:
            args = json.loads(params_str) if params_str.startswith('{') else dict(re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", params_str))
        except (json.JSONDecodeError, TypeError):
            return f"Error: Could not parse parameters from '{params_str}'"

        if tool_name == "list_github_files":
            try:
                owner, repo_name = args.get('repo_full_name', '/').split('/')
                if not all([owner, repo_name]):
                    error_msg = "Error: Missing parameters for list_github_files."
                    print(f"[ERROR] {error_msg} Args: {args}")
                    return error_msg
                response = await http_client.get(f"/github/repos/{owner}/{repo_name}/tree")
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return "Observation: Repository not found."
                else:
                    raise e
            except ValueError as ve:
                error_msg = f"Error: Invalid repository format in 'repo_full_name'. Expected 'owner/repo'. Args: {args}"
                print(f"[ERROR] {error_msg}")
                return error_msg
            except Exception as ex:
                error_msg = f"Error executing list_github_files: {str(ex)}. Args: {args}"
                print(f"[ERROR] {error_msg}")
                return error_msg
        elif tool_name == "get_github_file":
            try:
                print(f"[DEBUG] Raw args for get_github_file: {args}")

                owner, repo_name = args.get('repo_full_name', '/').split('/')
                file_path = args.get('file_path')
                if not all([owner, repo_name, file_path]): 
                    error_msg = "Error: Missing parameters for get_github_file."
                    print(f"[ERROR] {error_msg} Args: {args}")
                    return error_msg
                response = await http_client.get(f"/github/repos/{owner}/{repo_name}/contents/{file_path}")
                response.raise_for_status() 
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return "Observation: File not found. You can now proceed with creating a new file."
                else:
                    raise e
            except ValueError as ve:
                error_msg = f"Error: Invalid repository format in 'repo_full_name'. Expected 'owner/repo'. Args: {args}"
                print(f"[ERROR] {error_msg}")
                return error_msg
            except Exception as ex:
                error_msg = f"Error executing get_github_file: {str(ex)}. Args: {args}"
                print(f"[ERROR] {error_msg}")
                return error_msg
                    
        elif tool_name == "get_google_tasks":
            all_tasks = {}
            list_response = await http_client.get("/tasks/lists")
            list_response.raise_for_status()
            task_lists = list_response.json().get("task_lists", [])
            
            for task_list in task_lists:
                list_id = task_list['id']
                list_title = task_list['title']
                tasks_response = await http_client.get(f"/tasks/{list_id}")
                tasks_response.raise_for_status()
                tasks = tasks_response.json().get("tasks", [])
                if tasks:
                    all_tasks[list_title] = tasks
            return json.dumps(all_tasks)

        elif tool_name == "create_google_task":
            list_title_to_find = args.get('list_title')
            task_title_to_create = args.get('task_title')
            if not all([list_title_to_find, task_title_to_create]):
                return "Error: Missing 'list_title' or 'task_title' for create_google_task."

            list_response = await http_client.get("/tasks/lists")
            list_response.raise_for_status()
            task_lists = list_response.json().get("task_lists", [])
            
            target_list_id = None
            for task_list in task_lists:
                if task_list['title'].lower() == list_title_to_find.lower():
                    target_list_id = task_list['id']
                    break
            
            if not target_list_id:
                return f"Error: Could not find a Google Tasks list named '{list_title_to_find}'."

            create_response = await http_client.post(
                f"/tasks/{target_list_id}",
                json={"title": task_title_to_create}
            )
            create_response.raise_for_status()
            return f"Successfully created task '{task_title_to_create}' in list '{list_title_to_find}'."

        elif tool_name == "list_repos":
            response = await http_client.get("/github/repos")
            
        elif tool_name == "commit_github_file":
            owner, repo_name = args.get('repo_full_name', '/').split('/')
            file_path = args.get('file_path')
            new_content = args.get('new_content')
            commit_message = args.get('commit_message')
            if not all([owner, repo_name, file_path, new_content, commit_message]):
                return "Error: Missing parameters for commit_github_file."
            
            update_payload = {
                "new_content": new_content,
                "commit_message": commit_message
            }
            response = await http_client.put(
                f"/github/repos/{owner}/{repo_name}/contents/{file_path}",
                json=update_payload
            )
            
        elif tool_name == "web_search":
            query = args.get('query')
            if not query: return "Error: Missing 'query' parameter for web_search."
            # FIXED: Added a trailing slash to the URL
            response = await http_client.post("/search/", json={"query": query})
            
        else: 
            return f"Error: Tool '{tool_name}' is defined but not implemented."

        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e: return f"Error executing action '{action_str}': {str(e)}"

# --- Main Chat Endpoint with Multi-Mode Agentic Logic ---
@router.post("/", summary="Send a message to the agent")
async def handle_chat_message(chat_data: ChatHistory):
    user_message = chat_data.history[-1].parts[0]
    conversation_history_str = "\n".join([f"{msg.role}: {msg.parts[0]}" for msg in chat_data.history])
    
    print("\n" + "="*50 + f"\n| 📥 NEW REQUEST: \"{user_message}\"\n" + "="*50)

    # --- MODIFIED: Use 'lite' model for the initial routing ---
    router_model = get_model_for_task("lite")
    router_prompt_with_query = COGNITIVE_ROUTER_PROMPT + conversation_history_str
    router_response = router_model.generate_content(router_prompt_with_query)
    print(f"\n| 🧠 RAW ROUTER RESPONSE:\n{router_response.text}")  # ADD THIS LINE
    decision = parse_router_decision(router_response.text)    
    print("\n" + "-"*50 + "\n| 🧠 ROUTER DECISION\n" + "-"*50 + f"\n{json.dumps(decision, indent=2)}")

    final_response_text = ""
    mode_used = decision["MODE"]
    goal = decision.get("GOAL", user_message)
    session_id = None # Initialize session_id
    generated_readme_content = None

    if mode_used in ["MODE_2_AGILE_AGENT", "MODE_3_ARCHITECT_ENGINE"]:
        # --- MODIFIED: Determine the agent model based on the mode ---
        agent_task_type = "high_reasoning" if mode_used == "MODE_3_ARCHITECT_ENGINE" else "standard"
        agent_model = get_model_for_task(agent_task_type)
        
        max_iterations = 20
        agent_scratchpad = ""
        thought = decision.get("Thought", "I need to start working on the goal.")
        action_str = decision.get("Action", "Finish(answer='I'm not sure how to proceed.')")
        current_plan = None
        print("\n" + "="*50 + f"\n| 🏃‍♂️ ENTERING AGENT MODE: {mode_used}\n| GOAL: {goal}\n" + "="*50)


        # --- Stuck loop detection variables ---
        recent_action_obs = []  # List of (action, observation) tuples
        stuck_threshold = 3     # Number of repeats to consider stuck

        for i in range(max_iterations):
            print(f"\n--- STEP {i+1} ---")
            agent_scratchpad += f"Thought: {thought}\n"
            print(f"🤔 Thought: {thought}")
            print(f"🎬 Action: {action_str}")

            cleaned_action = action_str.strip()

            if cleaned_action.startswith("Finish("):
                final_response_text = cleaned_action[len("Finish("):-1].strip().strip("'\"")
                print("\n" + "="*50 + "\n| ✅ AGENT FINISHED\n" + "="*50)
                break

            observation = await execute_tool(cleaned_action, chat_data, goal, session_id)

            # --- Stuck loop detection: check last N (action, observation) pairs ---
            recent_action_obs.append((cleaned_action, str(observation).strip()))
            if len(recent_action_obs) > stuck_threshold:
                recent_action_obs.pop(0)
            # If all last N pairs are identical, break as stuck
            if len(recent_action_obs) == stuck_threshold and all(pair == recent_action_obs[0] for pair in recent_action_obs):
                print("\n[AGENT LOOP] Detected stuck loop (repeated action+observation). Breaking early.")
                break

            # --- Extract README.md markdown code if present in observation ---
            if "```markdown" in observation:
                md_match = re.search(r"```markdown\s*(.*?)```", observation, re.DOTALL)
                if md_match:
                    generated_readme_content = md_match.group(1).strip()

            if cleaned_action == "create_plan()" and "Session ID:" in observation:
                session_match = re.search(r"Session ID: ([\w-]+)", observation)
                if session_match:
                    session_id = session_match.group(1)
                    print(f"🔑 New Code Session Started: {session_id}")

            if cleaned_action == "create_plan()" and "I have analyzed the provided code" in observation:
                plan_match = re.search(r"```plan\s*(.*?)```", observation, re.DOTALL)
                if plan_match:
                    current_plan = plan_match.group(1).strip()
                    plan_lines = current_plan.split('\n')
                    total_steps = len([line for line in plan_lines if re.match(r'^\d+\.', line.strip())])
                    await plan_manager.send_plan_update({
                        "plan": current_plan, "current_step": 0, "total_steps": total_steps, "status": "plan_created"
                    })

            char_limit = 250
            print(f"👀 Observation: {observation[:char_limit]}{'...' if len(observation) > char_limit else ''}")

            agent_scratchpad += f"Action: {cleaned_action}\nObservation: {observation}\n"

            react_prompt_template = """Your GOAL is: '{goal}'. Previous steps are on the scratchpad. Decide the next thought and action.
If a plan exists, execute the next step. If all steps are done, use Finish().

--- CONTEXT ---
Current Date: {current_date}

--- ACTION FORMAT ---
You MUST format your action as a single-line Python function call.
Example: Action: web_search(query=\"latest AI trends\")

--- GUIDELINES FOR GITHUB OPERATIONS ---
When your goal is to create or modify a file on GitHub, you MUST first verify if the file exists.
1. Use the `get_github_file` tool to check for the file.
2. If the tool returns the file's content, the file EXISTS. Proceed with a plan to modify it.
3. If the tool observation explicitly says 'File not found', the file DOES NOT EXIST. Proceed with a plan to create it.
---

--- SCRATCHPAD ---
{scratchpad}
---
Tools: {tools}
Thought:"""

            react_prompt = react_prompt_template.format(goal=goal, scratchpad=agent_scratchpad, tools=json.dumps(TOOLS_AVAILABLE, indent=2), current_date=datetime.now().strftime("%B %d, %Y"))

            llm_response = agent_model.generate_content(react_prompt)
            parsed_action = parse_react_action("Thought: " + llm_response.text)
            thought, action_str = parsed_action["thought"], parsed_action["action"]


        # Always show the generated README.md if it was produced, even if the commit fails
        readme_found = False
        commit_status = None
        if 'commit_github_file' in agent_scratchpad:
            if 'Error executing action' in agent_scratchpad or '403' in agent_scratchpad or '500' in agent_scratchpad:
                commit_status = "❌ Commit to GitHub failed. You may not have permission, or the API returned an error."
            else:
                commit_status = "✅ README.md was successfully committed to the repository."

        # --- Always use LLM for final synthesis if agent mode was used ---
        synthesis_model = get_model_for_task("standard")
        # If markdown was generated, include it in the scratchpad context
        if generated_readme_content:
            agent_scratchpad += f"\n[Generated README.md]\n```markdown\n{generated_readme_content}\n```\n"
        else:
            scratchpad_blocks = re.findall(r"```markdown\s*([\s\S]+?)```", agent_scratchpad)
            if scratchpad_blocks:
                agent_scratchpad += f"\n[Generated README.md]\n```markdown\n{scratchpad_blocks[-1].strip()}\n```\n"
            else:
                last_obs_match = re.search(r"```markdown\s*([\s\S]+?)```", observation if 'observation' in locals() else "")
                if last_obs_match:
                    agent_scratchpad += f"\n[Generated README.md]\n```markdown\n{last_obs_match.group(1).strip()}\n```\n"
        synthesis_prompt = (
            f"You are an expert assistant. Here is the scratchpad of all steps taken:\n\n"
            f"{agent_scratchpad}\n\n"
            "Please synthesize a clear, final answer for the user, summarizing the outcome and next steps if needed.\n"
            "If any code (such as README.md) was generated, ALWAYS include it in your answer, even if errors occurred during commit or other actions.\n"
            "Present the code in a markdown block, and explain what the user should do next (e.g., copy the code manually if commit failed)."
        )
        final_response = synthesis_model.generate_content(synthesis_prompt)
        final_response_text = final_response.text
        # Ensure human-readable output if LLM returns JSON
        try:
            # Try to parse as JSON
            parsed = json.loads(final_response_text)
            if isinstance(parsed, dict) and 'answer' in parsed:
                final_response_text = parsed['answer']
        except Exception:
            pass

    elif mode_used == "MODE_1_DIRECT_DISPATCH":
        print(f"\n| ⚡️ EXECUTING DIRECT DISPATCH: {decision['TOOL']}")
        params_json = json.dumps(decision['PARAMS']) if isinstance(decision['PARAMS'], dict) else str(decision['PARAMS'])
        action_result = await execute_tool(f"{decision['TOOL']}({params_json})", chat_data, goal)
        synthesis_model = get_model_for_task("standard")
        synthesis_prompt = f"The user asked: '{user_message}'. The tool '{decision['TOOL']}' returned: {action_result}\n\nFormulate a natural language response based on this data."
        final_response = synthesis_model.generate_content(synthesis_prompt)
        final_response_text = final_response.text

        if decision['TOOL'] == "get_github_file":
            try:
                result_json = json.loads(action_result)
                file_content = result_json.get("content") or result_json.get("file_content") or ""
                import base64
                if result_json.get("encoding") == "base64":
                    file_content = base64.b64decode(file_content).decode("utf-8")
                final_response_text += f"\n\nHere is the content of `{decision['PARAMS']['file_path']}`:\n```html\n{file_content}\n```"
            except Exception as e:
                final_response_text += "\n\nError: Could not parse file content from the response."
    elif mode_used == "CONVERSATIONAL":
        conversational_model = get_model_for_task("standard")
        chat = conversational_model.start_chat(history=[m.model_dump() for m in chat_data.history[:-1]])
        response = chat.send_message(user_message)
        final_response_text = response.text

    # --- FINAL SYNTHESIS & CLEANUP ---
    if session_id and code_manager.get_code(session_id):
        final_code = code_manager.get_code(session_id)
        if 'Here is the complete modified code' not in final_response_text:
            final_response_text += f"\n\nHere is the complete modified code:\n```html\n{final_code}\n```"
        code_manager.cleanup_session(session_id)
        print(f"🧹 Session {session_id} cleaned up.")

    print(f"\n| 💬 FINAL RESPONSE TO USER:\n{final_response_text[:1000]}...")
    return {"role": "model", "parts": [final_response_text], "metadata": {"mode": mode_used}}

# WebSocket endpoint for plan updates
@router.websocket("/ws/plan-updates")
async def websocket_endpoint(websocket: WebSocket):
    await plan_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        plan_manager.disconnect(websocket)