# /app/skills/google_tasks.py

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Pydantic Models for API Data Validation ---
class TaskUpdate(BaseModel):
    status: Literal['completed', 'needsAction']

class TaskCreate(BaseModel):
    title: str

# --- FastAPI Router Setup ---
router = APIRouter(
    prefix="/tasks",
    tags=["Google Tasks"]
)

# --- Google API Helper Functions (The "Skill" Logic) ---
SCOPES = ['https://www.googleapis.com/auth/tasks']

def get_credentials(): # ... (no changes to this function)
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_tasks_service(): # ... (no changes to this function)
    credentials = get_credentials()
    try:
        service = build('tasks', 'v1', credentials=credentials)
        return service
    except HttpError: return None

def list_task_lists(service): # ... (no changes to this function)
    return service.tasklists().list(maxResults=20).execute().get('items', [])

def list_tasks(service, task_list_id): # ... (no changes to this function)
    return service.tasks().list(tasklist=task_list_id, showCompleted=False).execute().get('items', [])

def update_task(service, task_list_id, task_id, body): # ... (no changes to this function)
    return service.tasks().patch(tasklist=task_list_id, task=task_id, body=body).execute()

def create_task(service, task_list_id, body):
    """Creates a new task in the specified list."""
    try:
        return service.tasks().insert(tasklist=task_list_id, body=body).execute()
    except HttpError as err:
        print(f"An error occurred while creating the task: {err}")
        return None

# --- API Endpoints (Now part of the router) ---

@router.get("/lists", summary="Get all task lists")
async def get_all_task_lists():
    service = get_tasks_service()
    if not service: raise HTTPException(status_code=500, detail="Failed to connect to Google API.")
    task_lists = list_task_lists(service)
    if task_lists is None: raise HTTPException(status_code=404, detail="Could not fetch task lists.")
    return {"task_lists": task_lists}

@router.get("/{list_id}", summary="Get tasks for a specific list")
async def get_tasks_for_list(list_id: str):
    service = get_tasks_service()
    if not service: raise HTTPException(status_code=500, detail="Failed to connect to Google API.")
    tasks = list_tasks(service, list_id)
    return {"tasks": tasks if tasks is not None else []}

@router.post("/{list_id}", summary="Create a new task")
async def post_new_task(list_id: str, task_data: TaskCreate):
    service = get_tasks_service()
    if not service: raise HTTPException(status_code=500, detail="Failed to connect to Google API.")
    new_task = create_task(service, list_id, task_data.model_dump())
    if new_task is None: raise HTTPException(status_code=500, detail="Failed to create task.")
    return {"created_task": new_task}

@router.patch("/{list_id}/{task_id}", summary="Update a task's status")
async def patch_task_status(list_id: str, task_id: str, update_data: TaskUpdate):
    service = get_tasks_service()
    if not service: raise HTTPException(status_code=500, detail="Failed to connect to Google API.")
    updated_task = update_task(service, list_id, task_id, update_data.model_dump())
    if updated_task is None: raise HTTPException(status_code=500, detail="Failed to update task.")
    return {"updated_task": updated_task}