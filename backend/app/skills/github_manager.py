# /app/skills/github_manager.py

import os
from fastapi import APIRouter, HTTPException
from github import Github, GithubException
from pydantic import BaseModel

# --- NEW: Pydantic Model for the request body ---
class FileUpdate(BaseModel):
    commit_message: str
    new_content: str

# --- FastAPI Router Setup ---
router = APIRouter(
    prefix="/github",
    tags=["GitHub"]
)

# --- GitHub API Helper Functions ---
def get_github_service():
    """Initializes and returns an authenticated PyGithub instance."""
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        raise HTTPException(status_code=500, detail="GITHUB_PAT not found in .env file.")
    return Github(pat)

# --- API Endpoints ---

@router.get("/repos", summary="Get authenticated user's repositories")
async def get_user_repos():
    try:
        g = get_github_service()
        user = g.get_user()
        repos = user.get_repos(sort="pushed", direction="desc")
        
        repo_list = [
            {"name": repo.name, "full_name": repo.full_name, "description": repo.description, "url": repo.html_url}
            for repo in repos
        ]
        return {"repos": repo_list}
    except GithubException as e:
        raise HTTPException(status_code=500, detail=f"GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/repos/{owner}/{repo_name}/commits", summary="Get recent commits for a repository")
async def get_repo_commits(owner: str, repo_name: str):
    try:
        g = get_github_service()
        repo = g.get_repo(f"{owner}/{repo_name}")
        commits = repo.get_commits()
        
        commit_list = []
        for i, commit in enumerate(commits):
            if i >= 10:
                break
            commit_list.append({
                "sha": commit.sha,
                "message": commit.commit.message,
                "author": commit.commit.author.name,
                "date": commit.commit.author.date.isoformat(),
                "url": commit.html_url,
            })
        return {"commits": commit_list}
    except GithubException as e:
        raise HTTPException(status_code=404, detail=f"Repository not found or GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: Endpoint to list repository contents ---
@router.get("/repos/{owner}/{repo_name}/contents", summary="List repository contents")
async def get_repo_contents(owner: str, repo_name: str):
    """
    Lists the contents of the root directory of a repository.
    """
    try:
        g = get_github_service()
        repo = g.get_repo(f"{owner}/{repo_name}")
        contents = repo.get_contents("")
        
        file_list = [item.path for item in contents]
        return {"files": file_list}
        
    except GithubException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Repository not found: {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/repos/{owner}/{repo_name}/tree", summary="List all repository files recursively")
async def get_repo_tree(owner: str, repo_name: str):
    """
    Lists all files in the repository recursively using the Git Trees API.
    """
    try:
        g = get_github_service()
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        # Get the SHA for the latest commit on the default branch
        latest_commit_sha = repo.get_commits()[0].sha
        
        # Get the tree recursively
        tree = repo.get_git_tree(latest_commit_sha, recursive=True)
        
        # We only want to return files ("blobs"), not folders ("trees")
        file_list = [item.path for item in tree.tree if item.type == 'blob']
        
        return {"files": file_list}
        
    except GithubException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Repository not found: {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@router.get("/repos/{owner}/{repo_name}/contents/{file_path:path}", summary="Get file content")
async def get_repo_file_content(owner: str, repo_name: str, file_path: str):
    """
    Fetches the decoded content of a specific file from a repository.
    The :path converter allows file_path to contain slashes.
    """
    try:
        g = get_github_service()
        repo = g.get_repo(f"{owner}/{repo_name}")
        content = repo.get_contents(file_path)
        
        decoded_content = content.decoded_content.decode('utf-8')
        return {"content": decoded_content, "file_path": file_path, "repo": repo.full_name}
        
    except GithubException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"File or repository not found: {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/repos/{owner}/{repo_name}/contents/{file_path:path}", summary="Update or create a file and commit")
async def update_github_file(owner: str, repo_name: str, file_path: str, update_data: FileUpdate):
    """
    Updates the content of a specific file and creates a commit.
    If the file does not exist, creates it.
    """
    try:
        g = get_github_service()
        repo = g.get_repo(f"{owner}/{repo_name}")
        try:
            file_contents = repo.get_contents(file_path)
            update_result = repo.update_file(
                path=file_path,
                message=update_data.commit_message,
                content=update_data.new_content,
                sha=file_contents.sha
            )
        except GithubException as e:
            if e.status == 404:
                # File does not exist, create it
                update_result = repo.create_file(
                    path=file_path,
                    message=update_data.commit_message,
                    content=update_data.new_content
                )
            else:
                raise e
        return {
            "detail": "File updated or created successfully.",
            "commit_sha": update_result['commit'].sha,
            "file_path": file_path
        }
    except GithubException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"File or repository not found: {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub API error: {e.data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))