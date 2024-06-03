import subprocess


def get_current_git_commit(repo_path="."):
    try:
        # Run the git rev-parse command to get the current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Return the commit hash
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving git commit hash: {e}")
        return None


def is_git_repo_dirty(repo_path="."):
    try:
        # Run the git status command
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # If there is any output, the repo is dirty
        if result.stdout:
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error checking git status: {e}")
        return False


def get_git_remote_url(repo_path="."):
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving git remote URL: {e}")
        return None


def get_current_commit_url(repo_path="."):
    commit_hash = get_current_git_commit(repo_path)
    remote_url = get_git_remote_url(repo_path)

    if not commit_hash or not remote_url:
        return None

    # Handle different URL formats
    if remote_url.startswith("git@"):
        remote_url = remote_url.replace(":", "/").replace("git@", "https://")
    elif remote_url.startswith("https://") or remote_url.startswith("http://"):
        remote_url = remote_url.replace(".git", "")

    commit_url = f"{remote_url}/commit/{commit_hash}"
    return commit_url


def get_changed_files():
    try:
        # Run the git command to get the list of changed files
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )

        # Split the result by lines
        lines = result.stdout.strip().split("\n")

        # Extract the file paths from the lines
        changed_files = [
            line.strip().split(" ", 1)[1] for line in lines if len(line) > 3
        ]

        return changed_files

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git: {e}")
        return []
