import os


def check_in_github_actions():
    return os.getenv("GITHUB_ACTIONS") == "true"


def check_in_gitlab_ci():
    return os.getenv("GITLAB_CI") == "true"


def check_in_ci():
    return os.getenv("CI") == "true"
