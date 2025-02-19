# utils/file_handler.py
import os

def file_path(relative_path: str) -> str:
    """
    프로젝트 루트를 기준으로 경로를 반환합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 프로젝트 루트 디렉토리
    return os.path.join(project_root, relative_path)
