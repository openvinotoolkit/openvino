import subprocess


def record_event(event_name: str, app_name: str, app_version: str, event_segments: dict):
    subprocess.run(["python3", "./mo/utils/record_lr_event.py", event_name, app_name, app_version,
                    list(event_segments.keys())[0], list(event_segments.values())[0]])


def record_file(event_name: str, app_name: str, app_version: str, file_path: str):
    subprocess.run(["python3", "./mo/utils/record_lr_event.py", event_name, app_name, app_version, file_path])
