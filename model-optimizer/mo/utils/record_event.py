import subprocess


def record_event(event_name: str, app_name: str, app_version: str, event_segments: dict):
    subprocess.run(["python3", "record_lr_event.py", event_name, app_name, app_version,
                    event_segments.keys()[0], event_segments.values()[0]])
