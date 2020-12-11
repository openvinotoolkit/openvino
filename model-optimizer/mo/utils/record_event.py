from mo.utils.laternrock  import LanternRock

def record_event(event_name: str, app_name: str, app_version: str, event_segments: dict):
    """Record & upload event for Intel analytics server

        :arg event_name: name of the event

        :arg app_version: string with name of app that created the event

        :arg app_version: string with version of app that created the event

        :arg event_segments: dictionary with key-pairs that will be send to the server

    """

    tid = '48296c3d-eace-4c6b-bc96-e24d27310678'
    lr = LanternRock()
    lr.InitializeEx(app_name=app_name, app_version=app_version, telemetryid=tid)
    lr.RecordEventEx(None, event_name, count=1, sum=1.0, datadict=event_segments)
    lr.Deinitialize()
    lr.Upload(telemetryid=tid, options={'show': False})
