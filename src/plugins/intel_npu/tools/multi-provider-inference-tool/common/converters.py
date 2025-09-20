import json

def shape_to_list(shape):
    ret = shape
    if isinstance(shape, (str, bytes)):
        ret = json.loads(shape)
    return ret

def layout_to_str(layout):
    ret = layout
    if isinstance(layout, list):
        ret = "".join(layout)
    return ret
