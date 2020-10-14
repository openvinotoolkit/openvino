from fnmatch import fnmatch

from .detectron2.retinanet import RetinaNet

classes = [
    RetinaNet()
]

def register_model_hook(model):
    class_name = str(model.__class__)
    for cl in classes:
        if fnmatch(class_name, '*{}*'.format(cl.class_name)):
            cl.register_hook(model)
