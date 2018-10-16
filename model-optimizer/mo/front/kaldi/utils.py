import struct


def get_uint16(s):
    return struct.unpack('H', s)[0]


def get_uint32(s):
    return struct.unpack('I', s)[0]


class KaldiNode:
    def __init__(self, name):
        self.name = name
        self.blobs = [None, None]

    def set_weight(self, w):
        self.blobs[0] = w

    def set_bias(self, b):
        self.blobs[1] = b

    def set_attrs(self, attrs: dict):
        for k, v in attrs.items():
            setattr(self, k, v)
