
# WARNING: These functions implements a Vigenere cypher, they are NO WAY OK FOR SECURITY!
CHARS = [c for c in (chr(i) for i in range(32, 127))]


def _ascii_key(key):
    key = "".join([it for it in key if it in CHARS])
    assert len(key), "Provide a key containing ASCII characters"
    return key


def encode(text, key):
    assert isinstance(text, str), "Expected string type, got '{}'".format(type(text))
    assert isinstance(key, str), "Expected 'str' type, got '{}'".format(type(key))
    key = _ascii_key(key)
    res = ""
    for i, c in enumerate(text):
        if c not in CHARS:
            res += c
        else:
            text_index = CHARS.index(c)
            key_index = CHARS.index(key[i % len(key)])
            res += CHARS[(text_index + key_index) % len(CHARS)]
    return res


def decode(text, key):
    assert isinstance(text, str), "Expected 'bytes', got '{}'".format(type(text))
    assert isinstance(key, str), "Expected 'str' type, got '{}'".format(type(key))
    key = _ascii_key(key)
    res = ""
    for i, c in enumerate(text):
        if c not in CHARS:
            res += c
        else:
            text_index = CHARS.index(c)
            key_index = CHARS.index(key[i % len(key)])
            res += CHARS[(text_index - key_index) % len(CHARS)]
    return res
