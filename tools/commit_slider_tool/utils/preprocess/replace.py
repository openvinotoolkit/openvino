import re
def replace(cfg):
    prepCfg = cfg["specialConfig"]["preprocess"]
    filePath = prepCfg["file"]
    pattern = prepCfg["pattern"]
    replacement = prepCfg["replacement"]
    with open (filePath, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub(pattern, r'{}'.format(replacement), content, flags = re.M)
        f.seek(0)
        f.write(contentNew)
        f.close()
    