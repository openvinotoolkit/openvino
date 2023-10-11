import re
import fileinput


def replace(cfg, commit):
    prepCfg = cfg["runConfig"]["preprocess"]
    filePath = prepCfg["file"]
    pattern = prepCfg["pattern"]
    replacement = ''
    for line in fileinput.input(filePath, inplace=True):
        newLine = re.sub(pattern, r'{}'.format(replacement), line, flags=0)
        print(newLine, end='')
