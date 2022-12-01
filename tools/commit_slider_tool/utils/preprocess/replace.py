import re
import fileinput
def replace(cfg):
    prepCfg = cfg["specialConfig"]["preprocess"]
    filePath = prepCfg["file"]
    pattern = prepCfg["pattern"]
    # filePath = '/home/openvino-ci-35/Documents/ov/openvino/inference-engine/tests/functional/plugin/cpu/shared_tests_instances/skip_tests_config.cpp'
    # pattern = '(\s)*(.)*(SetScalePreProcessGetBlob)(.)*(\s)'
    replacement = ''
    for line in fileinput.input(filePath, inplace=True):
        newLine = re.sub(pattern, r'{}'.format(replacement), line, flags = 0)
        print(newLine, end='')