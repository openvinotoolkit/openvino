import os
import sys
import logging as log
from openvino.runtime import Core

support_device_list = ['CPU', 'GPU', 'MYRIAD']

def get_xml_file(dir, model_list):
    if os.path.isfile(dir):
        if os.path.splitext(dir)[-1] == '.xml':
            model_list.append(dir)
    else:
        for item in os.listdir(dir):
            aim_name = os.path.join(os.path.abspath(dir),item)
            if os.path.isfile(aim_name):
                if os.path.splitext(aim_name)[-1] == '.xml':
                    model_list.append(aim_name)
            else:
                get_xml_file(aim_name, model_list)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # Parsing and validation of input arguments
    if len(sys.argv) != 3:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>')
        return 1

    model_list_path = sys.argv[1]
    device_name = sys.argv[2]
    model_list = []
    get_xml_file(model_list_path, model_list)
    log.info(f'There will be {len(model_list)} model tested')
    if device_name not in support_device_list:
        log.error('Sample supports only CPU, GPU, MYRIAD')

    for model_path in model_list:
        ie = Core()
        # read model to device
        log.info(f'Reading the model: {model_path}')
        model = ie.read_model(model=model_path)
        try:
            compiled_model = ie.compile_model(model=model, device_name=device_name)
        except Exception as e:
            log.error("%s not support in device '%s': %s %s" % (model_path, device_name, e.__class__.__name__, e))
            log.error("Compile model error")
            continue
        log.info("Compile model successfully")

        original_op_names = []
        for node in model.get_ops():
            original_op_names.append(node.get_friendly_name())

        layers_dict = ie.query_model(model=model, device_name=device_name)
        query_op_names = list(layers_dict.keys())
        query_status = True
        for op_name in original_op_names:
            if op_name not in query_op_names:
                log.error("%s not support in device '%s': Node %s was not assigned on any pointed device" % (model_path, device_name, op_name))
                query_status = False
                break
            if layers_dict[op_name] != device_name:
                log.error("%s were not able to be assigned on any pointed device" % (op_name))
                query_status = False
                break
        if query_status == True:
            log.info("Query model successfully")
        else:
            log.error("Query model error")

    return 0

if __name__ == "__main__" :
    sys.exit(main())
