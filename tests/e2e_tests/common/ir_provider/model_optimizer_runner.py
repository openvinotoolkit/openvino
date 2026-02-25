# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.cli_parser import parse_input_value
from openvino.tools.ovc.cli_parser import split_inputs

from e2e_tests.test_utils.test_utils import log_timestamp
from e2e_tests.test_utils.path_utils import resolve_file_path
from .provider import ClassProvider
import sys
import logging as log
import os


class OVCMORunner(ClassProvider):
    """OpenVINO converter runner."""
    __action_name__ = "get_ovc_model"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.target_ir_name = config.get("target_ir_name")
        self._config = config
        self.xml = None
        self.bin = None
        self.prepared_model = None  # dynamically set prepared model
        self.args = self._build_arguments()

    def _build_arguments(self):
        """Construct model optimizer arguments."""
        args = {
            'output_dir': self._config['mo_out'],
        }

        if self._config['precision'] == 'FP32':
            args['compress_to_fp16'] = False
        else:
            args['compress_to_fp16'] = True

        if self.target_ir_name is not None:
            args.update({"model_name": self.target_ir_name})
        # if isinstance(self._config['model'], str):
        #     if os.path.splitext(self._config['model'])[1] == ".meta":
        #         args["input_meta_graph"] = args.pop("input_model")
        #     # If our model not a regular file but directory then remove
        #     # '--input_model' attr and add use '--saved_model_dir'
        #     if os.path.isdir(self._config['model']):
        #         args["saved_model_dir"] = args.pop("input_model")

        if 'proto' in self._config.keys():
            args.update({"input_proto": str(self._config['proto'])})

        if 'fusing' in self._config.keys() and not self._config['fusing']:
            args.update({"disable_fusing": None})

        if "additional_args" in self._config:
            if 'tensorflow_object' in self._config['additional_args']:
                self._config['additional_args']['tensorflow_object_detection_api_pipeline_config'] = self._config[
                    'additional_args'].pop('tensorflow_object')

            for key, val in self._config["additional_args"].items():
                if key == 'batch':
                    val = int(val)
                args.update({key: val})

        return args

    def get_ir(self, passthrough_data):
        from openvino import convert_model, save_model
        from openvino.tools.mo.utils.cli_parser import input_shape_to_input_cut_info, input_to_input_cut_info

        ir_name = self.target_ir_name if self.target_ir_name else 'model'
        xml_file = os.path.join(self.args['output_dir'], ir_name + '.xml')
        bin_file = os.path.join(self.args['output_dir'], ir_name + '.bin')
        compress_to_fp16 = self.args.pop('compress_to_fp16')
        self.args.pop('output_dir')

        filtered_args = {}
        args_to_pop = []
        for k in self.args:
            if k in ['example_input', 'output']:
                filtered_args[k] = self.args[k]
            if k in ['saved_model_dir']:
                filtered_args['input_model'] = self.args['saved_model_dir']
                args_to_pop.append('saved_model_dir')
            if k in ['input_checkpoint']:
                filtered_args['input_model'] = self.args['input_checkpoint']
                args_to_pop.append('input_checkpoint')
            if k in ['input_meta_graph']:
                filtered_args['input_model'] = self.args['input_meta_graph']
                args_to_pop.append('input_meta_graph')

        if 'input' in self.args and 'input_shape' not in self.args:
            inputs = []
            for input_value in split_inputs(self.args['input']):
                # Parse string with parameters for single input
                node_name, shape, value, data_type = parse_input_value(input_value)
                inputs.append([attr for attr in [node_name, shape, value, data_type] if attr is not None])
            filtered_args['input'] = inputs
        elif 'input_shape' in self.args and 'input' not in self.args:
            if isinstance(self.args['input_shape'], str):
                _, shape, _, _ = parse_input_value(self.args['input_shape'])
                filtered_args['input'] = shape
            else:
                filtered_args['input'] = self.args['input_shape']
        elif 'input' in self.args and 'input_shape' in self.args:
            filtered_args['input'] = input_to_input_cut_info(self.args['input'])
            input_shape_to_input_cut_info(self.args['input_shape'], filtered_args['input'])
            for idx in range(len(filtered_args['input'])):
                if filtered_args['input'][idx].type:
                    filtered_args['input'][idx] = (filtered_args['input'][idx].name, filtered_args['input'][idx].shape,
                                                   filtered_args['input'][idx].type)
                else:
                    filtered_args['input'][idx] = (filtered_args['input'][idx].name, filtered_args['input'][idx].shape)

        for key in args_to_pop:
            self.args.pop(key)

        removed_keys = sorted(self.args.keys() - filtered_args.keys())
        log.info(f"Removed MO args: {removed_keys}")
        removed_values = [self.args[k] for k in removed_keys]
        log.info(f"Removed MO values: {removed_values}")

        with log_timestamp('Convert Model'):
            for k, v in filtered_args.items():
                if k == 'example_input':
                    v = True
                log.info(f'{k}={v}')

            ov_model = convert_model(self.prepared_model,
                                          input=filtered_args.get('input'),
                                          output=filtered_args.get('output'),
                                          example_input=filtered_args.get('example_input'),
                                          extension=filtered_args.get('extension'),
                                          verbose=filtered_args.get('verbose'),
                                          share_weights=filtered_args.get('share_weights', True))
            save_model(ov_model, xml_file, compress_to_fp16)

        self.xml = resolve_file_path(xml_file, as_str=True)
        self.bin = resolve_file_path(bin_file, as_str=True)
        log.info(f'XML file with compress_to_fp16={compress_to_fp16} was saved to: {self.xml}')
        log.info(f'BIN file with compress_to_fp16={compress_to_fp16} was saved to: {self.bin}')

        return self.xml, self.bin
