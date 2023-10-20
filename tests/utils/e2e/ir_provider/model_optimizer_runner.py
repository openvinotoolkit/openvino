import time

from openvino.tools.mo.utils.cli_parser import parse_input_value
from openvino.tools.ovc.cli_parser import split_inputs

from e2e_oss.utils.test_utils import log_timestamp
from e2e_oss.utils.path_utils import resolve_file_path
from pathlib import Path
from .provider import ClassProvider
import subprocess
import sys
import logging as log
import textwrap
import os
import re


class MORunner(ClassProvider):
    """Model optimizer runner."""
    __action_name__ = "mo"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.mo = config["mo_runner"]
        self.target_ir_name = config.get("target_ir_name")
        self._config = config
        self.ir_gen_time = None
        self.mem_usage_mo = None

        self.xml = None
        self.bin = None
        self.mo_log = None
        self.prepared_model = None  # dynamically set prepared model
        self.use_cmd_tool = config.get('use_mo_cmd_tool', False)
        self.args = self._build_mo_arguments() if self.use_cmd_tool else self._build_mo_arguments_for_python_api()

    def _build_mo_arguments(self):
        """Construct model optimizer arguments."""
        args = {
            '--input_model': self._config['model'],
            '--output_dir': self._config['mo_out']
        }
        if self.target_ir_name is not None:
            args.update({"--model_name": self.target_ir_name})
        if os.path.splitext(self._config['model'])[1] == ".meta":
            args["--input_meta_graph"] = args.pop("--input_model")

        # If our model not a regular file but directory then remove '--input_model' attr and add use '--saved_model_dir'
        if os.path.isdir(self._config['model']):
            args["--saved_model_dir"] = args.pop("--input_model")

        if 'proto' in self._config.keys():
            args.update({"--input_proto": str(self._config['proto'])})

        if 'fusing' in self._config.keys() and not self._config['fusing']:
            args.update({"--disable_fusing": None})

        if self._config['precision'] == 'FP32':
            args['--compress_to_fp16'] = False
        else:
            args['--compress_to_fp16'] = True

        if "additional_args" in self._config:
            for key, val in self._config["additional_args"].items():
                # IF some key has boolean value it will be treated as argument
                # without value ('store_true' action)
                if key == 'example_input':
                    continue
                if not key.startswith('--'):
                    key = '--' + key
                args.update({
                                key: None
                            } if isinstance(val, bool) else {key: val})

        return args

    def _build_mo_arguments_for_python_api(self):
        """Construct model optimizer arguments."""
        args = {
            'input_model': self._config['model'],
            'output_dir': self._config['mo_out'],
        }

        if self._config['precision'] == 'FP32':
            args['compress_to_fp16'] = False
        else:
            args['compress_to_fp16'] = True

        if self.target_ir_name is not None:
            args.update({"model_name": self.target_ir_name})
        if isinstance(self._config['model'], str):
            if os.path.splitext(self._config['model'])[1] == ".meta":
                args["input_meta_graph"] = args.pop("input_model")
            # If our model not a regular file but directory then remove
            # '--input_model' attr and add use '--saved_model_dir'
            if os.path.isdir(self._config['model']):
                args["saved_model_dir"] = args.pop("input_model")

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

    def _prepare_command_line(self):
        """Construct subprocess style command-line."""
        cmd = [self.mo]
        if Path(self.mo).suffix == ".py":
            cmd.insert(0, sys.executable)

        for k, v in self.args.items():
            cmd.append(k)
            if v is not None:
                cmd.append(v)
        return [str(item) for item in cmd]

    def _apply_input_shape_arg(self, input_data):
        """Apply input_shape argument to Model Optimizer based on input data shape"""
        shape_arg = ""
        in_names_arg = ""
        for in_name, data in input_data.items():
            shape_arg = shape_arg + "["
            in_names_arg = in_names_arg + in_name + ","
            for dim_count, dim in enumerate(data.shape):
                shape_arg = shape_arg + str(dim)
                if dim_count < len(data.shape) - 1:
                    shape_arg = shape_arg + ","
            shape_arg = shape_arg + "],"
        shape_arg = shape_arg[:-1]
        in_names_arg = in_names_arg[:-1]
        self.args.update({"--input_shape": shape_arg})
        self.args.update({"--input": in_names_arg})
        # batch and input shape cannot be provided in the same time
        if "--batch" in self.args.keys():
            del self.args["--batch"]

    def _get_ir_through_tool(self, input_data):
        """Generate IR for Inference Engine. Dynamically called by framework at
        the point of 'get_ir' step processing.
        """
        if self._config.get('use_input_data_shape', False) and input_data is not None:
            self._apply_input_shape_arg(input_data)
        command_line = self._prepare_command_line()
        with log_timestamp('Model Optimizer'):
            log.info("Running Model Optimizer:\n{}".format(
                textwrap.fill(" ".join(command_line), 180)))
            ir_gen_time_start = time.time()
            result = subprocess.run(
                command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ir_gen_time = time.time() - ir_gen_time_start
        if result.stdout:
            log.info(result.stdout.decode('utf-8'))
        if result.stderr:
            log.error(result.stderr.decode('utf-8'))

        if self.target_ir_name is not None:
            model_base_name = self.target_ir_name
        else:
            model_base_name = os.path.basename(os.path.splitext(self._config["model"])[0])

        log_file_path = os.path.join(self._config['mo_out'], model_base_name + ".mo_log.txt")
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "w+") as log_file:
            log_file.write(" ".join(command_line) + "\n")
            log_file.write(result.stdout.decode('utf-8') + "\n")
            log_file.write(result.stderr.decode('utf-8') + "\n")
        self.mo_log = resolve_file_path(log_file_path, as_str=True)

        exit_code = result.returncode
        if exit_code != 0:
            raise RuntimeError(f"IR generation failed with {exit_code} exit code")

        mem_usage_mo_match = re.search(r'Memory consumed: (.*) MB.', result.stdout.decode('utf-8'))
        if mem_usage_mo_match is not None:
            mem_usage_mo = mem_usage_mo_match.group(1)
        else:
            mem_usage_mo = -1
        self.ir_gen_time = ir_gen_time
        self.mem_usage_mo = mem_usage_mo
        xml = re.search(r'\[ SUCCESS \] XML file: (.*)', result.stdout.decode('utf-8')).group(1).replace("\r", "")
        bin = re.search(r'\[ SUCCESS \] BIN file: (.*)', result.stdout.decode('utf-8')).group(1).replace("\r", "")
        self.xml = resolve_file_path(xml, as_str=True)
        self.bin = resolve_file_path(bin, as_str=True)
        return self.xml, self.bin

    def _get_ir_through_python(self, input_data):
        """Generate IR for Inference Engine. Dynamically called by framework at
        the point of 'get_ir' step processing.
        """
        from openvino.tools.mo import convert_model
        from openvino.runtime import serialize

        if self._config.get('use_input_data_shape', False) and input_data is not None:
            self._apply_input_shape_arg(input_data)

        self.args.update({'silent': False})
        # Load from memory case
        if self.prepared_model:
            self.args['input_model'] = self.prepared_model
        with log_timestamp('Model Optimizer'):
            log.info("Running Model Optimizer:\n")
            for k, v in self.args.items():
                if k == 'input_model' and self.prepared_model:
                    v = str(v.__class__)
                if k == 'example_input':
                    v = True
                log.info(f'{k}={v}')

            self.ov_model = convert_model(**self.args)
            ir_name = self.target_ir_name if self.target_ir_name else 'model'
            xml = os.path.join(self.args['output_dir'], ir_name + '.xml')
            bin = os.path.join(self.args['output_dir'], ir_name + '.bin')
            serialize(self.ov_model, xml, bin)
        self.xml = resolve_file_path(xml, as_str=True)
        self.bin = resolve_file_path(bin, as_str=True)

        log.info(f'XML file was serialized to: {self.xml}')
        log.info(f'BIN file was serialized to: {self.bin}')

        return self.xml, self.bin

    def get_ir(self, input_data=None):
        if self.use_cmd_tool:
            return self._get_ir_through_tool(input_data)
        else:
            return self._get_ir_through_python(input_data)


class OVCMORunner(MORunner):
    """OpenVINO converter runner."""
    __action_name__ = "get_ovc_model"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def get_ir(self, input_data=None):
        from openvino import convert_model, save_model
        from openvino.tools.mo.utils.cli_parser import input_shape_to_input_cut_info, input_to_input_cut_info

        ir_name = self.target_ir_name if self.target_ir_name else 'model'
        xml = os.path.join(self.args['output_dir'], ir_name + '.xml')
        bin = os.path.join(self.args['output_dir'], ir_name + '.bin')
        compress_to_fp16 = self.args.pop('compress_to_fp16')
        self.args.pop('output_dir')

        if self.prepared_model:
            self.args['input_model'] = self.prepared_model

        filtered_args = {}
        args_to_pop = []
        for k in self.args:
            if k in ['input_model', 'example_input', 'output']:
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
                if k == 'input_model' and self.prepared_model:
                    v = str(v.__class__)
                if k == 'example_input':
                    v = True
                log.info(f'{k}={v}')

            self.ov_model = convert_model(filtered_args['input_model'],
                                          input=filtered_args.get('input'),
                                          output=filtered_args.get('output'),
                                          example_input=filtered_args.get('example_input'),
                                          extension=filtered_args.get('extension'),
                                          verbose=filtered_args.get('verbose'),
                                          share_weights=filtered_args.get('share_weights', True))
            save_model(self.ov_model, xml, compress_to_fp16)

        self.xml = resolve_file_path(xml, as_str=True)
        self.bin = resolve_file_path(bin, as_str=True)
        log.info(f'XML file with compress_to_fp16={compress_to_fp16} was saved to: {self.xml}')
        log.info(f'BIN file with compress_to_fp16={compress_to_fp16} was saved to: {self.bin}')

        return self.xml, self.bin
