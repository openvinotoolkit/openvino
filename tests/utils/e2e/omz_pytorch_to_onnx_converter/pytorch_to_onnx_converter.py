import os
import subprocess
import sys
import textwrap

from utils.e2e.omz_pytorch_to_onnx_converter.provider import ClassProvider
from utils.openvino_resources import OpenVINOResources
import logging as log


class PytorchToOnnxConverter(ClassProvider):
    """OMZ pytorch to onnx converter runner."""
    __action_name__ = "convert_pytorch_to_onnx"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self._config = config
        self.omz_pytorch_to_onnx_converter = OpenVINOResources().omz_pytorch_to_onnx_converter
        self.prepared_model = None

    def _prepare_command_line(self):
        cmd = [sys.executable, self.omz_pytorch_to_onnx_converter]
        args = {k: v for k, v in self._config.items() if v}

        for k, v in args.items():
            # model-param key should be a dict
            if k == 'model-param':
                for param, value in args[k].items():
                    cmd.append('--{}'.format(k))
                    cmd.append('{}={}'.format(param, value))
            else:
                cmd.append('--{}'.format(k))
                cmd.append(v)

        return [str(item) for item in cmd]

    def convert_pytorch_to_onnx(self, input_data):
        os.environ['TORCH_HOME'] = self._config.pop('torch_model_zoo_path')
        converter_timeout = self._config.pop('converter_timeout')

        cmd = self._prepare_command_line()
        log.info("Running OMZ model converter from Pytorch to ONNX:\n{}".format(
            textwrap.fill(" ".join(cmd), 180)))
        result = subprocess.run(cmd, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=converter_timeout)

        if result.stdout:
            log.info(result.stdout.decode('utf-8'))
        if result.stderr:
            log.error(result.stderr.decode('utf-8'))

        exit_code = result.returncode
        if exit_code != 0:
            raise RuntimeError(f"Pytorch to ONNX model conversion failed with {exit_code} exit code")

        # TODO: remove onnx path from arguments outside,
        # TODO: make it the same way as for _load_pytorch_model through 'self.prepared_model' var
