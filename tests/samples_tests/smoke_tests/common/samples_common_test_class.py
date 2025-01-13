"""
 Copyright (C) 2018-2025 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import cv2
import contextlib
import io
import os
import itertools
import re
import subprocess
import sys
import requests
import time
import numpy as np
import zipfile

import logging as log
from common.common_utils import shell
from shutil import which
import openvino.runtime as ov

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

def get_devices():
    return os.environ.get("TEST_DEVICE", "CPU;MULTI:CPU;AUTO").split(';')


def get_cmd_output(*cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=60.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(f"'{' '.join(map(str, cmd))}' returned {error.returncode}. Output:")
        else:
            print(f"'{' '.join(map(str, cmd))}' timed out after {error.timeout} seconds. Output:")
        print(error.output)
        raise
    return output


def download(test_data_dir, file_path):
    if file_path.exists():
        return file_path
    lock_path = test_data_dir / 'download.lock'
    with contextlib.suppress(FileNotFoundError, PermissionError):
        lock_path.unlink()
    for _ in range(9999):  # Give up after about 3 hours
        with contextlib.suppress(FileExistsError, PermissionError):
            with lock_path.open('bx'):
                if not file_path.exists():
                    if test_data_dir / 'bvlcalexnet-12.onnx' == file_path:
                        response = requests.get("https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12.onnx?download=")
                        with file_path.open('wb') as nfnet:
                            nfnet.write(response.content)
                    elif test_data_dir / 'efficientnet-lite4-11-qdq.onnx' == file_path:
                        response = requests.get("https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11-qdq.onnx?download=")
                        with file_path.open('wb') as nfnet:
                            nfnet.write(response.content)
                    else:
                        response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino/ci_dependencies/test/2021.4/samples_smoke_tests_data_2021.4.zip")
                        with zipfile.ZipFile(io.BytesIO(response.content)) as zfile:
                            zfile.extractall(test_data_dir)
                        cv2.imwrite(str(test_data_dir / 'dog-224x224.bmp'), cv2.resize(cv2.imread(str(test_data_dir / 'samples_smoke_tests_data_2021.4/validation_set/227x227/dog.bmp')), (224, 224)))
            lock_path.unlink(missing_ok=True)
            assert file_path.exists()
            return file_path
        time.sleep(1.0)


def prepend(cache, inp='', model='', tmp_path=None):
    test_data_dir = cache.mkdir('test_data')
    if model:
        if type(model) is ov.ie_api.Model:
            model_sv_path = tmp_path / "model_with_4bit_input.xml"
            ov.save_model(model, model_sv_path)
            model = '-m', model_sv_path
        else:
            model = '-m', download(test_data_dir, test_data_dir / model)

    if inp:
        if os.path.exists(tmp_path / inp):
            inp = '-i', tmp_path / inp
        else:
            inp = '-i', download(test_data_dir, test_data_dir / inp)
        return *inp, *model
    else:
        return *model,


def get_tests(cmd_params, use_device=True, use_batch=False):
    # Several keys:
    #         use_device
    #         use_batch
    # you should specify 'False' when sample not required '-d' or '-b' keys.
    # Example: use_device=False for all 'hello_*" samples
    #          use batch = False for all samples except for hello_shape_infer_ssd
    new_cmd_params = []
    cmd_keys = list(cmd_params.keys())

    devices = os.environ.get("TEST_DEVICE", "CPU;MULTI:CPU;AUTO").split(';')

    # You can pass keys (like d, d_lpr ..) via use_device list. And the topology executes only on these devices
    # Use this option when a topology isn't supported in some plugin. In default CPU only.
    if isinstance(use_device, list):
        for dev_key in use_device:
            dev_list = np.array(cmd_params[dev_key])
            for _dev in dev_list:
                if not _dev in devices and _dev in cmd_params[dev_key]:
                    cmd_params[dev_key].remove(_dev)
        use_device = False

    for it in itertools.product(*[cmd_params[key] for key in cmd_params.keys()]):
        test = {}
        for key in zip(cmd_keys, it):
            test.update({key[0]: key[1]})

        # Fill array with images accordong batch size
        if 'batch' in test and use_batch == False:
            images = ""
            for i in range(test['batch']):
                images += test['i'] + " "
            images = images.rstrip()
            test['i'] = images
            # Remove batch attr
            del test['batch']

        # Delete bitstream param:
        if 'd' in test and 'bitstream' in test:
            del test['bitstream']
        # Add new tests params
        new_cmd_params.append(test)

    test_args = []

    for i in range(len(new_cmd_params)):
        # key "use_device" is to run sample with device, exception: helo_classification_sample
        if use_device:
            for d in devices:
                new_d = {}
                new_d.update(new_cmd_params[i])
                new_d.update({'d': d})
                test_args.append(new_d)
        else:
            test_args.append(new_cmd_params[i])

    return test_args


class SamplesCommonTestClass():

    @classmethod
    def made_executable_path(cls, path1, path2, sample_type):
        if hasattr(cls, 'executable_path'):
            return cls.executable_path

        executable_path = os.path.join(path1, path2, path2) if 'python' in sample_type.lower() \
            else os.path.join(path1, path2)
        is_windows = sys.platform.startswith('win')
        if 'python' in sample_type.lower():
            executable_path += '.py'
            if is_windows: 
                executable_path = 'python ' + executable_path
            else:
                executable_path = 'python3 ' + executable_path
        elif 'c' in sample_type.lower() and not 'c++' in sample_type.lower():
            executable_path += '_c'
        if is_windows and not 'python' in sample_type.lower():
            executable_path += '.exe'

        # This exeption is made for benchmark_app, because it locates in another place.
        if 'benchmark_app' in path2 and 'python' in sample_type.lower():
            executable_path = which(str('benchmark_app'))
        return executable_path

    @staticmethod
    def join_env_path(param, cache, executable_path, complete_path=True):
        test_data_dir = cache.mkdir('test_data')
        unpacked = test_data_dir / 'samples_smoke_tests_data_2021.4'
        if 'i' in param:
            # If batch > 1, then concatenate images
            if ' ' in param['i']:
                param['i'] = param['i'].split(' ')
            elif complete_path:
                param['i'] = list([param['i']])
        for k in param.keys():
            if ('i' == k) and complete_path:
                param['i'] = [str(download(test_data_dir, test_data_dir / e)) for e in param['i']]
                param['i'] = ' '.join(map(str, param['i']))
            elif 'm' == k and not param['m'].endswith('/samples/cpp/model_creation_sample/lenet.bin"'):
                param['m'] = download(test_data_dir, test_data_dir / param['m'])

    @staticmethod
    def get_cmd_line(param, use_preffix=True, long_hyphen=None):
        if long_hyphen is None:
            long_hyphen = []
        line = ''
        for key, value in param.items():
            if use_preffix and any([x for x in long_hyphen if key == x]):
                line += '--{} {} '.format(key, value)
            elif use_preffix and key not in long_hyphen:
                line += '-{} {} '.format(key, value)
            elif not use_preffix:
                line += '{} '.format(value)
        return line

    @staticmethod
    def find_fps(stdout):
        stdout = stdout.split('\n')
        for line in stdout:
            if 'fps' in line.lower():
                return float(re.findall(r"\d+\.\d+", line)[0])

    @staticmethod
    def get_hello_cmd_line(param, use_preffix=True, long_hyphen=None):
        line = ''
        for key in ['m', 'i', 'd']:
            if key in param:
                if use_preffix:
                    line += '-{} {} '.format(key, param[key])
                else:
                    line += '{} '.format(param[key])
        return line

    @staticmethod
    def get_hello_shape_cmd_line(param, use_preffix=True, long_hyphen=None):
        line = ''
        for key in ['m', 'i', 'd', 'batch']:
            if key in param:
                if use_preffix:
                    line += '-{} {} '.format(key, param[key])
                else:
                    line += '{} '.format(param[key])
        return line

    @staticmethod
    def get_hello_nv12_cmd_line(param, use_preffix=True, long_hyphen=None):
        line = ''
        for key in ['m', 'i', 'size', 'd']:
            if key in param:
                if use_preffix:
                    line += '-{} {} '.format(key, param[key])
                else:
                    line += '{} '.format(param[key])
        return line

    def _test(self, param, cache, use_preffix=True, get_cmd_func=None, get_shell_result=False, long_hyphen=None, complete_path=True):
        """
        :param param:
        :param use_preffix: use it when sample doesn't require keys (i.e. hello_classification <path_to_model> <path_to_image>
        instead of hello_classification -m  <path_to_model> -i <path_to_image>)
        :param get_cmd_func: to use specific cmd concatenate function, again for hello_request_classification sample
        :param get_shell_result: to return the result of sample running (retcode, strout, stderr) directly, \
                                 without failing inside _test function. Needed for negative test cases checking \
                                 (e.g. error messages validation)
        :param long_hyphen: to concatenate cmd param with '--', instead of '-', example: instance_segmentation_demo --labels
        :return:
        """
        # Copy param to another variable, because it is need to save original parameters without changes
        param_cp = dict(param)
        sample_type = param_cp.get('sample_type', "C++")
        if 'python' in sample_type.lower():
            executable_path = self.made_executable_path(os.environ['IE_APP_PYTHON_PATH'], self.sample_name,
                                      sample_type=sample_type)
        else:
            executable_path = self.made_executable_path(os.environ['IE_APP_PATH'], self.sample_name, sample_type=sample_type)

        if 'bitstream' in param_cp:
            del param_cp['bitstream']

        if 'precision' in param_cp:
            del param_cp['precision']

        if get_cmd_func is None:
            get_cmd_func = self.get_cmd_line

        self.join_env_path(param_cp, cache, executable_path=executable_path, complete_path=complete_path)

        # Updating all attributes in the original dictionary (param), because param_cp was changes (join_env_path)
        for key in param.keys():
            if key in param_cp:
                param[key] = param_cp[key]

        if 'sample_type' in param_cp:
            del param_cp['sample_type']

        cmd_line = get_cmd_func(param_cp, use_preffix=use_preffix, long_hyphen=long_hyphen)

        log.info("Running command: {} {}".format(executable_path, cmd_line))
        retcode, stdout, stderr = shell([executable_path, cmd_line])

        if get_shell_result:
            return retcode, stdout, stderr
        # Check return code
        if (retcode != 0):
            log.error(stderr)
        assert retcode == 0, "Sample execution failed"     
        return stdout
