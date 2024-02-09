"""
 Copyright (C) 2018-2023 Intel Corporation
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
import os
import itertools
import subprocess
import sys
import csv
import re
import pytest
import numpy as np

import logging as log
from common.common_utils import shell
from shutil import which

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

def get_devices():
    return set(os.environ.get("TEST_DEVICE", "CPU;MULTI:CPU;AUTO").split(';'))


def get_cmd_output(*cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(f"""command '{' '.join(map(str, cmd))}'
exited with code {error.returncode}. Output:
{error.output}""")
        else:
            print(f"""command '{' '.join(map(str, cmd))}'
timed out after {error.timeout} seconds. Output:
{error.output}""")
        raise
    return output


def prepend(cache, input='', model=''):
    test_data_dir = cache.mkdir('test_data_dir') / 'samples_smoke_tests_data_2021.4'
    if input:
        input = test_data_dir / 'validation_set' / input
        assert input.exists()
        input = '-i', input
    if model:
        model = test_data_dir / 'models' / 'public' / model
        assert model.exists()
        model = '-m', model
    return *input, *model


class Environment:
    """
    Environment used by tests.

    :attr env:  environment dictionary. populated dynamically from environment
                configuration file.
    """
    env = {}


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
    def made_executable_path(cls, path1, path2, sample_type='C++'):
        if hasattr(cls, 'executable_path'):
            return

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
        cls.executable_path = executable_path

    @staticmethod
    def join_env_path(param, cache, executable_path, complete_path=True):
        test_data_dir = cache.makedir('test_data_dir') / 'samples_smoke_tests_data_2021.4'
        inputs = test_data_dir / 'validation_set'
        models = test_data_dir / 'models' / 'public'
        if 'i' in param:
            # If batch > 1, then concatenate images
            if ' ' in param['i']:
                param['i'] = param['i'].split(' ')
            elif complete_path:
                param['i'] = list([param['i']])
        for k in param.keys():
            if ('i' == k) and complete_path:
                param['i'] = [inputs / e for e in param['i']]
                param['i'] = ' '.join(map(str, param['i']))
            elif 'm' == k and not param['m'].endswith('/samples/cpp/model_creation_sample/lenet.bin"'):
                param['m'] = models / param['m']

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
    def check_is_perf(stdout):
        # This function check if FPS in stdout. If yes - then need to run this sample for perfomance
        for line in stdout:
            if 'fps' in line.lower():
                return True
        return False

    @staticmethod
    def check_has_niter(param):
        # Check if niter has already in params, so it was set before
        if 'niter' in param:
            return True
        return False

    @staticmethod
    def find_fps(stdout):
        stdout = stdout.split('\n')
        for line in stdout:
            if 'fps' in line.lower():
                return float(re.findall(r"\d+\.\d+", line)[0])

    @staticmethod
    def write_csv(sample_name, sample_type, cmd_perf, fps_perf):
        csv_path = Environment.env['perf_csv_name']
        with open(csv_path, 'a', newline='') as f:
            perf_writer = csv.writer(f, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            perf_writer.writerow([sample_name, sample_type, cmd_perf.rstrip(), fps_perf])

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
            self.made_executable_path(os.environ['IE_APP_PYTHON_PATH'], self.sample_name,
                                      sample_type=sample_type)
        else:
            self.made_executable_path(os.environ['IE_APP_PATH'], self.sample_name, sample_type=sample_type)

        if 'bitstream' in param_cp:
            del param_cp['bitstream']

        if 'precision' in param_cp:
            del param_cp['precision']

        if get_cmd_func is None:
            get_cmd_func = self.get_cmd_line

        self.join_env_path(param_cp, cache, executable_path=self.executable_path, complete_path=complete_path)

        # Updating all attributes in the original dictionary (param), because param_cp was changes (join_env_path)
        for key in param.keys():
            if key in param_cp:
                param[key] = param_cp[key]

        if 'sample_type' in param_cp:
            del param_cp['sample_type']

        cmd_line = get_cmd_func(param_cp, use_preffix=use_preffix, long_hyphen=long_hyphen)

        log.info("Running command: {} {}".format(self.executable_path, cmd_line))
        retcode, stdout, stderr = shell([self.executable_path, cmd_line])

        if get_shell_result:
            return retcode, stdout, stderr
        # Check return code
        if (retcode != 0):
            log.error(stderr)
        assert retcode == 0, "Sample execution failed"     
        return stdout
