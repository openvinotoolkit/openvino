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
import glob
import itertools
import shutil
import sys
import csv
import re
import pytest
from glob import iglob
import numpy as np
from pathlib import Path
import requests 
import zipfile

import logging as log
from common.common_utils import shell
from distutils import spawn

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


def search_model_path_recursively(config_key, model_name):
    search_pattern = config_key + '/**/' + model_name
    path_found = list(iglob(search_pattern, recursive=True))
    if len(path_found) == 1:
        return path_found[0]
    elif len(path_found) == 0:
        raise FileNotFoundError("File not found for pattern {}".format(search_pattern))
    else:
        raise ValueError("More than one file with {} name".format(model_name))


class Environment:
    """
    Environment used by tests.

    :attr env:  environment dictionary. populated dynamically from environment
                configuration file.
    """
    env = {}

    @classmethod
    def abs_path(cls, env_key, *paths):
        """Construct absolute path by appending paths to environment value.

        :param cls: class
        :param env_key: Environment.env key used to get the base path
        :param paths:   paths to be appended to Environment.env value
        :return:    absolute path string where Environment.env[env_key] is
                    appended with paths
        """
        return str(Path(cls.env[env_key], *paths))


def get_tests(cmd_params, use_device=True, use_batch=False):
    # Several keys:
    #         use_device
    #         use_batch
    # you should specify 'False' when sample not required '-d' or '-b' keys.
    # Example: use_device=False for all 'hello_*" samples
    #          use batch = False for all samples except for hello_shape_infer_ssd
    new_cmd_params = []
    cmd_keys = list(cmd_params.keys())

    devices = os.environ["TEST_DEVICE"].split(';') if os.environ.get("TEST_DEVICE") else ["CPU"]

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

def getting_samples_data_zip(url, samples_path, size_of_chunk=128):
    if os.path.exists(samples_path) or os.path.exists(samples_path[:-4]):
        return		
    try:
        print("\nStart downloading samples_smoke_tests_data.zip...")
        samples_request = requests.get(url, stream=True)
        with open(samples_path, 'wb') as samples_file:
            for elem in samples_request.iter_content(chunk_size=size_of_chunk):
                samples_file.write(elem)
        print("\nsamples_smoke_tests_data.zip downloaded successfully")
        samples_file.close()
        print("\nExtracting of samples_smoke_tests_data.zip...")
        with zipfile.ZipFile(samples_path, 'r') as samples_zip:
            samples_zip.extractall(Environment.env['smoke_tests_path'])
        nameFolder = str(Environment.env['samples_data_zip'])[Environment.env['samples_data_zip'].rfind('/')+1:][:-4]
        smoke_tests_path = os.path.join(Environment.env['smoke_tests_path'])
        if os.path.exists(os.path.join(smoke_tests_path,nameFolder)):
            os.rename(os.path.join(smoke_tests_path, nameFolder), os.path.join(smoke_tests_path, 'samples_smoke_tests_data') )
        if os.path.exists(samples_path):
            print("\nRemoving samples_smoke_tests_data.zip...")
            os.remove(samples_path)	

    except Exception as error:
        print(error)
        print(f"Exception during downloading samples_smoke_tests_data.zip")

class SamplesCommonTestClass():

    @classmethod
    def made_executable_path(cls, path1, path2, sample_type='C++'):
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
            executable_path = spawn.find_executable(str('benchmark_app'))
        # if not hasattr(cls, 'executable_path'):
        cls.executable_path = executable_path

    @staticmethod
    def reset_models_path(model):
        pathList = model.split(os.sep)
        modelName = pathList[len(pathList)-1]
        precision = pathList[len(pathList)-2]
        for root, subFolder, files in os.walk(Environment.env['models_path']):
            for item in files:
                if item.endswith(modelName) :
                    if precision in root :
                        model = str(os.path.join(root,item))
                    else :
                        model = os.path.join(Environment.env['models_path'], model)
        return model

    @staticmethod
    def join_env_path(param, executable_path, complete_path=True):
        gpu_lib_path = os.path.join(os.environ.get('IE_APP_PATH'), 'lib')
        if 'i' in param:
            # If batch > 1, then concatenate images
            if ' ' in param['i']:
                param['i'] = param['i'].split(' ')
            elif complete_path:
                param['i'] = list([param['i']])
        for k in param.keys():
            if ('i' == k) and complete_path:
                param['i'] = [os.path.join(Environment.env['test_data'], e) for e in param['i']]
                param['i'] = ' '.join(param['i'])
            elif ('ref_m' == k):
                param['ref_m'] = SamplesCommonTestClass.reset_models_path(param['ref_m'])
            elif ('m' == k):
                param['m'] = SamplesCommonTestClass.reset_models_path(param['m'])
            elif ('m_ag' == k):
                param['m_ag'] = SamplesCommonTestClass.reset_models_path(param['m_ag'])
            elif ('m_hp' == k):
                param['m_hp'] = SamplesCommonTestClass.reset_models_path(param['m_hp'])
            elif ('m_va' == k):
                param['m_va'] = SamplesCommonTestClass.reset_models_path(param['m_va'])
            elif ('m_lpr' == k):
                param['m_lpr'] = SamplesCommonTestClass.reset_models_path(param['m_lpr'])
            elif ('m_em' == k):
                param['m_em'] = SamplesCommonTestClass.reset_models_path(param['m_em'])
            elif ('m_pa' == k):
                param['m_pa'] = SamplesCommonTestClass.reset_models_path(param['m_pa'])
            elif ('m_reid' == k):
                param['m_reid'] = SamplesCommonTestClass.reset_models_path(param['m_reid'])
            elif ('m_fd' == k):
                param['m_fd'] = SamplesCommonTestClass.reset_models_path(param['m_fd'])
            elif ('m_act' == k):
                param['m_act'] = SamplesCommonTestClass.reset_models_path(param['m_act'])
            elif ('m_lm' == k):
                param['m_lm'] = SamplesCommonTestClass.reset_models_path(param['m_lm'])
            elif ('m_det' == k):
                param['m_det'] = SamplesCommonTestClass.reset_models_path(param['m_det'])
            elif ('m_td' == k):
                param['m_td'] = SamplesCommonTestClass.reset_models_path(param['m_td'])
            elif ('m_tr' == k):
                param['m_tr'] = SamplesCommonTestClass.reset_models_path(param['m_tr'])
            elif ('m_en' == k):
                param['m_en'] = SamplesCommonTestClass.reset_models_path(param['m_en'])
            elif ('m_de' == k):
                param['m_de'] = SamplesCommonTestClass.reset_models_path(param['m_de'])
            elif ('l' == k and 'pascal_voc_classes' in param['l']):
                param['l'] = os.path.join(Environment.env['test_data'], param['l'])
            elif ('pp' == k):
                param['pp'] = gpu_lib_path
            elif ('r' == k) and complete_path:
                if len(param['r']) > 0:
                    param['r'] = os.path.join(Environment.env['test_data'], param['r'])
            elif ('o' == k) and complete_path:
                param['o'] = os.path.join(Environment.env['out_directory'], param['o'])
            elif ('wg' == k):
                param['wg'] = os.path.join(Environment.env['out_directory'], param['wg'])
            elif ('we' == k):
                param['we'] = os.path.join(Environment.env['out_directory'], param['we'])
            elif ('fg' == k):
                param['fg'] = os.path.join(Environment.env['test_data'], param['fg'])
            elif ('labels' == k):
                label_folder = os.path.dirname(executable_path.split(' ')[-1])
                param['labels'] = os.path.join(label_folder, param['labels'])
            elif ('lb') == k:
                param['lb'] = os.path.join(Environment.env['test_data'], param['lb'])

    @staticmethod
    def get_cmd_line(param, use_preffix=True, long_hyphen=None):
        if long_hyphen is None:
            long_hyphen = []
        line = ''
        for key in sorted(param.keys()):
            if use_preffix and any([x for x in long_hyphen if key == x]):
                line += '--{} {} '.format(key, param[key])
            elif use_preffix and key not in long_hyphen:
                line += '-{} {} '.format(key, param[key])
            elif not use_preffix:
                line += '{} '.format(param[key])
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
    def get_empty_cmd_line(param, use_preffix=True, long_hyphen=None):
        line = ''
        return line

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

    @classmethod
    def setup_class(cls):
        getting_samples_data_zip(Environment.env['samples_data_zip'], Environment.env['samples_path'])
        assert os.environ.get('IE_APP_PATH') is not None, "IE_APP_PATH environment variable is not specified!"
        assert os.path.exists(Environment.env['models_path']), \
            "Path for public models {} is not exist!".format(Environment.env['models_path'])
        assert os.path.exists(Environment.env['test_data']), \
            "Path for test data {} is not exist!".format(Environment.env['test_data'])
        cls.output_dir = Environment.env['out_directory']

    def _test(self, param, use_preffix=True, get_cmd_func=None, get_shell_result=False, long_hyphen=None, complete_path=True):
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
            assert os.environ.get('IE_APP_PYTHON_PATH') is not None, \
                "IE_APP_PYTHON_PATH environment variable is not specified!"
            self.made_executable_path(os.environ.get('IE_APP_PYTHON_PATH'), self.sample_name,
                                      sample_type=sample_type)
        else:
            self.made_executable_path(os.environ.get('IE_APP_PATH'), self.sample_name, sample_type=sample_type)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if 'bitstream' in param_cp:
            del param_cp['bitstream']

        if 'precision' in param_cp:
            del param_cp['precision']

        if get_cmd_func is None:
            get_cmd_func = self.get_cmd_line

        self.join_env_path(param_cp, executable_path=self.executable_path, complete_path=complete_path)

        # Updating all attributes in the original dictionary (param), because param_cp was changes (join_env_path)
        for key in param.keys():
            if key in param_cp:
                param[key] = param_cp[key]

        if 'sample_type' in param_cp:
            del param_cp['sample_type']

        cmd_line = get_cmd_func(param_cp, use_preffix=use_preffix, long_hyphen=long_hyphen)

        log.info("Running command: {} {}".format(self.executable_path, cmd_line))
        retcode, stdout, stderr = shell([self.executable_path, cmd_line])

        # Execute performance:
        if Environment.env['performance'] and retcode == 0:
            perf_iter = int(Environment.env['performance'])
            # Check if samples are for performance testing: if FPS in output
            is_perf = self.check_is_perf(stdout.split('\n'))
            is_niter = self.check_has_niter(param_cp)
            if not is_perf:
                # Skipping all tests for this sample, because no of them are ready for performance.
                # Add name of sample to global pytest variable, then skip it in setup method
                if 'list_of_skipped_samples' in Environment.env:
                    Environment.env['list_of_skipped_samples'].append(self.sample_name)
                else:
                    Environment.env.update({'list_of_skipped_samples': [self.sample_name]})
                pytest.skip('[INFO] Sample {} not executed for performance'.format(self.executable_path))
            else:
                log.info('Running perfomance for {} iteraction'.format(perf_iter))
                # Perf_iter = 0 when it isn't neccessary to add niter key
                if perf_iter > 0:
                    if is_niter:
                        log.warning('Changed value of niter param to {}'.format(perf_iter))
                        param_cp['niter'] = perf_iter
                    else:
                        log.warning('Added key: niter to param with value: {}'.format(perf_iter))
                        param_cp.update({'niter': perf_iter})
                cmd_perf = get_cmd_func(param_cp, use_preffix=use_preffix, long_hyphen=long_hyphen)
                retcode_perf, stdout_perf, stderr_perf = shell([self.executable_path, cmd_perf])
                if (retcode_perf != 0):
                    log.error(stderr_perf)
                assert retcode_perf == 0, "Execution sample for performance failed"
                fps_perf = self.find_fps(stdout_perf)
                self.write_csv(sample_name=self.sample_name, sample_type=sample_type, cmd_perf=cmd_perf, fps_perf=fps_perf)
                log.info('Perf results: {}'.format(fps_perf))
        if get_shell_result:
            return retcode, stdout, stderr
        # Check return code
        if (retcode != 0):
            log.error(stderr)
        assert retcode == 0, "Sample execution failed"     
        return stdout

    def setup_method(self):
        """
        Clean up IRs and npy files from self.output_dir if exist
        And skip several test for performance
        :return: """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        filenames = glob.glob('out*.bmp')
        [os.remove(fn) for fn in filenames]
        # Skip samples that are not for performance:
        if Environment.env['performance'] and 'list_of_skipped_samples' in Environment.env and \
                self.sample_name in Environment.env['list_of_skipped_samples']:
            pytest.skip('[Skip from setup] Sample {} not executed for performance'.format(self.sample_name))

    def teardown_method(self):
        """
        Clean up IRs and npy files from self.output_dir if exist
        :return: """
        is_save = getattr(self, 'save', None) 
        if not is_save and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        filenames = glob.glob('out*.bmp')
        [os.remove(fn) for fn in filenames]
