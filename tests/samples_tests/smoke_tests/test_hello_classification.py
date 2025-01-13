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
import os
import pathlib
import pytest
import re
import sys
import logging as log
from common.samples_common_test_class import get_cmd_output, get_tests, download
from common.samples_common_test_class import SamplesCommonTestClass
from common.common_utils import shell
from pathlib import Path
import shutil


def get_executable(sample_language):
    return pathlib.Path(os.environ['IE_APP_PATH'], 'hello_classification' + ('' if sample_language == 'C++' else '_c')).with_suffix('.exe' if os.name == 'nt' else '')


def prepend(cache, model, inp):
    test_data_dir = cache.mkdir('test_data')
    return download(test_data_dir, test_data_dir / model), download(test_data_dir, test_data_dir / inp)


test_data_fp32 = get_tests({
    'i': ['dog-224x224.bmp'],
    'm': ['bvlcalexnet-12.onnx'],  # Remove the model forom .md and .rst if removed from here
    'sample_type': ['C++', 'C'],
})

test_data_fp32_unicode = get_tests({
    'i': ['dog-224x224.bmp'],
    'm': ['bvlcalexnet-12.onnx'],
    'sample_type': ['C++'],
})


class Test_hello_classification(SamplesCommonTestClass):
    sample_name = 'hello_classification'

    @pytest.mark.parametrize("param", test_data_fp32)
    def test_hello_classification_fp32(self, param, cache):
        """
        Classification_sample_async has functional and accuracy tests.
        Also check ASCII path support.
        For accuracy find in output class of detected on image object
        """

        # Run _test function, that returns stdout or 0.
        output = get_cmd_output(get_executable(param['sample_type']), *prepend(cache, param['m'], param['i']), param['d'])

        output = output.split('\n')

        is_ok = True
        for line in range(len(output)):
            if re.match('\\d+ +\\d+.\\d+$', output[line].replace('[ INFO ]', '').strip()) is not None:
                top1 = output[line].replace('[ INFO ]', '').strip().split(' ')[0]
                top1 = re.sub('\\D', '', top1)
                if '215' not in top1:
                    is_ok = False
                    log.error('Expected class 262, Detected class {}'.format(top1))
                break
        assert is_ok, 'Wrong top1 class'
        log.info('Accuracy passed')

    @pytest.mark.parametrize("param", test_data_fp32_unicode)
    def test_hello_classification_check_unicode_path_support(self, param, cache, tmp_path):
        """
        Check UNICODE characters in paths.
        """
        #  Make temporary dirs, prepare temporary input data and temporary model
        if sys.platform.startswith("win"):  #issue 71298 need fix, then add condition: and param.get('sample_type') == "C":
            pytest.skip("C sample doesn't support unicode paths on Windows")

        tmp_image_dir = tmp_path / 'image'
        tmp_model_dir = tmp_path / 'model'

        tmp_image_dir.mkdir()
        tmp_model_dir.mkdir()

        test_data_dir = cache.makedir('test_data')
        # Copy files
        shutil.copy(test_data_dir / param['i'], tmp_image_dir)
        shutil.copy(test_data_dir / param['m'], tmp_model_dir)

        image_path = tmp_image_dir / Path(param['i']).name
        original_image_name = image_path.name.split(sep='.')[0]

        model_path = tmp_model_dir / Path(param['m']).name
        original_model_name = model_path.name.split(sep='.')[0]

        # List of encoded words
        # All b'...' lines are disabled by default. If you want to use theme check 'Testing' block (see below)
        encoded_words = [b'\xd1\x80\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9',  # russian
                         b'\xd7\xa2\xd7\x91\xd7\xa8\xd7\x99\xd7\xaa',  # hebrew
                         b'\xc4\x8desky',  # cesky
                         b'\xe4\xb8\xad\xe5\x9b\xbd\xe4\xba\xba',  # chinese
                         b'\xed\x95\x9c\xea\xb5\xad\xec\x9d\xb8',  # korean
                         b'\xe6\x97\xa5\xe6\x9c\xac\xe4\xba\xba',  # japanese
                         #  all
                         b'\xd1\x80\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9_\xd7\xa2\xd7\x91\xd7\xa8\xd7\x99\xd7\xaa_\xc4\x8desky_\xe4\xb8\xad\xe5\x9b\xbd\xe4\xba\xba_\xed\x95\x9c\xea\xb5\xad\xec\x9d\xb8_\xe6\x97\xa5\xe6\x9c\xac\xe4\xba\xba']

        # Reference run
        log.info("Reference run ...")
        executable_path = self.made_executable_path(os.environ.get('IE_APP_PATH'), self.sample_name, sample_type=param.get('sample_type'))
        cmd_line = f"{model_path} {image_path} {param.get('d', 'C++')}"
        ref_retcode, ref_stdout, ref_stderr = shell([executable_path, cmd_line])

        if ref_retcode != 0:
            log.error("Reference run FAILED with error:")
            log.error(ref_stderr)
            raise AssertionError("Sample execution failed!")
        log.info(ref_stdout)

        ref_probs = []
        for line in ref_stdout.split(sep='\n'):
            if re.match(r"\\d+\\s+\\d+.\\d+", line):
                prob_class = int(line.split()[0])
                prob = float(line.split()[1])
                ref_probs.append((prob_class, prob))

        for image_name in [encoded_words[-1]]:
            for model_name in [encoded_words[-1]]:

                new_image_path = tmp_image_dir / (original_image_name + f"_{image_name.decode('utf-8')}{image_path.suffix}")
                image_path.rename(new_image_path)
                image_path = new_image_path

                new_model_path = tmp_model_dir / (original_model_name + f"_{model_name.decode('utf-8')}.xml")
                model_path.rename(new_model_path)
                stdout = get_cmd_output(executable_path, new_model_path, image_path, param['d'])
                probs = []
                for line in stdout.split(sep='\n'):
                    if re.match(r"^\\d+\\s+\\d+.\\d+", line):
                        prob_class = int(line.split()[0])
                        prob = float(line.split()[1])
                        probs.append((prob_class, prob))
                assert ref_probs == probs
