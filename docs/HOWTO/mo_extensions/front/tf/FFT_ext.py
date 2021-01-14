"""
 Copyright (C) 2018-2020 Intel Corporation

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

# ! [fft_ext:extractor]
from ...ops.FFT import FFT
from mo.front.extractor import FrontExtractorOp
from mo.utils.error import Error


class FFT2DFrontExtractor(FrontExtractorOp):
    op = 'FFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 0
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT2DFrontExtractor(FrontExtractorOp):
    op = 'IFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 1
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled
# ! [fft_ext:extractor]
