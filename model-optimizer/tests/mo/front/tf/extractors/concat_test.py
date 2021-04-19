# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.tf.extractors.concat import tf_concat_ext
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class ConcatExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.concat.concat_infer'

    def test_concat(self):
        pb = PB({'attr': {
            'N': PB({'i': 3}),
        }})
        self.expected = {
            'type': 'Concat',
            'N': 3,
        }
        self.res = tf_concat_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None)
        self.compare()
