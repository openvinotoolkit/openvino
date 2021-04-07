# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.tf.extractors.identity import tf_identity_ext
from mo.utils.unittest.extractors import BaseExtractorsTestingClass


class IdentityExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.identity.copy_shape_infer'

    def test_identity(self):
        self.expected = {}
        self.res = tf_identity_ext(pb=None)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()
