# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize

from tests.test_transformations.utils.utils import MyModelPass, PatternReplacement, expect_exception


def test_registration_and_pass_name():
    manager = Manager()

    pass_a = manager.register_pass(PatternReplacement())
    pass_a.set_name("PatterReplacement")

    pass_b = manager.register_pass(MyModelPass())
    pass_b.set_name("ModelPass")

    pass_c = manager.register_pass(GraphRewrite())
    pass_c.set_name("Anchor")

    pass_d = pass_c.add_matcher(PatternReplacement())
    pass_d.set_name("PatterReplacement")

    pass_e = manager.register_pass(BackwardGraphRewrite())
    pass_e.set_name("BackAnchor")

    pass_f = pass_e.add_matcher(PatternReplacement())
    pass_f.set_name("PatterReplacement")

    PatternReplacement().set_name("PatternReplacement")
    MyModelPass().set_name("MyModelPass")
    GraphRewrite().set_name("Anchor")
    BackwardGraphRewrite().set_name("BackAnchor")


def test_negative_pass_registration():
    manager = Manager()
    expect_exception(lambda: manager.register_pass(PatternReplacement))
    expect_exception(lambda: manager.register_pass("PatternReplacement", PatternReplacement()))
    expect_exception(lambda: manager.register_pass("Serialize", Serialize("out.xml", "out.bin")))
    expect_exception(lambda: manager.register_pass(Serialize("out.xml", "out.bin", "out.wrong")))
