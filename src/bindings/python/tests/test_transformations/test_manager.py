# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize

from utils.utils import MyModelPass, PatternReplacement, expect_exception


def test_registration_and_pass_name():
    m = Manager()

    a = m.register_pass(PatternReplacement())
    a.set_name("PatterReplacement")

    b = m.register_pass(MyModelPass())
    b.set_name("ModelPass")

    c = m.register_pass(GraphRewrite())
    c.set_name("Anchor")

    d = c.add_matcher(PatternReplacement())
    d.set_name("PatterReplacement")

    e = m.register_pass(BackwardGraphRewrite())
    e.set_name("BackAnchor")

    f = e.add_matcher(PatternReplacement())
    f.set_name("PatterReplacement")

    PatternReplacement().set_name("PatternReplacement")
    MyModelPass().set_name("MyModelPass")
    GraphRewrite().set_name("Anchor")
    BackwardGraphRewrite().set_name("BackAnchor")

    # Preserve legacy behaviour when registered pass doesn't exist
    # and in this case we shouldn't throw an exception.
    m.register_pass("NotExistingPass")


def test_negative_pass_registration():
    m = Manager()
    expect_exception(lambda: m.register_pass(PatternReplacement))
    expect_exception(lambda: m.register_pass("PatternReplacement", PatternReplacement()))
    expect_exception(lambda: m.register_pass("Serialize", Serialize("out.xml", "out.bin")))
    expect_exception(lambda: m.register_pass("Serialize", "out.xml", "out.bin", "out.wrong"))
