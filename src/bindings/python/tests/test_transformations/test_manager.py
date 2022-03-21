# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize

from utils.utils import *


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


print("START")
m = Manager()
m.register_pass("ConstantFolding")
m.register_pass("Serialize", "out.xml", "out.bin")
print("END")