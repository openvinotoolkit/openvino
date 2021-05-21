# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager, FrontEndCapabilities

from pybind_mock_frontend import get_stat, reset_stat, FeStat

# Shall be initialized and destroyed after every test finished
# This is required as destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()

def test_load_by_framework_caps():
    frontEnds = fem.get_available_front_ends()
    assert frontEnds is not None
    print("FrontEnds: {}".format(frontEnds))
    assert 'mock_py' in frontEnds
    caps = [FrontEndCapabilities.DEFAULT,
            FrontEndCapabilities.CUT,
            FrontEndCapabilities.NAMES,
            FrontEndCapabilities.WILDCARDS,
            FrontEndCapabilities.CUT | FrontEndCapabilities.NAMES | FrontEndCapabilities.WILDCARDS]
    for cap in caps:
        fe = fem.load_by_framework(framework="mock_py", capabilities=cap)
        stat = get_stat(fe)
        assert cap == stat.load_flags


def test_load_from_file():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("abc.bin")
    stat = get_stat(fe)
    assert 'abc.bin' in stat.loaded_paths


def test_convert_model():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("")
    fe.convert(model)
    stat = get_stat(fe)
    assert stat.convertModelCount == 1


def test_convert_partially():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("")
    func = fe.convert_partially(model)
    stat = get_stat(fe)
    assert stat.convertPartCount == 1
    fe.convert(func)
    stat = get_stat(fe)
    assert stat.convertFuncCount == 1


def test_decode_and_normalize():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    stat = get_stat(fe)
    assert stat.normalizeCount == 0
    assert stat.decodeCount == 0
    model = fe.load_from_file("")
    func = fe.decode(model)
    stat = get_stat(fe)
    assert stat.normalizeCount == 0
    assert stat.decodeCount == 1
    fe.normalize(func)
    stat = get_stat(fe)
    assert stat.normalizeCount == 1
    assert stat.decodeCount == 1

#--------InputModel tests-----------------



# if __name__ == '__main__':
#     test_frontendmanager()