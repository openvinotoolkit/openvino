// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lib_close.hpp"

#include <gtest/gtest.h>

#include "onnx_utils.hpp"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;

INSTANTIATE_TEST_SUITE_P(ONNX,
                         FrontendLibCloseTest,
                         Values(std::make_tuple(ONNX_FE,
                                                path_join({std::string(TEST_ONNX_MODELS_DIRNAME),
                                                           std::string("external_data/external_data.onnx")}),
                                                "Y")),
                         FrontendLibCloseTest::get_test_case_name);
