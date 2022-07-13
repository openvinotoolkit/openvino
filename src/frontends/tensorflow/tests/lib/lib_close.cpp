// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lib_close.hpp"

#include <gtest/gtest.h>

#include "openvino/util/file_util.hpp"
#include "tf_utils.hpp"

using namespace testing;
using namespace ov::util;

INSTANTIATE_TEST_SUITE_P(Tensorflow,
                         FrontendLibCloseTest,
                         Values(std::make_tuple(TF_FE,
                                                path_join({std::string(TEST_TENSORFLOW_MODELS_DIRNAME),
                                                           std::string("2in_2out/2in_2out.pb")}),
                                                "Conv2D_1")),
                         FrontendLibCloseTest::get_test_case_name);
