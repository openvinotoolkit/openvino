// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_model.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;
using namespace ov::frontend::tensorflow::tests;

using TFConvertModelTest = FrontEndConvertModelTest;

static const std::vector<std::string> models{
    std::string("2in_2out/2in_2out.pb"),
    std::string("2in_2out/2in_2out.pb.frozen"),
    std::string("2in_2out/2in_2out.pb.frozen_text"),
    std::string("forward_edge_model/forward_edge_model.pbtxt"),
    std::string("forward_edge_model2/forward_edge_model2.pbtxt"),
    std::string("concat_with_non_constant_axis/concat_with_non_constant_axis.pbtxt"),
    std::string("gather_tree_model/gather_tree_model.pbtxt")};

INSTANTIATE_TEST_SUITE_P(TFConvertModelTest,
                         FrontEndConvertModelTest,
                         ::testing::Combine(::testing::Values(TF_FE),
                                            ::testing::Values(std::string(TEST_TENSORFLOW_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndConvertModelTest::getTestCaseName);
