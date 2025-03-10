// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "conversion_extension.hpp"
#include "gtest/gtest.h"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite::tests;

static std::string s_manifest = "";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_densify) {
    auto model = convert_model("densify.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>(Shape{1, 2, 3, 3}, {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 4}, {2, 1, 0, 0, 0, 3, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0});
    test_case.run();
}
