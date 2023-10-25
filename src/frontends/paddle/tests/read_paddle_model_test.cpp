// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <gtest/gtest.h>
#include <ie_blob.h>

#include <fstream>
#include <ie_core.hpp>
#include <set>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "frontend/shared/include/utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/serialize.hpp"

TEST(Paddle_Reader_Tests, ImportBasicModelToCore) {
    auto model = std::string(TEST_PADDLE_MODELS_DIRNAME) + "relu/relu.pdmodel";

    ov::Core core;
    auto function = core.read_model(FrontEndTestUtils::make_model_path(model));

    const auto inputType = ov::element::f32;
    const auto inputShape = ov::Shape{3};
    const auto data = std::make_shared<ov::opset1::Parameter>(inputType, inputShape);
    data->set_friendly_name("x");
    data->output(0).get_tensor().add_names({"x"});
    const auto relu = std::make_shared<ov::opset1::Relu>(data->output(0));
    relu->set_friendly_name("relu_0.tmp_0");
    relu->output(0).get_tensor().add_names({"relu_0.tmp_0"});
    const auto result = std::make_shared<ov::opset1::Result>(relu->output(0));
    result->set_friendly_name("relu_0.tmp_0/Result");
    const auto reference = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{data}, "Model0");
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::NAMES);
    const FunctionsComparator::Result res = func_comparator(function, reference);
    ASSERT_TRUE(res.valid) << res.message;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
TEST(Paddle_Reader_Tests, ImportBasicModelToCoreWstring) {
    std::string win_dir_path{TEST_PADDLE_MODELS_DIRNAME "relu/relu.pdmodel"};
    win_dir_path = FrontEndTestUtils::make_model_path(win_dir_path);
    std::wstring wmodel =
        ov::test::utils::addUnicodePostfixToPath(win_dir_path, ov::test::utils::test_unicode_postfix_vector[0]);
    bool is_copy_successfully = ov::test::utils::copyFile(win_dir_path, wmodel);
    if (!is_copy_successfully) {
        FAIL() << "Unable to copy from '" << win_dir_path << "' to '" << ov::util::wstring_to_string(wmodel) << "'";
    }

    ov::Core core;
    auto function = core.read_model(wmodel);
    ov::test::utils::removeFile(wmodel);

    const auto inputType = ov::element::f32;
    const auto inputShape = ov::Shape{3};
    const auto data = std::make_shared<ov::opset1::Parameter>(inputType, inputShape);
    data->set_friendly_name("x");
    data->output(0).get_tensor().add_names({"x"});
    const auto relu = std::make_shared<ov::opset1::Relu>(data->output(0));
    relu->set_friendly_name("relu_0.tmp_0");
    relu->output(0).get_tensor().add_names({"relu_0.tmp_0"});
    const auto result = std::make_shared<ov::opset1::Result>(relu->output(0));
    result->set_friendly_name("relu_0.tmp_0/Result");
    const auto reference = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{data}, "Model0");
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::NAMES);
    const FunctionsComparator::Result res = func_comparator(function, reference);
    ASSERT_TRUE(res.valid) << res.message;
}
#endif
