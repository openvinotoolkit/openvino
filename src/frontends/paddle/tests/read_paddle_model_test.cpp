// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <openvino/util/file_util.hpp>
#include <set>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "frontend/shared/include/utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"

TEST(Paddle_Reader_Tests, LoadModelMemoryToCore) {
    auto model =
        FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) + "conv2d_relu/conv2d_relu.pdmodel");
    auto param = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                    "conv2d_relu/conv2d_relu.pdiparams");

    ov::Core core;
    auto read_file = [&](const std::string& file_name, size_t& size) {
        FILE* sFile = fopen(file_name.c_str(), "r");
        fseek(sFile, 0, SEEK_END);
        size = ftell(sFile);
        uint8_t* ss = (uint8_t*)malloc(size);
        rewind(sFile);
        const size_t length = fread(&ss[0], 1, size, sFile);
        if (size != length) {
            std::cerr << "file size is not correct\n";
        }
        fclose(sFile);
        return ss;
    };

    size_t xml_size, bin_size;
    auto xml_ptr = read_file(model, xml_size);
    auto bin_ptr = read_file(param, bin_size);
    ov::Tensor weight_tensor = ov::Tensor(ov::element::u8, {1, bin_size}, bin_ptr);
    std::string model_str = std::string((char*)xml_ptr, xml_size);
    auto function = core.read_model(model_str, weight_tensor);

    const auto inputType = ov::element::f32;
    const auto inputShape = ov::Shape{1, 3, 4, 4};
    const auto data = std::make_shared<ov::opset1::Parameter>(inputType, inputShape);
    data->set_friendly_name("xxx");
    data->output(0).get_tensor().add_names({"xxx"});
    const auto weight = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{5, 3, 1, 1}, 1.0);
    const auto conv2d = std::make_shared<ov::opset1::Convolution>(data->output(0),
                                                                  weight->output(0),
                                                                  ov::Strides({1, 1}),
                                                                  ov::CoordinateDiff({1, 1}),
                                                                  ov::CoordinateDiff({1, 1}),
                                                                  ov::Strides({1, 1}));
    conv2d->set_friendly_name("conv2d_0.tmp_0");
    conv2d->output(0).get_tensor().add_names({"conv2d_0.tmp_0"});
    const auto relu = std::make_shared<ov::opset1::Relu>(conv2d->output(0));
    relu->set_friendly_name("relu_0.tmp_0");
    relu->output(0).get_tensor().add_names({"relu_0.tmp_0"});
    const auto bias = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{}, 0.0);
    const auto scale = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{}, 1.0);
    const auto mul = std::make_shared<ov::opset1::Multiply>(relu->output(0), scale);
    const auto add = std::make_shared<ov::opset1::Add>(mul->output(0), bias);
    add->set_friendly_name("scale_0.tmp_0");
    add->output(0).get_tensor().add_names({"save_infer_model/scale_0.tmp_0"});

    const auto result = std::make_shared<ov::opset1::Result>(add->output(0));
    result->set_friendly_name("save_infer_model/scale_0.tmp_0/Result");

    const auto reference = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{data}, "Model0");
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::NONE);
    const FunctionsComparator::Result res = func_comparator(function, reference);
    ASSERT_TRUE(res.valid) << res.message;
    free(xml_ptr);
    free(bin_ptr);
}

TEST(Paddle_Reader_Tests, ImportBasicModelToCore) {
    auto model = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) + "relu/relu.pdmodel");

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
    const auto bias = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{}, 0.0);
    const auto scale = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{}, 1.0);
    const auto mul = std::make_shared<ov::opset1::Multiply>(relu->output(0), scale);
    const auto add = std::make_shared<ov::opset1::Add>(mul->output(0), bias);
    add->set_friendly_name("save_infer_model/scale_0.tmp_0");
    add->output(0).get_tensor().add_names({"save_infer_model/scale_0.tmp_0"});

    const auto result = std::make_shared<ov::opset1::Result>(add->output(0));
    result->set_friendly_name("save_infer_model/scale_0.tmp_0/Result");

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
