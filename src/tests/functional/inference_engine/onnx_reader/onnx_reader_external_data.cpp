// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <fstream>
#include <algorithm>

#include <ie_blob.h>
#include <ie_core.hpp>
#include <file_utils.h>
#include <streambuf>
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include <ngraph/ngraph.hpp>

TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromFile) {
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(CommonTestUtils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "onnx_external_data.onnx"), "");
    auto function = cnnNetwork.getFunction();

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    std::shared_ptr<ngraph::Node> external_data_node;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
        if (op_type == "Constant") {
            count_constants += 1;
            external_data_node = op;
        }
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 2);

    const auto external_data_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(external_data_node);
    ASSERT_TRUE(external_data_node_const->get_vector<float>() == (std::vector<float>{1, 2, 3, 4}));
}

TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromStringException) {
    InferenceEngine::Core ie;
    const auto path = CommonTestUtils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "onnx_external_data.onnx");
    InferenceEngine::Blob::CPtr weights; // not used
    std::ifstream stream(path, std::ios::binary);
    std::string modelAsString((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    stream.close();
    try {
        auto cnnNetwork = ie.ReadNetwork(modelAsString, weights);
    }
    catch(const ngraph::ngraph_error& e) {
        EXPECT_PRED_FORMAT2(
            testing::IsSubstring,
            std::string("invalid external data:"),
            e.what());

        EXPECT_PRED_FORMAT2(
            testing::IsSubstring,
            std::string("data/tensor.data, offset: 0, data_length: 0)"),
            e.what());
    }
    catch(...) {
        FAIL() << "Reading network failed for unexpected reason";
    }
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromWstringNamedFile) {
    InferenceEngine::Core ie;
    std::string win_dir_path = CommonTestUtils::getModelFromTestModelZoo(
        ONNX_TEST_MODELS "onnx_external_data.onnx");
    std::wstring wmodel = CommonTestUtils::addUnicodePostfixToPath(win_dir_path,
        CommonTestUtils::test_unicode_postfix_vector[0]);
    bool is_copy_successfully = CommonTestUtils::copyFile(win_dir_path, wmodel);
    if (!is_copy_successfully) {
        FAIL() << "Unable to copy from '" << win_dir_path << "' to '"
                << ov::util::wstring_to_string(wmodel) << "'";
    }

    auto cnnNetwork = ie.ReadNetwork(wmodel, L"");
    CommonTestUtils::removeFile(wmodel);
    auto function = cnnNetwork.getFunction();

    int count_add = 0;
    int count_constants = 0;
    int count_parameters = 0;

    std::shared_ptr<ngraph::Node> external_data_node;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_add += (op_type == "Add" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
        if (op_type == "Constant") {
            count_constants += 1;
            external_data_node = op;
        }
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_add, 2);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 2);

    const auto external_data_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(external_data_node);
    ASSERT_TRUE(external_data_node_const->get_vector<float>() == (std::vector<float>{1, 2, 3, 4}));
}
#endif
