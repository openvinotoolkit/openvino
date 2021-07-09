// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <fstream>

#include <ie_blob.h>
#include <ie_core.hpp>
#include <file_utils.h>
#include <ngraph/ngraph.hpp>

TEST(PDPD_Reader_Tests, ImportBasicModelToCore) {
    auto model = std::string(PDPD_TEST_MODELS) + "relu.pdmodel";
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(model);
    auto function = cnnNetwork.getFunction();

    int count_relus = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_relus += (op_type == "Relu" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({ 3 }));
    ASSERT_EQ(count_relus, 1);
    ASSERT_EQ(count_constants, 6);
    ASSERT_EQ(count_parameters, 1);
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
TEST(PDPD_Reader_Tests, ImportBasicModelToCoreWstring) {
    std::string win_dir_path = ONNX_TEST_MODELS;
    std::replace(win_dir_path.begin(), win_dir_path.end(), '/', '\\');
    const std::wstring unicode_win_dir_path = FileUtils::multiByteCharToWString(win_dir_path.c_str());
    auto model = unicode_win_dir_path + L"ひらがな日本語.pdmodel";
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(model);
    auto function = cnnNetwork.getFunction();

    int count_relus = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_relus += (op_type == "Relu" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({ 3 }));
    ASSERT_EQ(count_relus, 1);
    ASSERT_EQ(count_constants, 6);
    ASSERT_EQ(count_parameters, 1);
}
#endif
