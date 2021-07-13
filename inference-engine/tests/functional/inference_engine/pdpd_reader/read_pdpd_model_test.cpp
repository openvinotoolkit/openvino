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
#include <ngraph/opsets/opset8.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

TEST(PDPD_Reader_Tests, ImportBasicModelToCore) {
    auto model = std::string(PDPD_TEST_MODELS) + "relu.pdmodel";
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(model);
    auto function = cnnNetwork.getFunction();

    const auto inputType = ngraph::element::f32;
    const auto inputShape = ngraph::Shape{ 3 };

    const auto data = std::make_shared<ngraph::opset8::Parameter>(inputType, inputShape);
    const auto relu = std::make_shared<ngraph::opset8::Relu>(data);
    const auto scale = std::make_shared<ngraph::opset8::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float>{1});
    const auto bias = std::make_shared<ngraph::opset8::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float>{0});
    const auto node_multiply = std::make_shared<ngraph::opset8::Multiply>(relu, scale);
    const auto node_add = std::make_shared<ngraph::opset8::Add>(node_multiply, bias);
    const auto result = std::make_shared<ngraph::opset8::Result>(node_add);
    const auto reference = std::make_shared<const ngraph::Function>(
        ngraph::NodeVector{ result },
        ngraph::ParameterVector{ data },
        "RefPDPDFunction");
    ngraph::helpers::CompareFunctions(*reference, *function);
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
TEST(PDPD_Reader_Tests, ImportBasicModelToCoreWstring) {
    std::string win_dir_path{ PDPD_TEST_MODELS };
    std::replace(win_dir_path.begin(), win_dir_path.end(), '/', '\\');
    const std::wstring unicode_win_dir_path = FileUtils::multiByteCharToWString(win_dir_path.c_str());
    auto model = unicode_win_dir_path + L"ひらがな日本語.pdmodel";
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(model);
    auto function = cnnNetwork.getFunction();

    const auto inputType = ngraph::element::f32;
    const auto inputShape = ngraph::Shape{ 3 };

    const auto data = std::make_shared<ngraph::opset8::Parameter>(inputType, inputShape);
    const auto relu = std::make_shared<ngraph::opset8::Relu>(data);
    const auto scale = std::make_shared<ngraph::opset8::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float>{1});
    const auto bias = std::make_shared<ngraph::opset8::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float>{0});
    const auto node_multiply = std::make_shared<ngraph::opset8::Multiply>(relu, scale);
    const auto node_add = std::make_shared<ngraph::opset8::Add>(node_multiply, bias);
    const auto result = std::make_shared<ngraph::opset8::Result>(node_add);
    const auto reference = std::make_shared<const ngraph::Function>(
        ngraph::NodeVector{ result },
        ngraph::ParameterVector{ data },
        "RefPDPDFunction");
    ngraph::helpers::CompareFunctions(*reference, *function);
}
#endif
