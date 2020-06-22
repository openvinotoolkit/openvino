// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/eltwise_3x.hpp"

namespace LayerTestsDefinitions {
    std::string Eltwise3x::getTestCaseName(const testing::TestParamInfo<Eltwise3xTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        std::vector<InferenceEngine::Precision> inputPrecisions;
        bool withQuantization;
        std::string targetName;
        std::tie(inputShapes, inputPrecisions, withQuantization, targetName) = obj.param;
        std::ostringstream results;

        results << "IS0=" << CommonTestUtils::vec2str(inputShapes[0]) << "_";
        results << "IS1=" << CommonTestUtils::vec2str(inputShapes[1]) << "_";
        results << "IS2=" << CommonTestUtils::vec2str(inputShapes[2]) << "_";
        results << "IS3=" << CommonTestUtils::vec2str(inputShapes[2]) << "_";
        results << "InPRC0=" << inputPrecisions[0].name() << "_";
        results << "InPRC1=" << inputPrecisions[1].name() << "_";
        results << "InPRC2=" << inputPrecisions[2].name() << "_";
        results << "InPRC3=" << inputPrecisions[3].name() << "_";
        results << "WithQuant=" << withQuantization << "_";
        results << "targetDevice=" << targetName;
        return results.str();
    }

    void Eltwise3x::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        std::vector<InferenceEngine::Precision> inputPrecisions;
        bool withQuantization;
        std::tie(inputShapes, inputPrecisions, withQuantization, targetDevice) = this->GetParam();

        auto ngPrc0 = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[0]);
        auto ngPrc1 = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[1]);
        auto ngPrc2 = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[2]);
        auto ngPrc3 = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[3]);

        auto ngraphInputs = ngraph::builder::makeParams(ngPrc0, {inputShapes[0]});

        std::vector<float> ngraphInput1Data(ngraph::shape_size(ngraph::Shape{inputShapes[1]}));
        auto ngraphInput1 = ngraph::builder::makeConstant(ngPrc1, ngraph::Shape{inputShapes[1]}, ngraphInput1Data, true);

        std::vector<float> ngraphInput2Data(ngraph::shape_size(ngraph::Shape{inputShapes[2]}));
        auto ngraphInput2 = ngraph::builder::makeConstant(ngPrc2, ngraph::Shape{inputShapes[2]}, ngraphInput2Data, true);

        std::vector<float> ngraphInput3Data(ngraph::shape_size(ngraph::Shape{inputShapes[3]}));
        auto ngraphInput3 = ngraph::builder::makeConstant(ngPrc3, ngraph::Shape{inputShapes[3]}, ngraphInput3Data, true);

        auto eltwise0 = std::make_shared<ngraph::opset1::Add>(ngraphInputs[0], ngraphInput1);
        auto eltwise1 = std::make_shared<ngraph::opset1::Multiply>(eltwise0, ngraphInput2);


        if (withQuantization) {
            std::vector<size_t> constShape(inputShapes[0].size(), 1);
            constShape[1] = inputShapes[0][1];

            auto fq = ngraph::builder::makeFakeQuantize(eltwise1, ::ngraph::element::Type(::ngraph::element::Type_t::f32), 256, constShape);
            auto eltwise2 = std::make_shared<ngraph::opset1::Subtract>(fq, ngraphInput3);

            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise2)};
            function = std::make_shared<ngraph::Function>(results, ngraphInputs, "eltwise_3x");
        } else {
            auto eltwise2 = std::make_shared<ngraph::opset1::Subtract>(eltwise1, ngraphInput3);

            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise2)};
            function = std::make_shared<ngraph::Function>(results, ngraphInputs, "eltwise_3x");
        }
    }

    TEST_P(Eltwise3x, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
