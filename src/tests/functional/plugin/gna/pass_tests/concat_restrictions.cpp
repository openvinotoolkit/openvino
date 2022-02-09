// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#include <vector>
#include <tuple>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

/* ============= Concat Layer Restrictions Tests ============= */

using ConcatRestrictionsParamsTuple = typename std::tuple<
    InferenceEngine::SizeVector,        // Input shapes
    unsigned int,                       // Concatenation axis
    InferenceEngine::Precision,         // Network Precision
    std::map<std::string, std::string>, // Configuration
    std::string>;                       // Device name

namespace ConcatTestsDefinitions {

struct ReLUConcatAxis {
    static const char* getName() { return "ReLUConcatAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape, const unsigned int& axis,
        const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::OutputVector concatInputs;

        ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto relu = ngraph::builder::makeActivation(params[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);
        concatInputs.push_back(relu);
        size_t totalSize = ngraph::shape_size(inputShape);
        auto constValues = CommonTestUtils::generate_float_numbers(totalSize, -0.1f, 0.1f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "type: Concat, and concatenation axis("; }
};

struct MatmulConcatAxis {
    static const char* getName() { return "MatmulConcatAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape, const unsigned int& axis,
        const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::OutputVector concatInputs;
        ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});
        ngraph::Shape mulConstShape;

        switch (inputShape.size()) {
        default:
        case 2:
            mulConstShape = {inputShape[1], inputShape[1]};
            break;
        case 3:
            mulConstShape = {inputShape[0], inputShape[2], inputShape[2]};
            break;
        case 4:
            mulConstShape = {inputShape[0], inputShape[1], inputShape[3], inputShape[3]};
            break;
        }

        size_t mulConstSize = ngraph::shape_size(mulConstShape);
        std::vector<float> weights1(mulConstSize);
        std::vector<float> weights2(mulConstSize);
        std::iota(weights1.begin(), weights1.end(), 0.0f);
        std::iota(weights2.begin(), weights2.end(), 0.0f);
        auto constMul1 = ngraph::builder::makeConstant<float>(ngPrc, mulConstShape, weights1);
        auto constMul2 = ngraph::builder::makeConstant<float>(ngPrc, mulConstShape, weights2);
        auto matmul1 = std::make_shared<ngraph::opset8::MatMul>(params[0], constMul1, false, true);
        concatInputs.push_back(matmul1);
        auto matmul2 = std::make_shared<ngraph::opset8::MatMul>(params[0], constMul2, false, true);
        concatInputs.push_back(matmul2);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "type: Concat, and concatenation axis("; }
};

struct ConvNCHWConcatAxis {
    static const char* getName() { return "ConvNCHWConcatAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape, const unsigned int& axis,
        const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::OutputVector concatInputs;
        ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});

        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[1] * kernelSize,
            -0.2f, 0.2f);
        auto conv = ngraph::builder::makeConvolution(params[0], ngPrc, {1, kernelSize}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, numOutChannels, true, filterWeights);

        concatInputs.push_back(conv);
        size_t totalSize = ngraph::shape_size(inputShape);
        auto constValues = CommonTestUtils::generate_float_numbers(totalSize, -0.0001f, 0.0001f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "for input dimensions"; }
};

struct ConvNHWCConcatAxis {
    static const char* getName() { return "ConvNHWCConcatAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape, const unsigned int& axis,
        const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::OutputVector concatInputs;
        ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto transposeInOrder = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2});
        auto transposeIn = std::make_shared<ngraph::opset8::Transpose>(params[0], transposeInOrder);
        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize,
            -0.2f, 0.2f);
        auto conv = ngraph::builder::makeConvolution(transposeIn, ngPrc, {1, kernelSize}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, numOutChannels, true, filterWeights);
        auto transposeOutOrder = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto transposeOut = std::make_shared<ngraph::opset8::Transpose>(conv, transposeOutOrder);

        concatInputs.push_back(transposeOut);
        size_t totalSize = ngraph::shape_size(inputShape);
        auto constValues = CommonTestUtils::generate_float_numbers(totalSize, -0.0001f, 0.0001f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "type: Concat, and concatenation axis("; }
};

struct ConvConcatNHWCAxis {
    static const char* getName() { return "ConvConcatNHWCAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape, const unsigned int& axis,
        const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::OutputVector concatInputs;
        ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto transposeInOrder = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2});
        auto transposeIn1 = std::make_shared<ngraph::opset8::Transpose>(params[0], transposeInOrder);
        auto transposeIn2 = std::make_shared<ngraph::opset8::Transpose>(params[0], transposeInOrder);
        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights1 = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize,
            -0.1f, 2.2f);
        std::vector<float> filterWeights2 = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize,
            -1.2f, 0.5f);
        auto conv1 = ngraph::builder::makeConvolution(transposeIn1, ngPrc, {1, kernelSize}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, numOutChannels, true, filterWeights1);
        auto conv2 = ngraph::builder::makeConvolution(transposeIn2, ngPrc, {1, kernelSize}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, numOutChannels, true, filterWeights2);

        concatInputs.push_back(conv1);
        concatInputs.push_back(conv2);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        auto transposeOutOrder = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto transposeOut = std::make_shared<ngraph::opset8::Transpose>(concat, transposeOutOrder);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(transposeOut)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "type: Concat, and concatenation axis("; }
};

template<typename T>
class ConcatRestrictions : public testing::WithParamInterface<ConcatRestrictionsParamsTuple>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConcatRestrictionsParamsTuple> obj) {
        InferenceEngine::SizeVector inputShape;
        unsigned int concatAxis;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> configuration;
        std::string targetDevice;
        std::tie(inputShape, concatAxis, netPrecision, configuration, targetDevice) = obj.param;
        std::ostringstream result;
        result << T::getName() << "_";
        result << "inputShape=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "concatAxis=" << concatAxis << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
        return result.str();
    }
    static const char* getMatch() { return T::getMatch(); }
protected:
    void SetUp() override {
        InferenceEngine::SizeVector inputShape;
        unsigned int concatAxis;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, concatAxis, netPrecision, configuration, targetDevice) = this->GetParam();
        function = T::createTopology(inputShape, concatAxis, netPrecision);
    }
};

using ReLUConcatRestrictionsNeg = ConcatRestrictions<ReLUConcatAxis>;
using ReLUConcatRestrictionsPos = ConcatRestrictions<ReLUConcatAxis>;
using MatMulConcatRestrictionsNeg = ConcatRestrictions<MatmulConcatAxis>;
using MatMulConcatRestrictionsPos = ConcatRestrictions<MatmulConcatAxis>;
using ConvNCHWConcatRestrictionsNeg = ConcatRestrictions<ConvNCHWConcatAxis>;
using ConvNCHWConcatRestrictionsPos = ConcatRestrictions<ConvNCHWConcatAxis>;
using ConvNHWCConcatRestrictionsNeg = ConcatRestrictions<ConvNHWCConcatAxis>;
using ConvNHWCConcatRestrictionsPos = ConcatRestrictions<ConvNHWCConcatAxis>;
using ConvConcatNHWCRestrictionsNeg = ConcatRestrictions<ConvConcatNHWCAxis>;
using ConvConcatNHWCRestrictionsPos = ConcatRestrictions<ConvConcatNHWCAxis>;

TEST_P(ReLUConcatRestrictionsNeg, CompareWithRefImpl) {
    std::stringstream what;
    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(what.rdbuf());
    LoadNetwork();
    EXPECT_TRUE(what.str().find(getMatch()) != std::string::npos);
    std::cout.rdbuf(sbuf);
};

// TODO: this test is left for future when GNA plugin handles const tranposition required for concats with interleaved layers
//TEST_P(ReLUConcatRestrictionsPos, CompareWithRefImpl) {
//    Run();
//};

TEST_P(MatMulConcatRestrictionsNeg, CompareWithRefImpl) {
    std::stringstream what;
    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(what.rdbuf());
    LoadNetwork();
    EXPECT_TRUE(what.str().find(getMatch()) != std::string::npos);
    std::cout.rdbuf(sbuf);
};

TEST_P(MatMulConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvNCHWConcatRestrictionsNeg, CompareWithRefImpl) {
    std::string what;
    try {
        LoadNetwork();
    }
    catch (const std::exception& e) {
        what.assign(e.what());
    }
    EXPECT_TRUE(what.find(getMatch()) != std::string::npos);
};

TEST_P(ConvNCHWConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvNHWCConcatRestrictionsNeg, CompareWithRefImpl) {
    std::stringstream what;
    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(what.rdbuf());
    LoadNetwork();
    EXPECT_TRUE(what.str().find(getMatch()) != std::string::npos);
    std::cout.rdbuf(sbuf);
};

TEST_P(ConvNHWCConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvConcatNHWCRestrictionsNeg, CompareWithRefImpl) {
    std::stringstream what;
    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(what.rdbuf());
    LoadNetwork();
    EXPECT_TRUE(what.str().find(getMatch()) != std::string::npos);
    std::cout.rdbuf(sbuf);
};

TEST_P(ConvConcatNHWCRestrictionsPos, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};
const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
        {"LOG_LEVEL", "LOG_WARNING"}
    }
};

// Negative 4D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul4D_neg = {{1, 2, 4, 8}};
const std::vector<unsigned int> concatAxisMatMul4D_neg = {2, 3};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_4d, MatMulConcatRestrictionsNeg,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesMatMul4D_neg),
                            ::testing::ValuesIn(concatAxisMatMul4D_neg),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 4D MatMul cases - TODO: this test fails with 4D Gemm computation errors
//const std::vector<std::vector<size_t>> inputShapesMatMul4D_pos = {{1, 2, 4, 8}};
//const std::vector<unsigned int> concatAxisMatMul4D_pos = {0, 1};
//
//INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_4d, MatMulConcatRestrictionsPos,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesMatMul4D_pos),
//        ::testing::ValuesIn(concatAxisMatMul4D_pos),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
//    MatMulConcatRestrictionsPos::getTestCaseName);

// Negative 3D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul3D_neg = {{2, 4, 8}};
const std::vector<unsigned int> concatAxisMatMul3D_neg = {0, 2};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_3d, MatMulConcatRestrictionsNeg,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesMatMul3D_neg),
                            ::testing::ValuesIn(concatAxisMatMul3D_neg),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 3D MatMul cases - TODO: this test fails with 3D Gemm computation errors
//const std::vector<std::vector<size_t>> inputShapesMatMul3D_pos = {{2, 4, 8}};
//const std::vector<unsigned int> concatAxisMatMul3D_pos = {1};
//
//INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_3d, MatMulConcatRestrictionsPos,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesMatMul3D_pos),
//        ::testing::ValuesIn(concatAxisMatMul3D_pos),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
//    MatMulConcatRestrictionsPos::getTestCaseName);

// Negative 2D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul2D_neg = {{8, 64}};
const std::vector<unsigned int> concatAxisMatMul2D_neg = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_2d, MatMulConcatRestrictionsNeg,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesMatMul2D_neg),
                            ::testing::ValuesIn(concatAxisMatMul2D_neg),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 2D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul2D_pos = {{8, 64}};
const std::vector<unsigned int> concatAxisMatMul2D_pos = {1};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_2d, MatMulConcatRestrictionsPos,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesMatMul2D_pos),
                            ::testing::ValuesIn(concatAxisMatMul2D_pos),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        MatMulConcatRestrictionsPos::getTestCaseName);

// Negative ReLU cases
const std::vector<std::vector<size_t>> inputShapesReLU_neg = {{64, 128}};
const std::vector<unsigned int> concatAxisReLU_neg = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_relu, ReLUConcatRestrictionsNeg,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesReLU_neg),
        ::testing::ValuesIn(concatAxisReLU_neg),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(configs),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
    ReLUConcatRestrictionsNeg::getTestCaseName);

// Positive ReLU cases
const std::vector<std::vector<size_t>> inputShapesReLU_pos = {{64, 128}};
const std::vector<unsigned int> concatAxisReLU_pos = {1};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_relu, ReLUConcatRestrictionsPos,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesReLU_pos),
        ::testing::ValuesIn(concatAxisReLU_pos),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(configs),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
    ReLUConcatRestrictionsPos::getTestCaseName);

// Negative cases NCHW
const std::vector<std::vector<size_t>> inputShapesConvNCHW_neg = {{1, 8, 16, 32}};
const std::vector<unsigned int> concatAxisConvNCHW_neg = {3}; // Axis 1 should be negative as well,
                                                              // but is handled by the plugin in this case

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvNCHWConcatRestrictionsNeg,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesConvNCHW_neg),
                            ::testing::ValuesIn(concatAxisConvNCHW_neg),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvNCHWConcatRestrictionsNeg::getTestCaseName);

// Positive cases NCHW
const std::vector<std::vector<size_t>> inputShapesConvNCHW_pos = {{1, 8, 1, 64}};
const std::vector<unsigned int> concatAxisConvNCHW_pos = {2, 3}; // TODO: incorrect output buffer calculation
                                                                 // when 0 axis is used

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvNCHWConcatRestrictionsPos,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesConvNCHW_pos),
                            ::testing::ValuesIn(concatAxisConvNCHW_pos),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvNCHWConcatRestrictionsPos::getTestCaseName);

// Negative cases NHWC
const std::vector<std::vector<size_t>> inputShapesNHWC_neg = {{1, 2, 16, 8}};
const std::vector<unsigned int> concatAxisNHWC_neg = {2, 3};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvNHWCConcatRestrictionsNeg,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesNHWC_neg),
                            ::testing::ValuesIn(concatAxisNHWC_neg),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvNHWCConcatRestrictionsNeg::getTestCaseName);

// Positive cases NHWC
const std::vector<std::vector<size_t>> inputShapesNHWC_pos = {{1, 1, 16, 8}};
const std::vector<unsigned int> concatAxisNHWC_pos = {1, 2};


INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvNHWCConcatRestrictionsPos,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapesNHWC_pos),
                            ::testing::ValuesIn(concatAxisNHWC_pos),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(configs),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvNHWCConcatRestrictionsPos::getTestCaseName);

// Negative cases NHWC with concat inside transposes - TODO: this test fails, because the transposes are not removed
//const std::vector<std::vector<size_t>> inputShapesConcatNHWC_neg = {{1, 1, 16, 8}};
//const std::vector<unsigned int> concatAxisConcatNHWC_neg = {1};
//
//INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvConcatNHWCRestrictionsNeg,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesConcatNHWC_neg),
//        ::testing::ValuesIn(concatAxisConcatNHWC_neg),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
//    ConvConcatNHWCRestrictionsNeg::getTestCaseName);

// Positive cases NHWC with concat inside transposes
const std::vector<std::vector<size_t>> inputShapesConcatNHWC_pos = {{1, 1, 16, 8}};
const std::vector<unsigned int> concatAxisConcatNHWC_pos = {2, 3}; // TODO: 0 fails with unsupported permute,
                                                                   // because the transposes are not removed


INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvConcatNHWCRestrictionsPos,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesConcatNHWC_pos),
        ::testing::ValuesIn(concatAxisConcatNHWC_pos),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(configs),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
    ConvConcatNHWCRestrictionsPos::getTestCaseName);

} // namespace ConcatTestsDefinitions
