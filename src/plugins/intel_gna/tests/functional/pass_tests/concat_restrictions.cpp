// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#include <ie_core.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "openvino/opsets/opset11.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

/* ============= Concat Layer Restrictions Tests ============= */

using ConcatRestrictionsParamsTuple = typename std::tuple<InferenceEngine::SizeVector,         // Input shapes
                                                          unsigned int,                        // Concatenation axis
                                                          InferenceEngine::Precision,          // Network Precision
                                                          std::map<std::string, std::string>,  // Configuration
                                                          std::string>;                        // Device name

namespace ConcatTestsDefinitions {

using namespace InferenceEngine;
using namespace ngraph::builder;
using namespace ov;
using namespace ov::element;
using namespace ov::opset11;
using namespace std;

shared_ptr<FakeQuantize> create_fq_node(const Type& type,
                                        const shared_ptr<ov::Node>& node,
                                        float fqMin,
                                        float fqMax,
                                        size_t levels) {
    auto fqInpMin = makeConstant<float>(type, {1}, {fqMin});
    auto fqInpMax = makeConstant<float>(type, {1}, {fqMax});
    auto fqOutMin = makeConstant<float>(type, {1}, {fqMin});
    auto fqOutMax = makeConstant<float>(type, {1}, {fqMax});
    return make_shared<FakeQuantize>(node, fqInpMin, fqInpMax, fqOutMin, fqOutMax, levels);
}

struct ReLUConcatAxis {
    static const char* getName() {
        return "ReLUConcatAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concatInputs;

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto relu = ngraph::builder::makeActivation(params[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);
        concatInputs.push_back(relu);
        size_t totalSize = ov::shape_size(inputShape);
        auto constValues = ov::test::utils::generate_float_numbers(totalSize, -0.1f, 0.1f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "Unsupported concatenation axis";
    }
};

struct MatmulConcatAxis {
    static const char* getName() {
        return "MatmulConcatAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concatInputs;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        ov::Shape mulConstShape;

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

        size_t mulConstSize = ov::shape_size(mulConstShape);
        std::vector<float> weights1(mulConstSize);
        std::vector<float> weights2(mulConstSize);
        std::iota(weights1.begin(), weights1.end(), 0.0f);
        std::iota(weights2.begin(), weights2.end(), 0.0f);
        auto constMul1 = ngraph::builder::makeConstant<float>(ngPrc, mulConstShape, weights1);
        auto constMul2 = ngraph::builder::makeConstant<float>(ngPrc, mulConstShape, weights2);
        auto matmul1 = std::make_shared<ov::opset10::MatMul>(params[0], constMul1, false, true);
        concatInputs.push_back(matmul1);
        auto matmul2 = std::make_shared<ov::opset10::MatMul>(params[0], constMul2, false, true);
        concatInputs.push_back(matmul2);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "Unsupported concatenation axis";
    }
};

struct ConvNCHWConcatAxis {
    static const char* getName() {
        return "ConvNCHWConcatAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concatInputs;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[1] * kernelSize, -0.2f, 0.2f);
        auto conv = ngraph::builder::makeConvolution(params[0],
                                                     ngPrc,
                                                     {1, kernelSize},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::VALID,
                                                     numOutChannels,
                                                     true,
                                                     filterWeights);

        concatInputs.push_back(conv);
        size_t totalSize = ov::shape_size(inputShape);
        auto constValues = ov::test::utils::generate_float_numbers(totalSize, -0.0001f, 0.0001f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "for input dimensions";
    }
};

struct ConvNHWCConcatAxis {
    static const char* getName() {
        return "ConvNHWCConcatAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concatInputs;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto transposeInOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transposeIn = std::make_shared<ov::opset10::Transpose>(params[0], transposeInOrder);
        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize, -0.2f, 0.2f);
        auto conv = ngraph::builder::makeConvolution(transposeIn,
                                                     ngPrc,
                                                     {1, kernelSize},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::VALID,
                                                     numOutChannels,
                                                     true,
                                                     filterWeights);
        auto transposeOutOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transposeOut = std::make_shared<ov::opset10::Transpose>(conv, transposeOutOrder);

        concatInputs.push_back(transposeOut);
        size_t totalSize = ov::shape_size(inputShape);
        auto constValues = ov::test::utils::generate_float_numbers(totalSize, -0.0001f, 0.0001f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {inputShape}, constValues);
        concatInputs.push_back(constNode);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(concat)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "Unsupported concatenation axis";
    }
};

struct ConvConcatNHWCAxis {
    static const char* getName() {
        return "ConvConcatNHWCAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concatInputs;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto transposeInOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transposeIn1 = std::make_shared<ov::opset10::Transpose>(params[0], transposeInOrder);
        auto transposeIn2 = std::make_shared<ov::opset10::Transpose>(params[0], transposeInOrder);
        size_t numOutChannels = 8;
        size_t kernelSize = 1;
        std::vector<float> filterWeights1 =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize, -0.1f, 2.2f);
        std::vector<float> filterWeights2 =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize, -1.2f, 0.5f);
        auto conv1 = ngraph::builder::makeConvolution(transposeIn1,
                                                      ngPrc,
                                                      {1, kernelSize},
                                                      {1, 1},
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ov::op::PadType::VALID,
                                                      numOutChannels,
                                                      true,
                                                      filterWeights1);
        auto conv2 = ngraph::builder::makeConvolution(transposeIn2,
                                                      ngPrc,
                                                      {1, kernelSize},
                                                      {1, 1},
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ov::op::PadType::VALID,
                                                      numOutChannels,
                                                      true,
                                                      filterWeights2);

        concatInputs.push_back(conv1);
        concatInputs.push_back(conv2);
        auto concat = ngraph::builder::makeConcat(concatInputs, axis);

        auto transposeOutOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transposeOut = std::make_shared<ov::opset10::Transpose>(concat, transposeOutOrder);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(transposeOut)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "Unsupported concatenation axis";
    }
};

struct ConvConcatConcatNHWCAxis {
    static const char* getName() {
        return "ConvConcatConcatNHWCAxis";
    }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::SizeVector& inputShape,
                                                            const unsigned int& axis,
                                                            const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::OutputVector concat1Inputs, concat2Inputs;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto transposeInOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transposeIn1 = std::make_shared<ov::opset10::Transpose>(params[0], transposeInOrder);
        auto transposeIn2 = std::make_shared<ov::opset10::Transpose>(params[0], transposeInOrder);
        size_t numOutChannels = 64;
        size_t kernelSize = 1;
        std::vector<float> filterWeights1 =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize, -0.1f, 2.2f);
        std::vector<float> filterWeights2 =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * kernelSize, -1.2f, 0.5f);
        auto conv1 = ngraph::builder::makeConvolution(transposeIn1,
                                                      ngPrc,
                                                      {1, kernelSize},
                                                      {1, 1},
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ov::op::PadType::VALID,
                                                      numOutChannels,
                                                      true,
                                                      filterWeights1);
        auto conv2 = ngraph::builder::makeConvolution(transposeIn2,
                                                      ngPrc,
                                                      {1, kernelSize},
                                                      {1, 1},
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ov::op::PadType::VALID,
                                                      numOutChannels,
                                                      true,
                                                      filterWeights2);

        auto transposeOutOrder = ov::opset10::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transposeOut1 = std::make_shared<ov::opset10::Transpose>(conv1, transposeOutOrder);
        auto transposeOut2 = std::make_shared<ov::opset10::Transpose>(conv2, transposeOutOrder);

        concat1Inputs.push_back(transposeOut1);
        concat1Inputs.push_back(transposeOut2);
        auto concat1 = ngraph::builder::makeConcat(concat1Inputs, 2);
        auto squeeze = std::make_shared<ov::opset10::Squeeze>(
            concat1,
            ov::opset10::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1}));

        size_t totalSize = ov::shape_size(squeeze->get_shape());
        auto constValues = ov::test::utils::generate_float_numbers(totalSize, -0.0001f, 0.0001f);
        auto constNode = ngraph::builder::makeConstant(ngPrc, {squeeze->get_shape()}, constValues);

        concat2Inputs.push_back(squeeze);
        concat2Inputs.push_back(constNode);
        auto concat2 = ngraph::builder::makeConcat(concat2Inputs, axis);
        auto reshape = std::make_shared<ov::opset10::Reshape>(
            concat2,
            ov::opset10::Constant::create(ov::element::i64,
                                          ov::Shape{2},
                                          ov::Shape{1, shape_size(concat2->get_shape())}),
            false);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(reshape)};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() {
        return "Unsupported concatenation axis";
    }
};

// This test performs checks on the following network:
//             Param1
//               |
//             Reshape          Param2
//               |                |
//           Convolution         FQ
//               |                |
//              ReLU           Reshape
//               |                |
//               FQ           Transpose
//               |                |
//             Reshape         Reshape
//               |                |
//            Transpose       Transpose
//                     \      /
//                      Concat
//                        |
//                      Reshape
//                        |
//                      Result
//
// We want to ensure this Concat topology will not be detected as unsupported one.

struct TransposeTransposeConcat {
    static const char* getName() {
        return "TransposeTransposeConcat";
    }

    static std::shared_ptr<ngraph::Function> createTopology(const SizeVector& input_shapes,
                                                            const unsigned int& axis,
                                                            const Precision& net_precision) {
        const float fq1 = 5.5, fq2 = 10.0;
        const size_t levels = 65536;
        const vector<size_t> invert = {1, 0};
        const vector<size_t> kernel_shape = {1, 3};
        const size_t input_channels = 8;
        const size_t output_channels = 64;

        IE_ASSERT(input_shapes[0] % input_channels == 0);
        IE_ASSERT(input_shapes[1] % input_shapes[0] == 0);

        vector<size_t> concat_input_shape = {input_shapes[1] / input_shapes[0], input_shapes[0]};
        vector<size_t> conv_input_shape = {1, input_channels, 1, input_shapes[0] / input_channels};

        auto ng_prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);
        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ng_prc, ov::Shape{1, input_shapes[0]}),
                                   std::make_shared<ov::op::v0::Parameter>(ng_prc, ov::Shape{1, input_shapes[1]})};
        // 1st concat input
        auto reshape_l1_const = make_shared<Constant>(i64, Shape{conv_input_shape.size()}, conv_input_shape);
        auto reshape_l1 = make_shared<Reshape>(inputs[0], reshape_l1_const, false);

        auto conv_l1_weights = makeConstant<float>(ng_prc,
                                                   {output_channels, input_channels, kernel_shape[0], kernel_shape[1]},
                                                   {},
                                                   true,
                                                   1.0f,
                                                   -1.0f);
        auto conv_l1_weights_fq = create_fq_node(ng_prc, conv_l1_weights, -fq1, fq1, levels);
        auto conv_l1 = make_shared<Convolution>(reshape_l1,
                                                conv_l1_weights_fq,
                                                vector<size_t>{1, 1},
                                                vector<ptrdiff_t>{0, 0},
                                                vector<ptrdiff_t>{0, 0},
                                                vector<size_t>{1, 1},
                                                ov::op::PadType::VALID);
        auto relu_l1 = make_shared<Relu>(conv_l1);
        auto fq_l1 = create_fq_node(ng_prc, relu_l1, -fq1, fq1, levels);
        auto reshape_l2_const = make_shared<Constant>(i64, Shape{concat_input_shape.size()}, concat_input_shape);
        auto reshape_l2 = make_shared<Reshape>(fq_l1, reshape_l2_const, false);
        auto transpose_l1_const = Constant::create(i64, Shape{invert.size()}, invert);
        auto transpose_l1 = make_shared<Transpose>(reshape_l2, transpose_l1_const);

        // 2nd concat input
        auto fq_r1 = create_fq_node(ng_prc, inputs[1], -fq2, fq2, levels);
        auto reshape_r1_const = make_shared<Constant>(i64, Shape{2}, concat_input_shape);
        auto reshape_r1 = make_shared<Reshape>(fq_r1, reshape_r1_const, false);
        auto transpose_r1_const = Constant::create(i64, Shape{invert.size()}, invert);
        auto transpose_r1 = make_shared<Transpose>(reshape_r1, transpose_r1_const);
        auto reshape_r3_const = make_shared<Constant>(i64, Shape{concat_input_shape.size()}, concat_input_shape);
        auto reshape_r3 = make_shared<Reshape>(transpose_r1, reshape_r3_const, false);
        auto transpose_r2_const = Constant::create(i64, Shape{invert.size()}, invert);
        auto transpose_r2 = make_shared<Transpose>(reshape_r3, transpose_r2_const);

        // Concat
        auto concat = makeConcat({transpose_l1, transpose_r2}, 0);

        auto width_after_conv = (conv_input_shape[3] - kernel_shape[1]) + 1;
        auto reshape_const =
            make_shared<Constant>(i64, Shape{2}, vector<size_t>{1, 2 * output_channels * width_after_conv});
        auto reshape = make_shared<Reshape>(concat, reshape_const, false);

        ResultVector result{make_shared<Result>(reshape)};
        auto model = make_shared<Model>(result, inputs, getName());
        return model;
    }
};

template <typename T>
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
        result << "inputShape=" << ov::test::utils::vec2str(inputShape) << "_";
        result << "concatAxis=" << concatAxis << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
        return result.str();
    }
    static const char* getMatch() {
        return T::getMatch();
    }

    Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

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
using ConvConcatConcatNHWCRestrictionsNeg = ConcatRestrictions<ConvConcatConcatNHWCAxis>;
using ConvConcatConcatNHWCRestrictionsPos = ConcatRestrictions<ConvConcatConcatNHWCAxis>;
using TransposeTransposeConcatPos = ConcatRestrictions<TransposeTransposeConcat>;

// TODO: those tests are left for future when GNA plugin handles const tranposition required for concats with
// interleaved layers
// TEST_P(ReLUConcatRestrictionsNeg, CompareWithRefImpl) {
//     ExpectLoadNetworkToThrow(getMatch());
// };
//
// TEST_P(ReLUConcatRestrictionsPos, CompareWithRefImpl) {
//    Run();
//};

TEST_P(MatMulConcatRestrictionsNeg, CompareWithRefImpl) {
    ExpectLoadNetworkToThrow(getMatch());
};

TEST_P(MatMulConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvNCHWConcatRestrictionsNeg, CompareWithRefImpl) {
    ExpectLoadNetworkToThrow(getMatch());
};

TEST_P(ConvNCHWConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvNHWCConcatRestrictionsNeg, CompareWithRefImpl) {
    ExpectLoadNetworkToThrow(getMatch());
};

TEST_P(ConvNHWCConcatRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvConcatNHWCRestrictionsNeg, CompareWithRefImpl) {
    ExpectLoadNetworkToThrow(getMatch());
};

TEST_P(ConvConcatNHWCRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvConcatConcatNHWCRestrictionsNeg, CompareWithRefImpl) {
    ExpectLoadNetworkToThrow(getMatch());
};

TEST_P(ConvConcatConcatNHWCRestrictionsPos, CompareWithRefImpl) {
    Run();
};

TEST_P(TransposeTransposeConcatPos, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};
const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

// Negative 4D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul4D_neg = {{1, 2, 4, 8}};
const std::vector<unsigned int> concatAxisMatMul4D_neg = {2, 3};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_4d,
                         MatMulConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesMatMul4D_neg),
                                            ::testing::ValuesIn(concatAxisMatMul4D_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 4D MatMul cases - TODO: this test fails with 4D Gemm computation errors
// const std::vector<std::vector<size_t>> inputShapesMatMul4D_pos = {{1, 2, 4, 8}};
// const std::vector<unsigned int> concatAxisMatMul4D_pos = {0, 1};
//
// INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_4d, MatMulConcatRestrictionsPos,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesMatMul4D_pos),
//        ::testing::ValuesIn(concatAxisMatMul4D_pos),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(ov::test::utils::DEVICE_GNA)),
//    MatMulConcatRestrictionsPos::getTestCaseName);

// Negative 3D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul3D_neg = {{2, 4, 8}};
const std::vector<unsigned int> concatAxisMatMul3D_neg = {0, 2};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_3d,
                         MatMulConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesMatMul3D_neg),
                                            ::testing::ValuesIn(concatAxisMatMul3D_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 3D MatMul cases - TODO: this test fails with 3D Gemm computation errors
// const std::vector<std::vector<size_t>> inputShapesMatMul3D_pos = {{2, 4, 8}};
// const std::vector<unsigned int> concatAxisMatMul3D_pos = {1};
//
// INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_3d, MatMulConcatRestrictionsPos,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesMatMul3D_pos),
//        ::testing::ValuesIn(concatAxisMatMul3D_pos),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(ov::test::utils::DEVICE_GNA)),
//    MatMulConcatRestrictionsPos::getTestCaseName);

// Negative 2D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul2D_neg = {{8, 64}};
const std::vector<unsigned int> concatAxisMatMul2D_neg = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_2d,
                         MatMulConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesMatMul2D_neg),
                                            ::testing::ValuesIn(concatAxisMatMul2D_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         MatMulConcatRestrictionsNeg::getTestCaseName);

// Positive 2D MatMul cases
const std::vector<std::vector<size_t>> inputShapesMatMul2D_pos = {{8, 64}};
const std::vector<unsigned int> concatAxisMatMul2D_pos = {1};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_matmul_2d,
                         MatMulConcatRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesMatMul2D_pos),
                                            ::testing::ValuesIn(concatAxisMatMul2D_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         MatMulConcatRestrictionsPos::getTestCaseName);

// Negative ReLU cases
const std::vector<std::vector<size_t>> inputShapesReLU_neg = {{64, 128}};
const std::vector<unsigned int> concatAxisReLU_neg = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_relu,
                         ReLUConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesReLU_neg),
                                            ::testing::ValuesIn(concatAxisReLU_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ReLUConcatRestrictionsNeg::getTestCaseName);

// Positive ReLU cases
const std::vector<std::vector<size_t>> inputShapesReLU_pos = {{64, 128}};
const std::vector<unsigned int> concatAxisReLU_pos = {1};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions_relu,
                         ReLUConcatRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesReLU_pos),
                                            ::testing::ValuesIn(concatAxisReLU_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ReLUConcatRestrictionsPos::getTestCaseName);

// Negative cases NCHW
const std::vector<std::vector<size_t>> inputShapesConvNCHW_neg = {{1, 8, 16, 32}};
const std::vector<unsigned int> concatAxisConvNCHW_neg = {3};  // Axis 1 should be negative as well,
                                                               // but is handled by the plugin in this case

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvNCHWConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConvNCHW_neg),
                                            ::testing::ValuesIn(concatAxisConvNCHW_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvNCHWConcatRestrictionsNeg::getTestCaseName);

// Positive cases NCHW
const std::vector<std::vector<size_t>> inputShapesConvNCHW_pos = {{1, 8, 1, 64}};
const std::vector<unsigned int> concatAxisConvNCHW_pos = {2, 3};  // TODO: incorrect output buffer calculation
                                                                  // when 0 axis is used

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvNCHWConcatRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConvNCHW_pos),
                                            ::testing::ValuesIn(concatAxisConvNCHW_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvNCHWConcatRestrictionsPos::getTestCaseName);

// Negative cases NHWC
const std::vector<std::vector<size_t>> inputShapesNHWC_neg = {{1, 2, 16, 8}};
const std::vector<unsigned int> concatAxisNHWC_neg = {2, 3};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvNHWCConcatRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesNHWC_neg),
                                            ::testing::ValuesIn(concatAxisNHWC_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvNHWCConcatRestrictionsNeg::getTestCaseName);

// Positive cases NHWC
const std::vector<std::vector<size_t>> inputShapesNHWC_pos = {{1, 1, 16, 8}};
const std::vector<unsigned int> concatAxisNHWC_pos = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvNHWCConcatRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesNHWC_pos),
                                            ::testing::ValuesIn(concatAxisNHWC_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvNHWCConcatRestrictionsPos::getTestCaseName);

// Negative cases NHWC with concat inside transposes - TODO: this test fails, because the transposes are not removed
// const std::vector<std::vector<size_t>> inputShapesConcatNHWC_neg = {{1, 1, 16, 8}};
// const std::vector<unsigned int> concatAxisConcatNHWC_neg = {1};
//
// INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions, ConvConcatNHWCRestrictionsNeg,
//    ::testing::Combine(
//        ::testing::ValuesIn(inputShapesConcatNHWC_neg),
//        ::testing::ValuesIn(concatAxisConcatNHWC_neg),
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::ValuesIn(configs),
//        ::testing::Values(ov::test::utils::DEVICE_GNA)),
//    ConvConcatNHWCRestrictionsNeg::getTestCaseName);

// Positive cases NHWC with concat inside transposes
const std::vector<std::vector<size_t>> inputShapesConcatNHWC_pos = {{1, 1, 16, 8}};
const std::vector<unsigned int> concatAxisConcatNHWC_pos = {2, 3};  // TODO: 0 fails with unsupported permute,
                                                                    // because the transposes are not removed

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvConcatNHWCRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConcatNHWC_pos),
                                            ::testing::ValuesIn(concatAxisConcatNHWC_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvConcatNHWCRestrictionsPos::getTestCaseName);

// Negative cases NHWC with two consecutive concats
const std::vector<std::vector<size_t>> inputShapesConcatConcatNHWC = {{1, 1, 16, 8}};
const std::vector<unsigned int> concatAxisConcatConcatNHWC_neg = {1};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvConcatConcatNHWCRestrictionsNeg,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConcatConcatNHWC),
                                            ::testing::ValuesIn(concatAxisConcatConcatNHWC_neg),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvConcatConcatNHWCRestrictionsNeg::getTestCaseName);

// Positive cases NHWC with two consecutive concats
const std::vector<unsigned int> concatAxisConcatConcatNHWC_pos = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         ConvConcatConcatNHWCRestrictionsPos,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConcatConcatNHWC),
                                            ::testing::ValuesIn(concatAxisConcatConcatNHWC_pos),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ConvConcatConcatNHWCRestrictionsPos::getTestCaseName);

const vector<SizeVector> ttc_input_shapes = {{64, 384}};
const vector<map<string, string>> ttc_configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
};
const vector<unsigned int> ttc_axis = {0};

INSTANTIATE_TEST_SUITE_P(smoke_concat_restrictions,
                         TransposeTransposeConcatPos,
                         ::testing::Combine(::testing::ValuesIn(ttc_input_shapes),
                                            ::testing::ValuesIn(ttc_axis),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ttc_configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         TransposeTransposeConcatPos::getTestCaseName);

}  // namespace ConcatTestsDefinitions
