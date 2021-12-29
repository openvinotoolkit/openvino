// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "base_reference_test.hpp"
#include "ngraph_functions/utils/data_utils.hpp"
#include "ngraph/runtime/reference/sequences.hpp"


using namespace reference_tests;
using namespace ov;

namespace {
struct LSTMSequenceParams {
    template <class T>
    LSTMSequenceParams(
        const size_t batchSize, const size_t inputSize, const size_t hiddenSize, const size_t seqLength,
        const op::RecurrentSequenceDirection& lstm_direction,
        const element::Type_t& iType,
        const std::vector<T>& XValues, const std::vector<T>& H_tValues, const std::vector<T>& C_tValues,
        const std::vector<int64_t>& S_tValues,
        const std::vector<T>& WValues, const std::vector<T>& RValues, const std::vector<T>& BValues,
        const std::vector<T>& YValues, const std::vector<T>& HoValues, const std::vector<T>& CoValues,
        const std::string& testcaseName = "") :
        batchSize(batchSize), inputSize(inputSize), hiddenSize(hiddenSize), seqLength(seqLength),
        lstm_direction(lstm_direction), iType(iType), oType(iType),
        testcaseName(testcaseName) {
            numDirections = (lstm_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

            Shape XShape = Shape{batchSize, seqLength, inputSize};
            Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
            Shape C_tShape = Shape{batchSize, numDirections, hiddenSize};
            Shape S_tShape = Shape{batchSize};
            Shape WShape = Shape{numDirections, 4 * hiddenSize, inputSize};
            Shape RShape = Shape{numDirections, 4 * hiddenSize, hiddenSize};
            Shape BShape = Shape{numDirections, 4 * hiddenSize};
            Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
            Shape HoShape = Shape{batchSize, numDirections, hiddenSize};
            Shape CoShape = Shape{batchSize, numDirections, hiddenSize};

            X = Tensor(XShape, iType, XValues);
            H_t = Tensor(H_tShape, iType, H_tValues);
            C_t = Tensor(C_tShape, iType, C_tValues);
            S_t = Tensor(S_tShape, element::Type_t::i64, S_tValues);
            W = Tensor(WShape, iType, WValues);
            R = Tensor(RShape, iType, RValues);
            B = Tensor(BShape, iType, BValues);
            Y = Tensor(YShape, oType, YValues);
            Ho = Tensor(HoShape, oType, HoValues);
            Co = Tensor(CoShape, oType, CoValues);
        }

    size_t batchSize;
    size_t inputSize;
    size_t hiddenSize;
    size_t seqLength;
    size_t numDirections;
    op::RecurrentSequenceDirection lstm_direction;
    element::Type_t iType;
    element::Type_t oType;

    Tensor X;
    Tensor H_t;
    Tensor C_t;
    Tensor S_t;
    Tensor W;
    Tensor R;
    Tensor B;
    Tensor Y;
    Tensor Ho;
    Tensor Co;
    std::string testcaseName;
};

struct LSTMSequenceV1Params {
    template <class T>
    LSTMSequenceV1Params(
        const size_t batchSize, const size_t inputSize, const size_t hiddenSize, const size_t seqLength,
        const bool input_forget, const op::RecurrentSequenceDirection& lstm_direction,
        const element::Type_t& iType,
        const std::vector<T>& XValues, const std::vector<T>& H_tValues, const std::vector<T>& C_tValues,
        const std::vector<int64_t>& S_tValues,
        const std::vector<T>& WValues, const std::vector<T>& RValues, const std::vector<T>& BValues, const std::vector<T>& PValues,
        const std::vector<T>& YValues, const std::vector<T>& HoValues, const std::vector<T>& CoValues,
        const std::string& testcaseName = "") :
        batchSize(batchSize), inputSize(inputSize), hiddenSize(hiddenSize), seqLength(seqLength),
        input_forget(input_forget), lstm_direction(lstm_direction), iType(iType), oType(iType),
        testcaseName(testcaseName) {
            numDirections = (lstm_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

            Shape XShape = Shape{batchSize, seqLength, inputSize};
            Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
            Shape C_tShape = Shape{batchSize, numDirections, hiddenSize};
            Shape S_tShape = Shape{batchSize};
            Shape WShape = Shape{numDirections, 4 * hiddenSize, inputSize};
            Shape RShape = Shape{numDirections, 4 * hiddenSize, hiddenSize};
            Shape BShape = Shape{numDirections, 4 * hiddenSize};
            Shape PShape = Shape{numDirections, 3 * hiddenSize};
            Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
            Shape HoShape = Shape{batchSize, numDirections, hiddenSize};
            Shape CoShape = Shape{batchSize, numDirections, hiddenSize};

            X = Tensor(XShape, iType, XValues);
            H_t = Tensor(H_tShape, iType, H_tValues);
            C_t = Tensor(C_tShape, iType, C_tValues);
            S_t = Tensor(S_tShape, element::Type_t::i64, S_tValues);
            W = Tensor(WShape, iType, WValues);
            R = Tensor(RShape, iType, RValues);
            B = Tensor(BShape, iType, BValues);
            P = Tensor(PShape, iType, PValues);
            Y = Tensor(YShape, oType, YValues);
            Ho = Tensor(HoShape, oType, HoValues);
            Co = Tensor(CoShape, oType, CoValues);
        }

    size_t batchSize;
    size_t inputSize;
    size_t hiddenSize;
    size_t seqLength;
    size_t numDirections;
    bool input_forget;
    op::RecurrentSequenceDirection lstm_direction;
    element::Type_t iType;
    element::Type_t oType;

    Tensor X;
    Tensor H_t;
    Tensor C_t;
    Tensor S_t;
    Tensor W;
    Tensor R;
    Tensor B;
    Tensor P;
    Tensor Y;
    Tensor Ho;
    Tensor Co;
    std::string testcaseName;
};

class ReferenceLSTMSequenceTest : public testing::TestWithParam<LSTMSequenceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data, params.S_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Y.data, params.Ho.data, params.Co.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType << "_";
        result << "xShape=" << param.X.shape << "_";
        result << "htShape=" << param.H_t.shape << "_";
        result << "ctShape=" << param.C_t.shape << "_";
        result << "stShape=" << param.S_t.shape << "_";
        result << "wShape=" << param.W.shape << "_";
        result << "rShape=" << param.R.shape << "_";
        result << "bShape=" << param.B.shape << "_";
        result << "YShape=" << param.Y.shape << "_";
        result << "hoShape=" << param.Ho.shape << "_";
        result << "coShape=" << param.Co.shape;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMSequenceParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto S_t = std::make_shared<op::v0::Parameter>(params.S_t.type, params.S_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_sequence =
            std::make_shared<op::v5::LSTMSequence>(X,
                                               H_t,
                                               C_t,
                                               S_t,
                                               W,
                                               R,
                                               B,
                                               params.hiddenSize,
                                               params.lstm_direction);

        auto function = std::make_shared<Model>(lstm_sequence->outputs(), ParameterVector{X, H_t, C_t, S_t, W, R, B});
        return function;
    }
};

class ReferenceLSTMSequenceV1Test : public testing::TestWithParam<LSTMSequenceV1Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data, params.S_t.data, params.W.data, params.R.data, params.B.data, params.P.data};
        refOutData = {params.Y.data, params.Ho.data, params.Co.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceV1Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType << "_";
        result << "xShape=" << param.X.shape << "_";
        result << "htShape=" << param.H_t.shape << "_";
        result << "ctShape=" << param.C_t.shape << "_";
        result << "stShape=" << param.S_t.shape << "_";
        result << "wShape=" << param.W.shape << "_";
        result << "rShape=" << param.R.shape << "_";
        result << "bShape=" << param.B.shape << "_";
        result << "pShape=" << param.P.shape << "_";
        result << "YShape=" << param.Y.shape << "_";
        result << "hoShape=" << param.Ho.shape << "_";
        result << "coShape=" << param.Co.shape;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMSequenceV1Params& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto S_t = std::make_shared<op::v0::Parameter>(params.S_t.type, params.S_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto P = std::make_shared<op::v0::Parameter>(params.P.type, params.P.shape);

        const auto lstm_sequence =
            std::make_shared<op::v0::LSTMSequence>(X,
                                               H_t,
                                               C_t,
                                               S_t,
                                               W,
                                               R,
                                               B,
                                               P,
                                               params.hiddenSize,
                                               params.lstm_direction,
                                               ov::op::LSTMWeightsFormat::FICO,
                                               std::vector<float>{},
                                               std::vector<float>{},
                                               std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                               0.f,
                                               params.input_forget);

        auto function = std::make_shared<Model>(lstm_sequence->outputs(), ParameterVector{X, H_t, C_t, S_t, W, R, B, P});
        return function;
    }
};

TEST_P(ReferenceLSTMSequenceTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceLSTMSequenceV1Test, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<LSTMSequenceParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;

    size_t batchSize = 10;
    size_t inputSize = 10;
    size_t hiddenSize = 1;
    size_t seqLength = 2;
    op::RecurrentSequenceDirection lstm_direction = op::RecurrentSequenceDirection::FORWARD;
    size_t numDirections = (lstm_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

    Shape XShape = Shape{batchSize, seqLength, inputSize};
    Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
    Shape C_tShape = Shape{batchSize, numDirections, hiddenSize};
    Shape S_tShape = Shape{batchSize};
    Shape WShape = Shape{numDirections, 4 * hiddenSize, inputSize};
    Shape RShape = Shape{numDirections, 4 * hiddenSize, hiddenSize};
    Shape BShape = Shape{numDirections, 4 * hiddenSize};
    Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
    Shape HoShape = Shape{batchSize, numDirections, hiddenSize};
    Shape CoShape = Shape{batchSize, numDirections, hiddenSize};

    std::vector<T> X = NGraphFunctions::Utils::generateVector<ET>(shape_size(XShape));
    std::vector<T> H = NGraphFunctions::Utils::generateVector<ET>(shape_size(H_tShape));
    std::vector<T> C = NGraphFunctions::Utils::generateVector<ET>(shape_size(C_tShape));
    std::vector<int64_t> S = NGraphFunctions::Utils::generateVector<element::Type_t::i64>(shape_size(S_tShape), seqLength);
    std::vector<T> W = NGraphFunctions::Utils::generateVector<ET>(shape_size(WShape));
    std::vector<T> R = NGraphFunctions::Utils::generateVector<ET>(shape_size(RShape));
    std::vector<T> B = NGraphFunctions::Utils::generateVector<ET>(shape_size(BShape));

    std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"};
    std::vector<T> Y(shape_size(YShape));
    std::vector<T> Ho(shape_size(HoShape));
    std::vector<T> Co(shape_size(CoShape));

    ngraph::runtime::reference::lstm_sequence<T, int64_t>(reinterpret_cast<const char*>(X.data()),
                   XShape,
                   reinterpret_cast<const char*>(H.data()),
                   H_tShape,
                   reinterpret_cast<const char*>(C.data()),
                   C_tShape,
                   reinterpret_cast<const char*>(S.data()),
                   S_tShape,
                   reinterpret_cast<const char*>(W.data()),
                   WShape,
                   reinterpret_cast<const char*>(R.data()),
                   RShape,
                   reinterpret_cast<const char*>(B.data()),
                   BShape,
                   reinterpret_cast<char*>(Y.data()),
                   reinterpret_cast<char*>(Ho.data()),
                   reinterpret_cast<char*>(Co.data()),
                   activations[0],
                   activations[1],
                   activations[2],
                   0.f,
                   op::RecurrentSequenceDirection::FORWARD);

    std::vector<LSTMSequenceParams> params {
        LSTMSequenceParams(
            batchSize, inputSize, hiddenSize, seqLength,
            lstm_direction,
            ET,
            X,
            H,
            C,
            S,
            W,
            R,
            B,
            Y,
            Ho,
            Co),
    };
    return params;
}


template <element::Type_t ET>
std::vector<LSTMSequenceV1Params> generateV1Params() {
    using T = typename element_type_traits<ET>::value_type;

    size_t batchSize = 10;
    size_t inputSize = 10;
    size_t hiddenSize = 1;
    size_t seqLength = 2;
    bool input_forget = false;
    op::RecurrentSequenceDirection lstm_direction = op::RecurrentSequenceDirection::FORWARD;
    size_t numDirections = (lstm_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

    Shape XShape = Shape{batchSize, seqLength, inputSize};
    Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
    Shape C_tShape = Shape{batchSize, numDirections, hiddenSize};
    Shape S_tShape = Shape{batchSize};
    Shape WShape = Shape{numDirections, 4 * hiddenSize, inputSize};
    Shape RShape = Shape{numDirections, 4 * hiddenSize, hiddenSize};
    Shape BShape = Shape{numDirections, 4 * hiddenSize};
    Shape PShape = Shape{numDirections, 3 * hiddenSize};
    Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
    Shape HoShape = Shape{batchSize, numDirections, hiddenSize};
    Shape CoShape = Shape{batchSize, numDirections, hiddenSize};

    std::vector<T> X = NGraphFunctions::Utils::generateVector<ET>(shape_size(XShape));
    std::vector<T> H = NGraphFunctions::Utils::generateVector<ET>(shape_size(H_tShape));
    std::vector<T> C = NGraphFunctions::Utils::generateVector<ET>(shape_size(C_tShape));
    std::vector<int64_t> S = NGraphFunctions::Utils::generateVector<element::Type_t::i64>(shape_size(S_tShape), seqLength);
    std::vector<T> W = NGraphFunctions::Utils::generateVector<ET>(shape_size(WShape));
    std::vector<T> R = NGraphFunctions::Utils::generateVector<ET>(shape_size(RShape));
    std::vector<T> B = NGraphFunctions::Utils::generateVector<ET>(shape_size(BShape));
    std::vector<T> P(shape_size(PShape), 0.f);

    std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"};
    std::vector<T> Y(shape_size(YShape));
    std::vector<T> Ho(shape_size(HoShape));
    std::vector<T> Co(shape_size(CoShape));

    ngraph::runtime::reference::lstm_sequence_v1<T, int64_t>(reinterpret_cast<const char*>(X.data()),
                   XShape,
                   reinterpret_cast<const char*>(H.data()),
                   H_tShape,
                   reinterpret_cast<const char*>(C.data()),
                   C_tShape,
                   reinterpret_cast<const char*>(S.data()),
                   S_tShape,
                   reinterpret_cast<const char*>(W.data()),
                   WShape,
                   reinterpret_cast<const char*>(R.data()),
                   RShape,
                   reinterpret_cast<const char*>(B.data()),
                   BShape,
                   reinterpret_cast<const char*>(P.data()),
                   PShape,
                   reinterpret_cast<char*>(Y.data()),
                   reinterpret_cast<char*>(Ho.data()),
                   reinterpret_cast<char*>(Co.data()),
                   activations[0],
                   activations[1],
                   activations[2],
                   0.f,
                   input_forget,
                   op::RecurrentSequenceDirection::FORWARD);

    std::vector<LSTMSequenceV1Params> params {
        LSTMSequenceV1Params(
            batchSize, inputSize, hiddenSize, seqLength,
            input_forget, lstm_direction,
            ET,
            X,
            H,
            C,
            S,
            W,
            R,
            B,
            P,
            Y,
            Ho,
            Co),
    };
    return params;
}

std::vector<LSTMSequenceParams> generateCombinedParams() {
    const std::vector<std::vector<LSTMSequenceParams>> generatedParams {
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<LSTMSequenceParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<LSTMSequenceV1Params> generateV1CombinedParams() {
    const std::vector<std::vector<LSTMSequenceV1Params>> generatedParams {
        generateV1Params<element::Type_t::bf16>(),
        generateV1Params<element::Type_t::f16>(),
        generateV1Params<element::Type_t::f32>(),
        generateV1Params<element::Type_t::f64>(),
    };
    std::vector<LSTMSequenceV1Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequence_With_Hardcoded_Refs, ReferenceLSTMSequenceTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceLSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequence_With_Hardcoded_Refs, ReferenceLSTMSequenceV1Test,
    testing::ValuesIn(generateV1CombinedParams()), ReferenceLSTMSequenceV1Test::getTestCaseName);

} // namespace