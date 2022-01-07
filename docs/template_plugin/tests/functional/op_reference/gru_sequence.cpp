// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/gru_sequence.hpp"
#include "base_reference_test.hpp"
#include "ngraph_functions/utils/data_utils.hpp"
#include "ngraph/runtime/reference/sequences.hpp"


using namespace reference_tests;
using namespace ov;

namespace {
struct GRUSequenceParams {
    template <class T>
    GRUSequenceParams(
        const size_t batchSize, const size_t inputSize, const size_t hiddenSize, const size_t seqLength,
        const float clip, const bool linear_before_reset, const op::RecurrentSequenceDirection& gru_direction,
        const element::Type_t& iType,
        const std::vector<T>& XValues, const std::vector<T>& H_tValues, const std::vector<int64_t>& S_tValues,
        const std::vector<T>& WValues, const std::vector<T>& RValues, const std::vector<T>& BValues,
        const std::vector<T>& YValues, const std::vector<T>& HoValues,
        const std::string& testcaseName = "") :
        batchSize(batchSize), inputSize(inputSize), hiddenSize(hiddenSize), seqLength(seqLength),
        clip(clip), linear_before_reset(linear_before_reset), gru_direction(gru_direction), iType(iType), oType(iType),
        testcaseName(testcaseName) {
            numDirections = (gru_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

            Shape XShape = Shape{batchSize, seqLength, inputSize};
            Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
            Shape S_tShape = Shape{batchSize};
            Shape WShape = Shape{numDirections, 3 * hiddenSize, inputSize};
            Shape RShape = Shape{numDirections, 3 * hiddenSize, hiddenSize};
            Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
            Shape HoShape = Shape{batchSize, numDirections, hiddenSize};

            X = Tensor(XShape, iType, XValues);
            H_t = Tensor(H_tShape, iType, H_tValues);
            S_t = Tensor(S_tShape, element::Type_t::i64, S_tValues);
            W = Tensor(WShape, iType, WValues);
            R = Tensor(RShape, iType, RValues);
            Y = Tensor(YShape, oType, YValues);
            Ho = Tensor(HoShape, oType, HoValues);

            if (linear_before_reset == true) {
                Shape BShape = Shape{numDirections, 4 * hiddenSize};
                B = Tensor(BShape, iType, BValues);
            } else {
                Shape BShape = Shape{numDirections, 3 * hiddenSize};
                B = Tensor(BShape, iType, BValues);
            }
        }

    size_t batchSize;
    size_t inputSize;
    size_t hiddenSize;
    size_t seqLength;
    size_t numDirections;
    float clip;
    bool linear_before_reset;
    op::RecurrentSequenceDirection gru_direction;
    element::Type_t iType;
    element::Type_t oType;

    Tensor X;
    Tensor H_t;
    Tensor S_t;
    Tensor W;
    Tensor R;
    Tensor B;
    Tensor Y;
    Tensor Ho;
    std::string testcaseName;
};

class ReferenceGRUSequenceTest : public testing::TestWithParam<GRUSequenceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.S_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Y.data, params.Ho.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GRUSequenceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType << "_";
        result << "xShape=" << param.X.shape << "_";
        result << "htShape=" << param.H_t.shape << "_";
        result << "stShape=" << param.S_t.shape << "_";
        result << "wShape=" << param.W.shape << "_";
        result << "rShape=" << param.R.shape << "_";
        result << "bShape=" << param.B.shape << "_";
        result << "YShape=" << param.Y.shape << "_";
        result << "hoShape=" << param.Ho.shape << "_";
        result << "clip=" << param.clip << "_";
        result << "linear_before_reset=" << param.linear_before_reset << "_";
        result << "LSTMdirection=" << param.gru_direction;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GRUSequenceParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto S_t = std::make_shared<op::v0::Parameter>(params.S_t.type, params.S_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_sequence =
            std::make_shared<op::v5::GRUSequence>(X,
                                               H_t,
                                               S_t,
                                               W,
                                               R,
                                               B,
                                               params.hiddenSize,
                                               params.gru_direction,
                                               std::vector<std::string>{"sigmoid", "tanh"},
                                               std::vector<float>{},
                                               std::vector<float>{},
                                               params.clip,
                                               params.linear_before_reset);

        auto function = std::make_shared<Model>(lstm_sequence->outputs(), ParameterVector{X, H_t, S_t, W, R, B});
        return function;
    }
};

TEST_P(ReferenceGRUSequenceTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GRUSequenceParams> generateParamsRuntimeRef(size_t batchSize,
                                                         size_t inputSize,
                                                         size_t hiddenSize,
                                                         size_t seqLength,
                                                         float clip,
                                                         bool linear_before_reset,
                                                         op::RecurrentSequenceDirection gru_direction) {
    using T = typename element_type_traits<ET>::value_type;

    size_t numDirections = (gru_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

    Shape XShape = Shape{batchSize, seqLength, inputSize};
    Shape H_tShape = Shape{batchSize, numDirections, hiddenSize};
    Shape S_tShape = Shape{batchSize};
    Shape WShape = Shape{numDirections, 3 * hiddenSize, inputSize};
    Shape RShape = Shape{numDirections, 3 * hiddenSize, hiddenSize};
    Shape BShape;
    Shape YShape = Shape{batchSize, numDirections, seqLength, hiddenSize};
    Shape HoShape = Shape{batchSize, numDirections, hiddenSize};

    if (linear_before_reset == true) {
        BShape = Shape{numDirections, 4 * hiddenSize};
    } else {
        BShape = Shape{numDirections, 3 * hiddenSize};
    }

    std::vector<T> X = NGraphFunctions::Utils::generateVector<ET>(shape_size(XShape));
    std::vector<T> H = NGraphFunctions::Utils::generateVector<ET>(shape_size(H_tShape));
    std::vector<int64_t> S = NGraphFunctions::Utils::generateVector<element::Type_t::i64>(shape_size(S_tShape), seqLength);
    std::vector<T> W = NGraphFunctions::Utils::generateVector<ET>(shape_size(WShape));
    std::vector<T> R = NGraphFunctions::Utils::generateVector<ET>(shape_size(RShape));
    std::vector<T> B = NGraphFunctions::Utils::generateVector<ET>(shape_size(BShape));

    std::vector<std::string> activations = {"sigmoid", "tanh"};
    std::vector<T> Y(shape_size(YShape));
    std::vector<T> Ho(shape_size(HoShape));

    ngraph::runtime::reference::gru_sequence<T, int64_t>(reinterpret_cast<const char*>(X.data()),
                   XShape,
                   reinterpret_cast<const char*>(H.data()),
                   H_tShape,
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
                   activations[0],
                   activations[1],
                   clip,
                   gru_direction,
                   linear_before_reset);

    std::vector<GRUSequenceParams> params {
        GRUSequenceParams(
            batchSize, inputSize, hiddenSize, seqLength,
            clip, linear_before_reset, gru_direction,
            ET,
            X,
            H,
            S,
            W,
            R,
            B,
            Y,
            Ho),
    };
    return params;
}

std::vector<GRUSequenceParams> generateCombinedParams() {
    const std::vector<std::vector<GRUSequenceParams>> generatedParams {
        generateParamsRuntimeRef<element::Type_t::bf16>(10, 10, 3, 2, 0.f, false, op::RecurrentSequenceDirection::FORWARD),
        generateParamsRuntimeRef<element::Type_t::f16>(10, 10, 3, 2, 0.f, false, op::RecurrentSequenceDirection::FORWARD),
        generateParamsRuntimeRef<element::Type_t::f32>(10, 10, 3, 2, 0.f, false, op::RecurrentSequenceDirection::FORWARD),
        generateParamsRuntimeRef<element::Type_t::f64>(10, 10, 3, 2, 0.f, false, op::RecurrentSequenceDirection::FORWARD),
        generateParamsRuntimeRef<element::Type_t::bf16>(10, 10, 3, 2, 3.5f, false, op::RecurrentSequenceDirection::REVERSE),
        generateParamsRuntimeRef<element::Type_t::f16>(10, 10, 3, 2, 3.5f, false, op::RecurrentSequenceDirection::REVERSE),
        generateParamsRuntimeRef<element::Type_t::f32>(10, 10, 3, 2, 3.5f, false, op::RecurrentSequenceDirection::REVERSE),
        generateParamsRuntimeRef<element::Type_t::f64>(10, 10, 3, 2, 3.5f, false, op::RecurrentSequenceDirection::REVERSE),
        generateParamsRuntimeRef<element::Type_t::bf16>(10, 10, 10, 10, 0.f, true, op::RecurrentSequenceDirection::BIDIRECTIONAL),
        generateParamsRuntimeRef<element::Type_t::f16>(10, 10, 10, 10, 0.f, true, op::RecurrentSequenceDirection::BIDIRECTIONAL),
        generateParamsRuntimeRef<element::Type_t::f32>(10, 10, 10, 10, 0.f, true, op::RecurrentSequenceDirection::BIDIRECTIONAL),
        generateParamsRuntimeRef<element::Type_t::f64>(10, 10, 10, 10, 0.f, true, op::RecurrentSequenceDirection::BIDIRECTIONAL),
    };
    std::vector<GRUSequenceParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GRUSequence_With_Hardcoded_Refs, ReferenceGRUSequenceTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceGRUSequenceTest::getTestCaseName);

} // namespace