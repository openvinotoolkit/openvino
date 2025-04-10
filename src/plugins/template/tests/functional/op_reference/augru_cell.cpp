// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/augru_cell.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct AUGRUCellParams {
    AUGRUCellParams(const int32_t batchSize,
                    const int32_t inputSize,
                    const int32_t hiddenSize,
                    const int32_t gatesCount,
                    const reference_tests::Tensor& X,
                    const reference_tests::Tensor& H_t,
                    const reference_tests::Tensor& W,
                    const reference_tests::Tensor& R,
                    const reference_tests::Tensor& B,
                    const reference_tests::Tensor& A,
                    const reference_tests::Tensor& Ho,
                    const std::string& testcaseName = "")
        : batchSize(batchSize),
          inputSize(inputSize),
          hiddenSize(hiddenSize),
          gatesCount(gatesCount),
          X(X),
          H_t(H_t),
          W(W),
          R(R),
          B(B),
          A(A),
          Ho(Ho),
          testcaseName(testcaseName) {}

    int32_t batchSize;
    int32_t inputSize;
    int32_t hiddenSize;
    int32_t gatesCount;
    reference_tests::Tensor X;
    reference_tests::Tensor H_t;
    reference_tests::Tensor W;
    reference_tests::Tensor R;
    reference_tests::Tensor B;
    reference_tests::Tensor A;
    reference_tests::Tensor Ho;
    std::string testcaseName;
};

class ReferenceAUGRUCellTest : public testing::TestWithParam<AUGRUCellParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.W.data, params.R.data, params.B.data, params.A.data};
        refOutData = {params.Ho.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AUGRUCellParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_gCount=" << param.gatesCount;
        result << "_xType=" << param.X.type;
        result << "_xShape=" << param.X.shape;
        result << "_htType=" << param.H_t.type;
        result << "_htShape=" << param.H_t.shape;
        result << "_wType=" << param.W.type;
        result << "_wShape=" << param.W.shape;
        result << "_rType=" << param.R.type;
        result << "_rShape=" << param.R.shape;
        result << "_hoType=" << param.Ho.type;
        if (param.testcaseName != "") {
            result << "_hoShape=" << param.Ho.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_hoShape=" << param.Ho.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const AUGRUCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);

        const auto augru_cell = std::make_shared<ov::op::internal::AUGRUCell>(X, H_t, W, R, B, A, params.hiddenSize);

        auto function = std::make_shared<Model>(NodeVector{augru_cell}, ParameterVector{X, H_t, W, R, B, A});
        return function;
    }
};

TEST_P(ReferenceAUGRUCellTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<AUGRUCellParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<AUGRUCellParams> params{
        AUGRUCellParams(
            2,
            3,
            3,
            3,
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f}),
            reference_tests::Tensor(
                ET,
                {3 * 3, 3},
                std::vector<T>{0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
                               0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
                               0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
                               0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f,
                                                                   0.1623331f,  0.01389796f, 0.99669236f, 0.53901845f,
                                                                   0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f,
                                                                   0.40111724f, 0.37497616f, 0.2903471f,  0.6796794f,
                                                                   0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f,
                                                                   0.791677f,   0.76497287f, 0.9895242f,  0.7845312f,
                                                                   0.51267904f, 0.49030215f, 0.08498167f}),
            reference_tests::Tensor(ET,
                                    {3 * 3},
                                    std::vector<T>{0.8286678f + 0.9175602f,
                                                   0.9153158f + 0.14958014f,
                                                   0.9581612f + 0.49230585f,
                                                   0.6639213f + 0.63162816f,
                                                   0.84239805f + 0.4161903f,
                                                   0.5282445f + 0.22148274f,
                                                   0.14153397f + 0.50496656f,
                                                   0.22404431f + 0.34798595f,
                                                   0.6549655f + 0.6699164f}),
            reference_tests::Tensor(ET, {2, 1}, std::vector<T>(2, 0)),
            reference_tests::Tensor(ET,
                                    {2, 3},
                                    std::vector<T>{0.480763f, 0.996927f, 0.830836f, 0.50231f, 0.894105f, 0.58932f}),
            "augru_zero_attentional_gate"),
        AUGRUCellParams(
            2,
            3,
            3,
            3,
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f}),
            reference_tests::Tensor(
                ET,
                {3 * 3, 3},
                std::vector<T>{0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
                               0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
                               0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
                               0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f,
                                                                   0.1623331f,  0.01389796f, 0.99669236f, 0.53901845f,
                                                                   0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f,
                                                                   0.40111724f, 0.37497616f, 0.2903471f,  0.6796794f,
                                                                   0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f,
                                                                   0.791677f,   0.76497287f, 0.9895242f,  0.7845312f,
                                                                   0.51267904f, 0.49030215f, 0.08498167f}),
            reference_tests::Tensor(ET,
                                    {3 * 3},
                                    std::vector<T>{0.8286678f + 0.9175602f,
                                                   0.9153158f + 0.14958014f,
                                                   0.9581612f + 0.49230585f,
                                                   0.6639213f + 0.63162816f,
                                                   0.84239805f + 0.4161903f,
                                                   0.5282445f + 0.22148274f,
                                                   0.14153397f + 0.50496656f,
                                                   0.22404431f + 0.34798595f,
                                                   0.6549655f + 0.6699164f}),
            reference_tests::Tensor(ET, {2, 1}, std::vector<T>{0.4f, 0.7f}),
            reference_tests::Tensor(ET,
                                    {2, 3},
                                    std::vector<T>{0.686381f, 0.997266f, 0.897868f, 0.832165f, 0.963144f, 0.868483f}),
            "augru_attentional_gate_provided"),
        AUGRUCellParams(
            2,
            3,
            4,
            3,
            reference_tests::Tensor(ET, {1, 3}, std::vector<T>{0.64053001f, 0.84805253f, 0.84544252f}),
            reference_tests::Tensor(ET, {1, 4}, std::vector<T>{0.43200249f, 0.08097203f, 0.68151259f, 0.09095205f}),
            reference_tests::Tensor(
                ET,
                {3 * 4, 3},
                std::vector<T>{0.23187583f, 0.66675389f, 0.01945467f, 0.60630121f, 0.18400699f, 0.16003634f,
                               0.04636866f, 0.75745989f, 0.96562912f, 0.56330529f, 0.20863093f, 0.93179716f,
                               0.99211225f, 0.73087621f, 0.21175275f, 0.03808638f, 0.63130526f, 0.76965886f,
                               0.67656870f, 0.57886251f, 0.94375534f, 0.88943972f, 0.96256618f, 0.38204562f,
                               0.76424904f, 0.30076485f, 0.60250044f, 0.40778284f, 0.70017757f, 0.00410288f,
                               0.97978094f, 0.73106175f, 0.22250106f, 0.44011834f, 0.11434720f, 0.62128995}),
            reference_tests::Tensor(
                ET,
                {3 * 4, 4},
                std::vector<T>{0.60702709f, 0.47515485f, 0.26202747f, 0.53851601f, 0.73423241f, 0.11627945f,
                               0.04631785f, 0.43604361f, 0.12472080f, 0.47546322f, 0.23103632f, 0.36108585f,
                               0.45139418f, 0.79838954f, 0.28194170f, 0.76877929f, 0.28428253f, 0.13822001f,
                               0.51670576f, 0.80312243f, 0.11050813f, 0.19925340f, 0.29769184f, 0.78933459f,
                               0.79981487f, 0.55313454f, 0.04135296f, 0.50578146f, 0.76553680f, 0.44311704f,
                               0.30525652f, 0.26301583f, 0.41771479f, 0.18182059f, 0.11106816f, 0.67427757f,
                               0.59174944f, 0.13339960f, 0.33362533f, 0.78938375f, 0.99260256f, 0.86955733f,
                               0.24899024f, 0.87134874f, 0.02803802f, 0.61244129f, 0.40803782f, 0.90735816f,
                               0.51267904f, 0.49030215f, 0.08498167f}),
            reference_tests::Tensor(ET,
                                    {3 * 4},
                                    std::vector<T>{0.61387895f,
                                                   0.56121052f,
                                                   0.89328753f,
                                                   0.15302506f,
                                                   0.90491122f,
                                                   0.78289335f,
                                                   0.97930211f,
                                                   0.75002178f,
                                                   0.92500923f,
                                                   0.18957983f,
                                                   0.07849785f,
                                                   0.76568159f}),
            reference_tests::Tensor(ET, {1, 1}, std::vector<T>{0.3333f}),
            reference_tests::Tensor(ET, {1, 4}, std::vector<T>{0.666063f, 0.451451f, 0.792762f, 0.453281f}),
            "augru_different_input_and_hidden_size"),
    };
    return params;
}

std::vector<AUGRUCellParams> generateCombinedParams() {
    const std::vector<std::vector<AUGRUCellParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<AUGRUCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_AUGRUCell_With_Hardcoded_Refs,
                         ReferenceAUGRUCellTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceAUGRUCellTest::getTestCaseName);

}  // namespace
