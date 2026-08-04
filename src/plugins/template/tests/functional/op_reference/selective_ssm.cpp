// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct SelectiveSSMParams {
    PartialShape AShape;
    PartialShape dtShape;
    PartialShape BShape;
    PartialShape xShape;
    PartialShape CShape;
    PartialShape stateShape;
    std::string testcaseName;
    reference_tests::Tensor AData;
    reference_tests::Tensor dtData;
    reference_tests::Tensor BData;
    reference_tests::Tensor xData;
    reference_tests::Tensor CData;
    reference_tests::Tensor stateData;
    reference_tests::Tensor expectedOutput;
    reference_tests::Tensor expectedState;
};

template <typename T>
SelectiveSSMParams prepare_case(const std::vector<T>& A,
                                const std::vector<T>& dt,
                                const std::vector<T>& B,
                                const std::vector<T>& x,
                                const std::vector<T>& C,
                                const std::vector<T>& state,
                                const std::vector<T>& out,
                                const std::vector<T>& out_state,
                                const std::string& name) {
    SelectiveSSMParams ret;
    const auto et = element::from<T>();
    ret.AShape = PartialShape{2};
    ret.dtShape = PartialShape{1, 2, 2};
    ret.BShape = PartialShape{1, 2, 1, 2};
    ret.xShape = PartialShape{1, 2, 2, 2};
    ret.CShape = PartialShape{1, 2, 1, 2};
    ret.stateShape = PartialShape{1, 2, 2, 2};
    ret.testcaseName = name;
    ret.AData = reference_tests::Tensor(et, ret.AShape.to_shape(), A);
    ret.dtData = reference_tests::Tensor(et, ret.dtShape.to_shape(), dt);
    ret.BData = reference_tests::Tensor(et, ret.BShape.to_shape(), B);
    ret.xData = reference_tests::Tensor(et, ret.xShape.to_shape(), x);
    ret.CData = reference_tests::Tensor(et, ret.CShape.to_shape(), C);
    ret.stateData = reference_tests::Tensor(et, ret.stateShape.to_shape(), state);
    ret.expectedOutput = reference_tests::Tensor(et, ret.xShape.to_shape(), out);
    ret.expectedState = reference_tests::Tensor(et, ret.stateShape.to_shape(), out_state);
    return ret;
}

class ReferenceSelectiveSSMTest : public testing::TestWithParam<SelectiveSSMParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& p = GetParam();
        function = CreateFunction(p);
        inputData = {p.AData.data, p.dtData.data, p.BData.data, p.xData.data, p.CData.data, p.stateData.data};
        refOutData = {p.expectedOutput.data, p.expectedState.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SelectiveSSMParams>& obj) {
        return obj.param.testcaseName;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SelectiveSSMParams& p) {
        const auto et = p.AData.data.get_element_type();
        auto A = std::make_shared<op::v0::Parameter>(et, p.AShape);
        auto dt = std::make_shared<op::v0::Parameter>(et, p.dtShape);
        auto B = std::make_shared<op::v0::Parameter>(et, p.BShape);
        auto x = std::make_shared<op::v0::Parameter>(et, p.xShape);
        auto C = std::make_shared<op::v0::Parameter>(et, p.CShape);
        auto state = std::make_shared<op::v0::Parameter>(et, p.stateShape);
        auto ssm = std::make_shared<op::internal::SelectiveSSM>(A, dt, B, x, C, state);
        return std::make_shared<Model>(OutputVector{ssm->output(0), ssm->output(1)}, ParameterVector{A, dt, B, x, C, state});
    }
};

TEST_P(ReferenceSelectiveSSMTest, CompareWithHardcodedRefs) {
    Exec();
}

std::vector<SelectiveSSMParams> generate_params() {
    // Manually computed reference for A=[-1,-2], grouped B/C shared by both heads.
    return {prepare_case<float>({-1.f, -2.f},
                                {1.f, 0.5f, 2.f, 1.f},
                                {2.f, 3.f, 4.f, 5.f},
                                {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f},
                                {0.25f, 0.5f, 0.75f, 1.25f},
                                std::vector<float>(8, 0.f),
                                {2.f, 4.f, 3.f, 4.f, 93.21051f, 112.42102f, 65.815765f, 75.42102f},
                                {40.27067f, 50.406006f, 48.54134f, 60.81201f, 28.406006f, 35.60901f, 32.54134f, 40.81201f},
                                "basic_grouped_case")};
}

INSTANTIATE_TEST_SUITE_P(smoke_SelectiveSSM,
                         ReferenceSelectiveSSMTest,
                         ::testing::ValuesIn(generate_params()),
                         ReferenceSelectiveSSMTest::getTestCaseName);

}  // namespace
