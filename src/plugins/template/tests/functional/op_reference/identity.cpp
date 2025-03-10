// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
using IdentityParams = std::tuple<reference_tests::Tensor, std::string>;

class ReferenceIdentity : public testing::TestWithParam<IdentityParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto& matrices = std::get<0>(params);
        function = CreateFunction(params);
        inputData = {matrices.data};
        refOutData = {matrices.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<IdentityParams>& obj) {
        std::ostringstream name;
        const auto& matrices = std::get<0>(obj.param);
        name << std::get<1>(obj.param);
        name << "_input_type_";
        name << matrices.type;
        name << "_shape_";
        name << matrices.shape;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const IdentityParams& params) {
        const auto& matrices = std::get<0>(params);
        const auto in_matrices = std::make_shared<ov::op::v0::Parameter>(matrices.type, matrices.shape);
        const auto identity = std::make_shared<ov::op::v16::Identity>(in_matrices);
        return std::make_shared<ov::Model>(identity->outputs(), ov::ParameterVector{in_matrices});
    }
};

template <ov::element::Type_t ET>
std::vector<IdentityParams> generateIdentityParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;

    const ov::Shape matrices_2_2_shape{2, 2};
    const ov::Shape matrices_2_3_3_shape{2, 3, 3};

    reference_tests::Tensor matrices_2_2(matrices_2_2_shape, ET, std::vector<VT>{1, 2, 4, 5});

    reference_tests::Tensor matrices_2_3_3(matrices_2_3_3_shape,
                                           ET,
                                           std::vector<VT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 4, 5, 3, 1, 2, 6});

    std::vector<IdentityParams> params;
    params.emplace_back(matrices_2_2, "single");
    params.emplace_back(matrices_2_3_3, "many");

    return params;
}

std::vector<IdentityParams> generateIdentityParams() {
    std::vector<std::vector<IdentityParams>> combo_params{generateIdentityParams<ov::element::boolean>(),
                                                          generateIdentityParams<ov::element::bf16>(),
                                                          generateIdentityParams<ov::element::f16>(),
                                                          generateIdentityParams<ov::element::f64>(),
                                                          generateIdentityParams<ov::element::f32>(),
                                                          generateIdentityParams<ov::element::i8>(),
                                                          generateIdentityParams<ov::element::i16>(),
                                                          generateIdentityParams<ov::element::i32>(),
                                                          generateIdentityParams<ov::element::i64>(),
                                                          generateIdentityParams<ov::element::u8>(),
                                                          generateIdentityParams<ov::element::u16>(),
                                                          generateIdentityParams<ov::element::u32>(),
                                                          generateIdentityParams<ov::element::u64>()};
    std::vector<IdentityParams> test_params;
    for (auto& params : combo_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceIdentity, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceIdentity,
                         ::testing::ValuesIn(generateIdentityParams()),
                         ReferenceIdentity::getTestCaseName);
