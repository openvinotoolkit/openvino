// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
struct IdentityParams {
    IdentityParams(const reference_tests::Tensor& matrices, bool copy, std::string name)
        : matrices{matrices},
          test_case_name{std::move(name)} {}

    reference_tests::Tensor matrices;
    std::string test_case_name;
};

class ReferenceIdentity : public testing::TestWithParam<IdentityParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.matrices.data};
        refOutData = {params.matrices.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<IdentityParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_input_type_";
        name << obj.param.matrices.type;
        name << "_shape_";
        name << obj.param.matrices.shape;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const IdentityParams& params) {
        const auto in_matrices = std::make_shared<ov::op::v0::Parameter>(params.matrices.type, params.matrices.shape);
        const auto identity = std::make_shared<ov::op::v15::Identity>(in_matrices);
        return std::make_shared<ov::Model>(identity->outputs(), ov::ParameterVector{in_matrices});
    }
};

template <ov::element::Type_t ET>
std::vector<IdentityParams> generateIdentityParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;

    const ov::Shape matrices_2_2_shape{2, 2};
    const ov::Shape matrices_2_3_3_shape{2, 3, 3};

    reference_tests::Tensor matrices_2_2(matrices_2_2_shape, ET, std::vector<VT>{0.5f, 1.0f, 3.0f, 2.0f});

    reference_tests::Tensor matrices_2_3_3(matrices_2_3_3_shape,
                                           ET,
                                           std::vector<VT>{2.0f,
                                                           -1.0f,
                                                           0.0f,
                                                           -1.0f,
                                                           2.0f,
                                                           -1.0f,
                                                           0.0f,
                                                           -1.0f,
                                                           2.0f,

                                                           3.0f,
                                                           1.0f,
                                                           2.0f,
                                                           0.0f,
                                                           4.0f,
                                                           1.0f,
                                                           2.0f,
                                                           -2.0f,
                                                           0.0f});

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
                                                          generateIdentityParams<ov::element::i4>(),
                                                          generateIdentityParams<ov::element::i8>(),
                                                          generateIdentityParams<ov::element::i16>(),
                                                          generateIdentityParams<ov::element::i32>(),
                                                          generateIdentityParams<ov::element::i64>(),
                                                          generateIdentityParams<ov::element::u1>(),
                                                          generateIdentityParams<ov::element::u4>(),
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
