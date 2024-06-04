// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
struct InverseParams {
    InverseParams(const reference_tests::Tensor& matrices,
                  const reference_tests::Tensor& expected_tensor,
                  bool adjoint,
                  std::string name)
        : matrices{matrices},
          expected_tensor(expected_tensor),
          adjoint(adjoint),
          test_case_name{std::move(name)} {}

    reference_tests::Tensor matrices;
    reference_tests::Tensor expected_tensor;
    bool adjoint;
    std::string test_case_name;
};

class ReferenceInverse : public testing::TestWithParam<InverseParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.matrices.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InverseParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_input_type_";
        name << obj.param.matrices.type;
        name << "_shape_";
        name << obj.param.matrices.shape;
        name << "_adjoint_";
        name << obj.param.adjoint;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const InverseParams& params) {
        const auto in_matrices = std::make_shared<ov::op::v0::Parameter>(params.matrices.type, params.matrices.shape);
        const auto inverse = std::make_shared<ov::op::v14::Inverse>(in_matrices, params.adjoint);
        return std::make_shared<ov::Model>(inverse->outputs(), ov::ParameterVector{in_matrices});
    }
};

template <ov::element::Type_t ET>
std::vector<InverseParams> generateInverseParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;

    const ov::Shape matrices_2_2_shape{2, 2};
    const ov::Shape matrices_4_4_shape{4, 4};
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

    reference_tests::Tensor matrices_4_4_v1(
        matrices_4_4_shape,
        ET,
        std::vector<VT>{7, -2, 5, 8, -6, 3, -2, 27, 10, -12, 23, 21, 1, -21, 16, 15});
    reference_tests::Tensor matrices_4_4_v2(
        matrices_4_4_shape,
        ET,
        std::vector<
            VT>{5.0f, 6.0f, 6.0f, 8.0f, 2.0f, 2.0f, 2.0f, 8.0f, 6.0f, 6.0f, 2.0f, 8.0f, 2.0f, 3.0f, 6.0f, 7.0f});

    reference_tests::Tensor output_2_2_no_adjoint(matrices_2_2_shape, ET, std::vector<VT>{-1.0f, 0.5f, 1.5f, -0.25f});
    reference_tests::Tensor output_2_2_adjoint(matrices_2_2_shape, ET, std::vector<VT>{-1.0f, 1.5f, 0.5f, -0.25f});

    reference_tests::Tensor output_2_3_3_no_adjoint(matrices_2_3_3_shape,
                                                    ET,
                                                    std::vector<VT>{0.75f,
                                                                    0.5f,
                                                                    0.25f,
                                                                    0.5f,
                                                                    1.0f,
                                                                    0.5f,
                                                                    0.25f,
                                                                    0.5f,
                                                                    0.75f,

                                                                    -0.25f,
                                                                    0.5f,
                                                                    0.875f,
                                                                    -0.25f,
                                                                    0.5f,
                                                                    0.378f,
                                                                    1.0f,
                                                                    -1.0f,
                                                                    -1.5f});
    reference_tests::Tensor output_2_3_3_adjoint(matrices_2_3_3_shape,
                                                 ET,
                                                 std::vector<VT>{0.75f,
                                                                 0.5f,
                                                                 0.25f,
                                                                 0.5f,
                                                                 1.0f,
                                                                 0.5f,
                                                                 0.25f,
                                                                 0.5f,
                                                                 0.75f,
                                                                 -0.25f,
                                                                 -0.25f,
                                                                 1.0f,
                                                                 0.5f,
                                                                 0.5f,
                                                                 -1.0f,
                                                                 0.875f,
                                                                 0.378f,
                                                                 -1.5f});

    reference_tests::Tensor output_4_4_v1_no_adjoint(matrices_4_4_shape,
                                                     ET,
                                                     std::vector<VT>{0.190005f,
                                                                     -0.0227165f,
                                                                     -0.047196f,
                                                                     0.00562826f,
                                                                     -0.0882213f,
                                                                     0.0116182f,
                                                                     0.0768744f,
                                                                     -0.0814855f,
                                                                     -0.164983f,
                                                                     -0.0113243f,
                                                                     0.113786f,
                                                                     -0.0509256f,
                                                                     0.0398047f,
                                                                     0.0298592f,
                                                                     -0.010601f,
                                                                     0.0065324f});
    reference_tests::Tensor output_4_4_v1_adjoint(matrices_4_4_shape,
                                                  ET,
                                                  std::vector<VT>{0.190005f,
                                                                  -0.0882213f,
                                                                  -0.164983f,
                                                                  0.0398047f,
                                                                  -0.0227165f,
                                                                  0.0116182f,
                                                                  -0.0113243f,
                                                                  0.0298592f,
                                                                  -0.047196f,
                                                                  0.0768744f,
                                                                  0.113786f,
                                                                  -0.010601f,
                                                                  0.00562826f,
                                                                  -0.0814855f,
                                                                  -0.0509256f,
                                                                  0.0065324f});

    reference_tests::Tensor output_4_4_v2_no_adjoint(matrices_4_4_shape,
                                                     ET,
                                                     std::vector<VT>{-17.0f,
                                                                     -9.0f,
                                                                     12.0f,
                                                                     16.0f,
                                                                     17.0f,
                                                                     8.75f,
                                                                     -11.75f,
                                                                     -16.0f,
                                                                     -4.0f,
                                                                     -2.25f,
                                                                     2.75f,
                                                                     4.0f,
                                                                     1.0f,
                                                                     0.75f,
                                                                     -0.75f,
                                                                     -1.0f});
    reference_tests::Tensor output_4_4_v2_adjoint(matrices_4_4_shape,
                                                  ET,
                                                  std::vector<VT>{-17.0f,
                                                                  17.0f,
                                                                  -4.0f,
                                                                  1.0f,
                                                                  -9.0f,
                                                                  8.75f,
                                                                  -2.25f,
                                                                  0.75f,
                                                                  12.0f,
                                                                  -11.75f,
                                                                  2.75f,
                                                                  -0.75f,
                                                                  16.0f,
                                                                  -16.0f,
                                                                  4.0f,
                                                                  -1.0f});

    std::vector<InverseParams> params;
    params.emplace_back(matrices_2_2, output_2_2_no_adjoint, false, "single_simple");
    params.emplace_back(matrices_2_2, output_2_2_adjoint, true, "single_simple");
    params.emplace_back(matrices_2_3_3, output_2_3_3_no_adjoint, false, "many_simple");
    params.emplace_back(matrices_2_3_3, output_2_3_3_adjoint, true, "many_simple");
    params.emplace_back(matrices_4_4_v1, output_4_4_v1_no_adjoint, false, "single_complex");
    params.emplace_back(matrices_4_4_v1, output_4_4_v1_adjoint, true, "single_complex");
    params.emplace_back(matrices_4_4_v2, output_4_4_v2_no_adjoint, false, "single_complex_cpu");
    params.emplace_back(matrices_4_4_v2, output_4_4_v2_adjoint, true, "single_complex_cpu");
    return params;
}

std::vector<InverseParams> generateInverseParams() {
    std::vector<std::vector<InverseParams>> combo_params{generateInverseParams<ov::element::f64>(),
                                                         generateInverseParams<ov::element::f32>(),
                                                         generateInverseParams<ov::element::f16>(),
                                                         generateInverseParams<ov::element::bf16>()};
    std::vector<InverseParams> test_params;
    for (auto& params : combo_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceInverse, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceInverse,
                         ::testing::ValuesIn(generateInverseParams()),
                         ReferenceInverse::getTestCaseName);
