// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

enum DYNAMISM_MODE {
    BOTH_INPUTS_DYNAMIC,
    A_INPUT_DYNAMIC,
    B_INPUT_DYNAMIC
};

struct MatMul_input_setup {
    ngraph::Shape shape;
    bool transpose;
    // data for MatMul shape normalization and calculation
    uint64_t rank_diff;
    std::vector<int64_t> gather_idxs_for_transpose, batch_gather_idxs;
    int64_t channel_idx;
};

struct MatMulTestCase {
    MatMul_input_setup A, B;
};

const auto combinations = testing::Combine(
        testing::Values(
                DYNAMISM_MODE::BOTH_INPUTS_DYNAMIC,
                DYNAMISM_MODE::A_INPUT_DYNAMIC,
                DYNAMISM_MODE::B_INPUT_DYNAMIC),
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
// JIRA: 33925           MatMulTestCase{{{1024}, false, 1, {}, {}, 0}, {{1024, 1000}, false, 0, {}, {}, 1}},
// JIRA: 33925           MatMulTestCase{{{1024}, true, 1, {1, 0}, {}, 0}, {{1, 1000}, false, 0, {}, {}, 1}},
                MatMulTestCase{{{5, 10, 1024}, false, 0, {}, {0}, 1}, {{1024, 1000}, false, 1, {}, {0}, 2}},
                MatMulTestCase{{{5, 10, 1024}, false, 0, {}, {0}, 1}, {{1, 1024, 1000}, false, 0, {}, {0}, 2}},
                MatMulTestCase{{{5, 1024, 10}, true, 0, {0, 2, 1}, {0}, 1}, {{1, 1000, 1024}, true, 0, {0, 2, 1}, {0}, 2}},
                MatMulTestCase{{{3, 1024, 10}, true, 1, {0, 1, 3, 2}, {0, 1}, 2}, {{5, 1, 1000, 1024}, true, 0, {0, 1, 3, 2}, {0, 1}, 3}}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

using Parameters = std::tuple<
    DYNAMISM_MODE,
    DataType,
    MatMulTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_MatMul : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    ngraph::ParameterVector setting_up_input_dynamism(
            const DYNAMISM_MODE mode,
            const std::shared_ptr<ngraph::opset3::Parameter> input_A,
            const std::shared_ptr<ngraph::opset3::Parameter> input_B,
            std::shared_ptr<ngraph::Node>& renewed_input_A,
            std::shared_ptr<ngraph::Node>& renewed_input_B,
            std::shared_ptr<ngraph::Node>& A_shape_node,
            std::shared_ptr<ngraph::Node>& B_shape_node) const {
        ngraph::ParameterVector parameters{input_A, input_B};

        auto input_A_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_A->get_shape().size()});
        auto input_B_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_B->get_shape().size()});

        auto dsr_A = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input_A, input_A_dsr);
        auto dsr_B = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input_B, input_B_dsr);

        switch (mode) {
            case DYNAMISM_MODE::BOTH_INPUTS_DYNAMIC: {
                parameters.push_back(input_A_dsr);
                parameters.push_back(input_B_dsr);
                renewed_input_A = dsr_A;
                renewed_input_B = dsr_B;
                A_shape_node = input_A_dsr;
                B_shape_node = input_B_dsr;
                break;
            }
            case DYNAMISM_MODE::A_INPUT_DYNAMIC: {
                parameters.push_back(input_A_dsr);
                renewed_input_A = dsr_A;
                renewed_input_B = input_B;
                A_shape_node = input_A_dsr;
                B_shape_node = ngraph::opset3::Constant::create(ngraph::element::i64, {input_B->get_shape().size()}, input_B->get_shape());
                break;
            }
            case DYNAMISM_MODE::B_INPUT_DYNAMIC: {
                parameters.push_back(input_B_dsr);
                renewed_input_A = input_A;
                renewed_input_B = dsr_B;
                A_shape_node = ngraph::opset3::Constant::create(ngraph::element::i64, {input_A->get_shape().size()}, input_A->get_shape());
                B_shape_node = input_B_dsr;
                break;
            }
            default:
                NGRAPH_UNREACHABLE("UNKNOWN DYNAMISM MODE for MatMul DSR graph comparison test");
        }
        return parameters;
    }

    void SetUp() override {
        const auto& params = GetParam();
        const auto& mode = std::get<0>(params);
        const auto& data_type = std::get<1>(params);
        const auto& matmul_setup = std::get<2>(params);
        targetDevice = std::get<3>(params);

        auto input_A = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.A.shape);
        auto input_B = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.B.shape);

        std::shared_ptr<ngraph::Node> explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape;
        const auto parameters = setting_up_input_dynamism(mode, input_A, input_B, explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape);
        const auto node = std::make_shared<ngraph::opset3::MatMul>(explicit_A_input, explicit_B_input, matmul_setup.A.transpose, matmul_setup.B.transpose);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, parameters, "DSR-MatMul");
    }
};

TEST_P(DSR_MatMul, CompareWithReference) {
    Run();
}
// JIRA: 33997
INSTANTIATE_TEST_CASE_P(DISABLED_DynamicMatMul, DSR_MatMul, combinations);
}  // namespace
