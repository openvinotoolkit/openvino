// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include <common_test_utils/test_common.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_matmul.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

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
            MatMulTestCase{{{3, 1024, 10}, true, 1, {0, 1, 3, 2}, {0, 1}, 2}, {{5, 1, 1000, 1024}, true, 0, {0, 1, 3, 2}, {0, 1}, 3}}));


class DynamicToStaticShapeMatMul: public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DYNAMISM_MODE, ngraph::element::Type_t, MatMulTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& mode = std::get<0>(parameters);
        const auto& data_type = std::get<1>(parameters);
        const auto& matmul_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(mode, data_type, matmul_setup),
                                          *reference(mode, data_type, matmul_setup));
    }

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

    std::shared_ptr<const ngraph::Function> transform(
            const DYNAMISM_MODE mode,
            const ngraph::element::Type_t& data_type,
            const MatMulTestCase& matmul_setup) const {
        auto input_A = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.A.shape);
        auto input_B = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.B.shape);

        std::shared_ptr<ngraph::Node> explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape;
        const auto parameters = setting_up_input_dynamism(mode, input_A, input_B, explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape);
        const auto node = std::make_shared<ngraph::opset3::MatMul>(explicit_A_input, explicit_B_input, matmul_setup.A.transpose, matmul_setup.B.transpose);

        const auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{node}, parameters, "Actual");
        node->set_output_type(0, node->get_output_element_type(0), ngraph::PartialShape::dynamic(node->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{ngraph::opset3::MatMul::type_info, vpu::dynamicToStaticShapeMatMul}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<ngraph::Function> reference(
            const DYNAMISM_MODE mode,
            const ngraph::element::Type_t& data_type,
            const MatMulTestCase& matmul_setup) {
        auto input_A = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.A.shape);
        auto input_B = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.B.shape);
        std::shared_ptr<ngraph::Node> explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape;
        const auto parameters = setting_up_input_dynamism(mode, input_A, input_B, explicit_A_input, explicit_B_input, normalized_A_shape, normalized_B_shape);
        const auto node = std::make_shared<ngraph::opset3::MatMul>(explicit_A_input, explicit_B_input, matmul_setup.A.transpose, matmul_setup.B.transpose);

        // A
        if (matmul_setup.A.rank_diff) {
            ngraph::OutputVector extended_shape_parts = {
                    ngraph::opset3::Constant::create(
                            ngraph::element::i64, {matmul_setup.A.rank_diff}, std::vector<int64_t>(matmul_setup.A.rank_diff, 1)), normalized_A_shape};
            normalized_A_shape = std::make_shared<ngraph::opset3::Concat>(extended_shape_parts, 0);
        }
        if (!matmul_setup.A.gather_idxs_for_transpose.empty()) {
            const auto indices = ngraph::opset3::Constant::create(
                    ngraph::element::i64, {matmul_setup.A.gather_idxs_for_transpose.size()}, matmul_setup.A.gather_idxs_for_transpose);
            const auto axis = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            normalized_A_shape = std::make_shared<ngraph::opset3::Gather>(normalized_A_shape, indices, axis);
        }
        // B
        if (matmul_setup.B.rank_diff) {
            ngraph::OutputVector extended_shape_parts = {
                    ngraph::opset3::Constant::create(
                            ngraph::element::i64, {matmul_setup.B.rank_diff}, std::vector<int64_t>(matmul_setup.B.rank_diff, 1)), normalized_B_shape};
            normalized_B_shape = std::make_shared<ngraph::opset3::Concat>(extended_shape_parts, 0);
        }
        if (!matmul_setup.B.gather_idxs_for_transpose.empty()) {
            const auto indices = ngraph::opset3::Constant::create(
                    ngraph::element::i64, {matmul_setup.B.gather_idxs_for_transpose.size()}, matmul_setup.B.gather_idxs_for_transpose);
            const auto axis = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            normalized_B_shape = std::make_shared<ngraph::opset3::Gather>(normalized_B_shape, indices, axis);
        }
        // Common
        ngraph::OutputVector output_dims;
        if (!matmul_setup.A.batch_gather_idxs.empty()) {
            const auto max_shape = std::make_shared<ngraph::opset3::Maximum>(normalized_A_shape, normalized_B_shape);
            const auto indices = ngraph::opset3::Constant::create(
                    ngraph::element::i64, {matmul_setup.A.batch_gather_idxs.size()}, matmul_setup.A.batch_gather_idxs);
            const auto axis = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto batch_dims = std::make_shared<ngraph::opset3::Gather>(max_shape, indices, axis);
            output_dims.push_back(batch_dims);
        }
        const auto input_channels = std::make_shared<ngraph::opset3::Gather>(
                normalized_A_shape,
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {matmul_setup.A.channel_idx}),
                ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
        const auto output_channels = std::make_shared<ngraph::opset3::Gather>(
                normalized_B_shape,
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {matmul_setup.B.channel_idx}),
                ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
        output_dims.push_back(input_channels);
        output_dims.push_back(output_channels);

        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        const auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{dsr_final}, parameters, "Transformed-MatMul");
        return function;
    }
};


TEST_P(DynamicToStaticShapeMatMul, CompareFunctions) {
}
INSTANTIATE_TEST_SUITE_P(smoke_MatMul, DynamicToStaticShapeMatMul, combinations);

}  // namespace