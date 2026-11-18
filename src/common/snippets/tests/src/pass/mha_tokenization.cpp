// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mha_tokenization.hpp"

#include <gtest/gtest.h>

#include <pass/mha_tokenization.hpp>
#include <subgraph_mha.hpp>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMHASnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::ExtractReshapesFromMHA>();
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMHASnippets>(mha_config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(common_config);
    disable_rt_info_check();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_2D_Static) {
    const auto& f = MHA2DFunction(
        std::vector<PartialShape>{{12, 64}, {64, 12}, {12, 12}, {12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_2D_Static_Const_B) {
    const auto& f = MHA2DFunction(
        std::vector<PartialShape>{{12, 64}, {64, 12}, {12, 12}, {12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_2D_Dynamic) {
    const auto& f = MHA2DFunction(
        std::vector<PartialShape>{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_2D_Partially_Dynamic) {
    const auto& f = MHA2DFunction(
        std::vector<PartialShape>{{12, 64}, {64, 12}, {-1, 12}, {12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_2D_Partially_Dynamic_Const_B) {
    const auto& f = MHA2DFunction(
        std::vector<PartialShape>{{12, 64}, {64, 12}, {-1, 12}, {12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_ShapeOf_Broadcast) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, ShapeOfExternalConsumer) {
    const auto f = MHAFunction(
        std::vector<PartialShape>{{-1, 128, 12, 64}, {-1, 128, 12, 64}, {-1, 1, 128, 128}, {-1, 128, 12, 64}},
        std::vector<ov::element::Type>(4, ov::element::f32),
        false,
        false,
        true,
        true);
    model = f.getOriginal();
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v3::ShapeOf>(node)) {
            model->add_results({std::make_shared<ov::op::v0::Result>(node)});
            break;
        }
    }
    ASSERT_EQ(model->get_results().size(), 2);
    run();
}

class RuntimeBroadcastTarget : public TokenizeMHASnippetsTests, public testing::WithParamInterface<bool> {};

TEST_P(RuntimeBroadcastTarget, PreserveBatchExpansion) {
    const auto make_parameters = []() {
        return ParameterVector{
            std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4)),
            std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4)),
            std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1}),
            std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4)),
            std::make_shared<op::v0::Parameter>(GetParam() ? element::f32 : element::i64,
                                                GetParam() ? PartialShape::dynamic(4) : PartialShape{4})};
    };
    const auto make_broadcast = [](const ParameterVector& params) {
        const Output<Node> target =
            GetParam() ? Output<Node>(std::make_shared<op::v3::ShapeOf>(params[4])) : params[4]->output(0);
        return std::make_shared<op::v3::Broadcast>(params[2], target, op::BroadcastType::NUMPY);
    };
    const auto make_mha = [](const OutputVector& inputs) {
        const auto matmul0 = std::make_shared<op::v0::MatMul>(inputs[0], inputs[1]);
        const auto add = std::make_shared<op::v1::Add>(matmul0, inputs[2]);
        const auto softmax = std::make_shared<op::v1::Softmax>(add, 3);
        return std::make_shared<op::v0::MatMul>(softmax, inputs[3]);
    };

    const auto params = make_parameters();
    const auto broadcast = make_broadcast(params);
    model = std::make_shared<Model>(OutputVector{make_mha({params[0], params[1], broadcast, params[3]})}, params);

    const auto ref_params = make_parameters();
    const auto ref_broadcast = make_broadcast(ref_params);
    const OutputVector inputs{ref_params[0], ref_params[1], ref_broadcast, ref_params[3]};
    ParameterVector body_params;
    OutputVector body_inputs;
    for (const auto& input : inputs) {
        body_params.push_back(std::make_shared<op::v0::Parameter>(input.get_element_type(), input.get_partial_shape()));
        body_inputs.push_back(body_params.back());
    }
    const auto result = std::make_shared<ov::snippets::op::Result>(make_mha(body_inputs));
    const auto body = std::make_shared<Model>(ResultVector{result}, body_params);
    const auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(inputs, body);
    model_ref = std::make_shared<Model>(OutputVector{subgraph}, ref_params);

    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMHASnippets>(mha_config);
    disable_rt_info_check();
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, RuntimeBroadcastTarget, testing::Values(false, true));

TEST_F(TokenizeMHASnippetsTests, ShapeOfReshapeExternalConsumer) {
    const auto f = MHAFunction(
        std::vector<PartialShape>{{-1, 128, 12, 64}, {-1, 128, 12, 64}, {-1, 12, 128, 128}, {-1, 128, 12, 64}},
        std::vector<ov::element::Type>(4, ov::element::f32),
        false,
        true,
        false,
        true);
    model = f.getOriginal();
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v3::ShapeOf>(node)) {
            model->add_results({std::make_shared<ov::op::v0::Result>(node)});
            break;
        }
    }
    ASSERT_EQ(model->get_results().size(), 2);
    run();
}

TEST_F(TokenizeMHASnippetsTests, ShapeOfReshapeIndirectConsumer) {
    const auto f =
        MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                    std::vector<ov::element::Type>(4, ov::element::f32),
                    false,
                    true,
                    false,
                    true);
    model = f.getOriginal();
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<op::v3::ShapeOf>(node)) {
            // Build Concat(-1, Gather(ShapeOf(add))) as the first Reshape target.
            const auto restore = *node->get_output_target_inputs(0).begin();
            const auto softmax = restore.get_node()->get_input_node_shared_ptr(0);
            const auto reshape0 = softmax->get_input_node_shared_ptr(0);
            const auto index = op::v0::Constant::create(element::i64, Shape{1}, {3});
            const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
            const auto last_dim = std::make_shared<op::v1::Gather>(node, index, axis);
            const auto minus_one = op::v0::Constant::create(element::i64, Shape{1}, {-1});
            reshape0->set_argument(1, std::make_shared<op::v0::Concat>(OutputVector{minus_one, last_dim}, 0));
            restore.replace_source_output(op::v0::Constant::create(element::i64, Shape{4}, {1, 12, 128, 128}));
            break;
        }
    }
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_V3_Broadcast) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_ShapeOf_Reshape) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        true,
        false,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Const_B) {
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        true,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_MatMul1_Const_B_Are_Wei_Blocked) {
    // This pattern maps to the LIR shape flow from lir_179_MarkLoops_data_flow_out.xml:
    // Brgemm(Q, K) -> Softmax -> Brgemm(Softmax, V_const), where only MatMul1 B is constant.
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{1, 300, 8, 32}, {1, 300, 8, 32}, {1, 8, 300, 300}, {1, 300, 8, 32}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        false,
        false,
        false,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic_Const_B) {
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic_Disabled_Dynamic) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    model = f.getOriginal();
    common_config = ov::snippets::pass::CommonOptimizations::Config(1);
    mha_config = ov::snippets::pass::TokenizeMHASnippets::Config(
        ov::snippets::pass::TokenizationConfig(std::numeric_limits<size_t>::max()),
        true,
        false,
        {3, 4});
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic_Disabled_Dynamic_Const_B) {
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false,
        true,
        true);
    model = f.getOriginal();
    common_config = ov::snippets::pass::CommonOptimizations::Config(1);
    mha_config = ov::snippets::pass::TokenizeMHASnippets::Config(
        ov::snippets::pass::TokenizationConfig(std::numeric_limits<size_t>::max()),
        true,
        false,
        {3, 4});
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic_M) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, -1, 12, 64}, {1, 128, 12, 64}, {1, 12, -1, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic_M_Const_B) {
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{1, -1, 12, 64}, {1, 128, 12, 64}, {1, 12, -1, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic_K) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{1, 128, 12, -1}, {1, 128, 12, -1}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_3D) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{128, 12, 64}, {128, 12, 64}, {12, 128, 128}, {128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_3D_Const_B) {
    const auto& f = MHAConstBFunction(
        std::vector<PartialShape>{{128, 12, 64}, {128, 12, 64}, {12, 128, 128}, {128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        true,
        true,
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_3D_Dynamic) {
    const auto& f = MHAFunction(
        std::vector<PartialShape>{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true,
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose) {
    const auto& f = MHAMatMul0TransposeFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose_Const_B) {
    const auto& f = MHAMatMul0TransposeFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose_Dynamic) {
    const auto& f = MHAMatMul0TransposeFunction(
        std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
        std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
        false);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_int_Matmuls) {
    const auto& f = MHAINT8MatMulTypeRelaxedFunction(
        std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}});
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 128, 12, 64}},
                                   true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_extraction) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                   true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction_and_unsupported_existing_transpose) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 12, 64, 128}, {1, 128, 12, 64}},
                                   true,
                                   std::vector<int64_t>{0, 3, 1, 2});
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_extraction_and_unsupported_existing_transpose) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                   true,
                                   std::vector<int64_t>{0, 3, 1, 2});
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_fusion) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 64, 128, 12}, {1, 128, 12, 64}},
                                   false,
                                   std::vector<int64_t>{0, 2, 1, 3});
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_fusion) {
    const auto& f =
        MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                   false,
                                   std::vector<int64_t>{0, 2, 1, 3});
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Reshape_extraction) {
    const auto& f = MHAWithExtractedReshapeFunction(std::vector<PartialShape>{{400, 196, 80},
                                                                              {400, 80, 196},
                                                                              {400, 14, 14, 14},
                                                                              {400, 14, 14, 1, 14},
                                                                              {400, 196, 80}},
                                                    true);
    execute_and_validate_function(*this, f);
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Rank_Change_Reshape_extraction) {
    const auto& f = MHARankUpgradeToReductionFunction(std::vector<PartialShape>{{100, 16, 100, 32},
                                                                                {100, 16, 32, 100},
                                                                                {1, 16, 100, 100},
                                                                                {1, 100, 1, 100, 100},
                                                                                {100, 16, 100, 32}});
    execute_and_validate_function(*this, f);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
