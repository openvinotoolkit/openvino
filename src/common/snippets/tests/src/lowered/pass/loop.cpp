// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset10.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/cleanup_loop_offsets.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/optimize_loop_single_evaluation.hpp"
#include "snippets/lowered/pass/validate_unified_loops.hpp"
#include "snippets/lowered/pass/validate_expanded_loops.hpp"
#include "snippets/lowered/pass/normalize_loop_ids.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

using Snippets_TailProcessingTransformation = ::testing::Test;
// [Inserted Loop number, [ptr_increments, final_offsets]
using ref_map = std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>>;
using namespace ov::snippets::lowered;

constexpr static size_t vector_size = 16;

static void init_linear_ir(const std::vector<ov::Shape>& in_shapes, LinearIR& linear_ir, size_t block_size) {
    Config lir_config;
    lir_config.m_manual_build_support = true;
    linear_ir = LinearIR(lir_config, std::make_shared<ov::snippets::IShapeInferSnippetsFactory>());

    const ov::element::Type input_precision = ov::element::f32;
    const auto param0 = linear_ir.push_node<ov::opset10::Parameter>(input_precision, in_shapes[0]);
    const auto param1 = linear_ir.push_node<ov::opset10::Parameter>(input_precision, in_shapes[1]);
    const auto param2 = linear_ir.push_node<ov::opset10::Parameter>(input_precision, in_shapes[2]);
    const auto matmul = linear_ir.push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
    const auto add = linear_ir.push_node<ov::opset10::Add>(matmul.second, param2.second);
    const auto result = linear_ir.push_node<ov::opset10::Result>(add.second);

    const auto loop_manager = linear_ir.get_loop_manager();
    linear_ir.get_loop_manager()->mark_loop(matmul.first, add.first, in_shapes[0].front(), block_size, 1,
                                            std::vector<LoopPort>{LoopPort((*matmul.first)->get_input_port(0)),
                                                                  LoopPort((*matmul.first)->get_input_port(1), false)},
                                            std::vector<LoopPort>{LoopPort((*matmul.first)->get_output_port(0))});
    linear_ir.get_loop_manager()->mark_loop(add.first, result.first, in_shapes[2].back(), vector_size, 0,
                                            std::vector<LoopPort>{LoopPort((*add.first)->get_input_port(0)),
                                                                  LoopPort((*add.first)->get_input_port(1))},
                                            std::vector<LoopPort>{LoopPort((*add.first)->get_output_port(0))});
    linear_ir.get_loop_manager()->mark_loop(add.first, result.first, in_shapes[2].front(), 1, 1,
                                            std::vector<LoopPort>{LoopPort((*add.first)->get_input_port(0)),
                                                                  LoopPort((*add.first)->get_input_port(1))},
                                            std::vector<LoopPort>{LoopPort((*add.first)->get_output_port(0))});
}

static void apply_transformations(LinearIR& linear_ir, const std::shared_ptr<ov::snippets::lowered::pass::PassConfig>& config) {
    const auto is_loop_decomp_disabled = config->is_disabled<ov::snippets::lowered::pass::InsertSpecificIterations>();
    if (is_loop_decomp_disabled) {
        config->disable<ov::snippets::lowered::pass::ValidateExpandedLoops>();
    }

    ov::snippets::lowered::pass::PassPipeline pipeline(config);
    pipeline.register_pass<ov::snippets::lowered::pass::SplitLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertBuffers>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoadStore>(vector_size);
    pipeline.register_pass<ov::snippets::lowered::pass::ValidateUnifiedLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InitLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertSpecificIterations>();
    pipeline.register_pass<ov::snippets::lowered::pass::NormalizeLoopIDs>(!is_loop_decomp_disabled);
    pipeline.register_pass<ov::snippets::lowered::pass::ValidateExpandedLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::CleanupLoopOffsets>();
    pipeline.register_pass<ov::snippets::lowered::pass::OptimizeLoopSingleEvaluation>();
    pipeline.run(linear_ir);
}

static void validate(const LinearIR& linear_ir, const ref_map& reference) {
    std::set<size_t> loops;
    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(node);
        if (!loop_end)
            continue;
        const auto loop_num = loop_end->get_id();
        ASSERT_GT(reference.count(loop_num), 0);
        loops.insert(loop_num);
        ASSERT_TRUE(loop_end->get_ptr_increments() == reference.at(loop_num).first);
        ASSERT_TRUE(loop_end->get_finalization_offsets() == reference.at(loop_num).second);
    }
    ASSERT_EQ(loops.size(), reference.size());
}

TEST(Snippets_TailProcessingTransformation, BlockedWOTail_OriginalPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {12, 16};
    ov::Shape inputShape1 = {16, 20};
    ov::Shape inputShape2 = {12, 20};
    init_linear_ir({inputShape0, inputShape1, inputShape2}, linear_ir, 4);

    auto config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    config->disable<ov::snippets::lowered::pass::CleanupLoopOffsets>();
    config->disable<ov::snippets::lowered::pass::InsertSpecificIterations>();
    config->disable<ov::snippets::lowered::pass::OptimizeLoopSingleEvaluation>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 1), std::vector<int64_t>(3, -20)};
    reference[1] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -80)};
    reference[2] = { {16, 0, 20, 20}, {-192, 0, -240, -240}};

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedWOTail_CleanUpPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {12, 16};
    ov::Shape inputShape1 = {16, 20};
    ov::Shape inputShape2 = {12, 20};
    init_linear_ir({inputShape0, inputShape1, inputShape2}, linear_ir, 4);

    auto config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    config->disable<ov::snippets::lowered::pass::InsertSpecificIterations>();
    config->disable<ov::snippets::lowered::pass::OptimizeLoopSingleEvaluation>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 1), std::vector<int64_t>(3, 0)};
    reference[1] = { std::vector<int64_t>(3, 0), {0, -80, 0}}; // -80 - finalization offset for Buffer ptr
    reference[2] = { {16, 0, 0, 0}, std::vector<int64_t>(4, 0)};

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedTail_OriginalPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {14, 16};
    ov::Shape inputShape1 = {16, 20};
    ov::Shape inputShape2 = {14, 20};
    init_linear_ir({inputShape0, inputShape1, inputShape2}, linear_ir, 4);

    auto config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    config->disable<ov::snippets::lowered::pass::CleanupLoopOffsets>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)}; // Vector Inner
    reference[1] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, -16)}; // Tail Inner

    reference[2] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -80)}; // Inner Vector Blocked
    reference[3] = { {16, 0, 20, 20}, std::vector<int64_t>(4, 0)}; // Outer Vector Blocked

    reference[4] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -40)}; // Inner Tail Blocked
    reference[5] = { std::vector<int64_t>(4, 0), {-192, 0, -240, -240}}; // Outer Tail Blocked

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedTail_CleanUpPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {14, 16};
    ov::Shape inputShape1 = {16, 20};
    ov::Shape inputShape2 = {14, 20};
    init_linear_ir({inputShape0, inputShape1, inputShape2}, linear_ir, 4);

    apply_transformations(linear_ir, std::make_shared<ov::snippets::lowered::pass::PassConfig>());

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)}; // Vector Inner
    reference[1] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 4)}; // Tail Inner

    reference[2] = { std::vector<int64_t>(3, 0), {0, -80, 0}}; // Inner Vector Blocked (-80 - finalization offset for Buffer ptr)
    reference[3] = { {16, 0, 0, 0}, std::vector<int64_t>(4, 0)}; // Outer Vector Blocked

    reference[4] = { std::vector<int64_t>(3, 0), {0, -40, 0}}; // Inner Tail Blocked (-40 - finalization offset for Buffer ptr)
    reference[5] = { std::vector<int64_t>(4, 0), {32, 0, 0, 0}}; // Outer Tail Blocked

    validate(linear_ir, reference);
}
