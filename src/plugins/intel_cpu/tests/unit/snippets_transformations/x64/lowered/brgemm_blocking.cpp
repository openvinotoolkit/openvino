// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"

#include "lir_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/cpu_iter_handlers.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"

namespace ov {
namespace test {
namespace snippets {
using namespace ov::intel_cpu;
using namespace ov::snippets::lowered;
using namespace ov::snippets;

namespace {
void create_brgemm_loop_infos(const LinearIRPtr& linear_ir,
                              const ExpressionPtr& brgemm_expr,
                              size_t m = 0, size_t m_blk = 0,
                              size_t k = 0, size_t k_blk = 0,
                              size_t n = 0, size_t n_blk = 0) {
    const bool k_block = k != 0 && k_blk != 0;
    const bool n_block = k != 0 && k_blk != 0;
    const bool m_block = m != 0 && m_blk != 0;
    if (k_block) {
        create_and_add_unified_loop_info(linear_ir, k, k_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0)), LoopPort(brgemm_expr->get_input_port(1), true, 1)},
                                        {LoopPort(brgemm_expr->get_output_port(0), false)});
        const auto& loop_info = linear_ir->get_loop_manager()->get_loop_info<UnifiedLoopInfo>(0);
        loop_info->register_pass_to_handler<SpecificLoopIterType::FIRST_ITER, ov::intel_cpu::pass::SetBrgemmBeta>(0.f);
    }
    if (n_block) {
        create_and_add_unified_loop_info(linear_ir, n, n_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0), false), LoopPort(brgemm_expr->get_input_port(1))},
                                        {LoopPort(brgemm_expr->get_output_port(0))});
    }
    if (m_block) {
        create_and_add_unified_loop_info(linear_ir, m, m_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0), true, 1), LoopPort(brgemm_expr->get_input_port(1), false, 1)},
                                        {LoopPort(brgemm_expr->get_output_port(0), true, 1)});
    }
}

void create_brgemm_with_copy_b_loop_infos(const LinearIRPtr& linear_ir,
                                          const ExpressionPtr& brgemm_expr,
                                          const ExpressionPtr& copy_b_expr,
                                          size_t m = 0, size_t m_blk = 0,
                                          size_t k = 0, size_t k_blk = 0,
                                          size_t n = 0, size_t n_blk = 0) {
    const bool k_block = k != 0 && k_blk != 0;
    const bool n_block = k != 0 && k_blk != 0;
    const bool m_block = m != 0 && m_blk != 0;
    if (k_block) {
        create_and_add_unified_loop_info(linear_ir, k, k_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0)), LoopPort(copy_b_expr->get_input_port(0), true, 1)},
                                        {LoopPort(brgemm_expr->get_output_port(0), false)});
        const auto& loop_info = linear_ir->get_loop_manager()->get_loop_info<UnifiedLoopInfo>(0);
        loop_info->register_pass_to_handler<SpecificLoopIterType::FIRST_ITER, ov::intel_cpu::pass::SetBrgemmBeta>(0.f);
    }
    if (n_block) {
        create_and_add_unified_loop_info(linear_ir, n, n_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0), false), LoopPort(copy_b_expr->get_input_port(0))},
                                        {LoopPort(brgemm_expr->get_output_port(0))});
    }
    if (m_block) {
        const auto& second_input_port = k_block || n_block ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1);
        create_and_add_unified_loop_info(linear_ir, m, m_blk,
                                        {LoopPort(brgemm_expr->get_input_port(0), true, 1), LoopPort(second_input_port, false, 1)},
                                        {LoopPort(brgemm_expr->get_output_port(0), true, 1)});
    }
}
} // namespace

class BrgemmBlockingTest : public LoweredPassTestsF {
public:
    BrgemmBlockingTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<ov::intel_cpu::pass::BrgemmBlocking>();
    }
};

TEST_F(BrgemmBlockingTest, Floating) {
    const size_t m_blk = 32;
    const size_t k_blk = 16;
    const size_t n_blk = 64;
    const ov::PartialShape input_shape_a{1, 384, 16, 64};
    const ov::PartialShape input_shape_b{1, 384, 16, 64};
    const auto precision = ov::element::f32;
    const VectorDims layout_a{0, 2, 1, 3};
    const VectorDims layout_b{0, 2, 3, 1};
    const VectorDims layout_c{0, 2, 1, 3};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, data_b.second, BrgemmCPU::Type::Floating,
                                                      0, 0, 0, layout_a, layout_b, layout_c, m_blk, k_blk, n_blk);
        init_expr_descriptors(*brgemm.first, {}, {layout_a, layout_b, layout_c});
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, data_b.second, BrgemmCPU::Type::Floating,
                                                          0, 0, 0, layout_a, layout_b, layout_c, m_blk, k_blk, n_blk);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}}, {layout_a, layout_b, layout_c});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, 384, m_blk, 64, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmBlockingTest, BlockingIsNotNeeded) {
    const size_t m = 32;
    const size_t k = 16;
    const size_t n = 64;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::f32;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, data_b.second, BrgemmCPU::Type::Floating);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, data_b.second, BrgemmCPU::Type::Floating);
        brgemm.second->set_beta(0.f);
        init_expr_descriptors(*brgemm.first, {{m, k}, {k, n}, {m, n}});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmBlockingTest, WithDataRepacking) {
    const size_t m_blk = 32;
    const size_t k_blk = 16;
    const size_t n_blk = 64;
    const ov::PartialShape input_shape_a{1, 16, 384, 64};
    const ov::PartialShape input_shape_b{1, 16, 64, 384};
    const auto precision_a = ov::element::u8;
    const auto precision_b = ov::element::i8;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision_a, BrgemmCopyB::Type::OnlyRepacking,
                                                        0, 0, 0, VectorDims{}, k_blk, n_blk);
        init_expr_descriptors(*copy_b.first);

        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, BrgemmCPU::Type::WithDataRepacking,
                                                      0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision_a, BrgemmCopyB::Type::OnlyRepacking,
                                                            0, 0, 0, VectorDims{}, k_blk, n_blk);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{k_blk, n_blk}, {k_blk, n_blk}});
        copy_b_expr->set_loop_ids({2, 1, 0});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, BrgemmCPU::Type::WithDataRepacking,
                                                          0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}});
        create_brgemm_with_copy_b_loop_infos(linear_ir_ref, brgemm_expr, copy_b_expr, 384, m_blk, 64, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmBlockingTest, WithDataRepackingOnlyByM) {
    const size_t m_blk = 32;
    const size_t k = 64;
    const size_t n = 384;
    const ov::PartialShape input_shape_a{1, 16, 384, 64};
    const ov::PartialShape input_shape_b{1, 16, 64, 384};
    const auto precision_a = ov::element::u8;
    const auto precision_b = ov::element::i8;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision_a, BrgemmCopyB::Type::OnlyRepacking,
                                                        0, 0, 0, VectorDims{}, k, n);
        init_expr_descriptors(*copy_b.first);

        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, BrgemmCPU::Type::WithDataRepacking,
                                                      0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k, n);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision_a, BrgemmCopyB::Type::OnlyRepacking,
                                                            0, 0, 0, VectorDims{}, k, n);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{k, n}, {k, n}});
        copy_b_expr->set_loop_ids({});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, BrgemmCPU::Type::WithDataRepacking,
                                                          0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k, n, 0.f);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k}, {k, n}, {m_blk, n}});
        create_brgemm_with_copy_b_loop_infos(linear_ir_ref, brgemm_expr, copy_b_expr, 384, m_blk);
        brgemm_expr->set_loop_ids({0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmBlockingTest, WithCompensations) {
    const size_t m_blk = 32;
    const size_t k_blk = 16;
    const size_t n_blk = 64;
    const ov::PartialShape input_shape_a{1, 16, 384, 64};
    const ov::PartialShape input_shape_b{1, 16, 64, 384};
    const auto precision = ov::element::i8;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision, BrgemmCopyB::Type::WithCompensations,
                                                        0, 0, 0, VectorDims{}, k_blk, n_blk);
        init_expr_descriptors(*copy_b.first);
        const auto& copy_b_n = copy_b.second;
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b_n->output(0), copy_b_n->output(1), BrgemmCPU::Type::WithCompensations,
                                                      0, 0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision, BrgemmCopyB::Type::WithCompensations,
                                                            0, 0, 0, VectorDims{}, k_blk, n_blk);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{k_blk, n_blk}, {k_blk, n_blk}, {1, n_blk}});
        copy_b_expr->set_loop_ids({2, 1, 0});

        const auto& copy_b_n = copy_b.second;
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b_n->output(0), copy_b_n->output(1), BrgemmCPU::Type::WithCompensations,
                                                          0, 0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {1, n_blk}, {m_blk, n_blk}});
        create_brgemm_with_copy_b_loop_infos(linear_ir_ref, brgemm_expr, copy_b_expr, 384, m_blk, 64, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmBlockingTest, AMX) {
    const size_t m_blk = 32;
    const size_t k_blk = 16;
    const size_t n_blk = 64;
    const ov::PartialShape input_shape_a{1, 16, 384, 64};
    const ov::PartialShape input_shape_b{1, 16, 64, 384};
    const auto precision = ov::element::bf16;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto scratch = linear_ir->push_node<snippets::op::NewMemoryBuffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision, BrgemmCopyB::Type::OnlyRepacking,
                                                        0, 0, 0, VectorDims{}, k_blk, n_blk);
        init_expr_descriptors(*copy_b.first);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, scratch.second, BrgemmCPU::Type::AMX,
                                                      0, 0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision, BrgemmCopyB::Type::OnlyRepacking,
                                                            0, 0, 0, VectorDims{}, k_blk, n_blk);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{k_blk, n_blk}, {k_blk, n_blk}});
        copy_b_expr->set_loop_ids({2, 1, 0});

        auto scratch = linear_ir_ref->push_node<snippets::op::NewMemoryBuffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
        scratch.first->get()->set_loop_ids({2, 1, 0});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, scratch.second, BrgemmCPU::Type::AMX,
                                                          0, 0, 0, 0, VectorDims{}, VectorDims{}, VectorDims{}, m_blk, k_blk, n_blk);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, get_default_subtensor(), {m_blk, n_blk}});
        create_brgemm_with_copy_b_loop_infos(linear_ir_ref, brgemm_expr, copy_b_expr, 384, m_blk, 64, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

#ifdef SNIPPETS_LIBXSMM_TPP
TEST_F(BrgemmBlockingTest, TPPFloating) {
    const size_t m_blk = 32;
    const size_t k_blk = 16;
    const size_t n_blk = 64;
    const ov::PartialShape input_shape_a{1, 384, 16, 64};
    const ov::PartialShape input_shape_b{1, 384, 16, 64};
    const auto precision = ov::element::f32;
    const VectorDims layout_a{0, 2, 1, 3};
    const VectorDims layout_b{0, 2, 3, 1};
    const VectorDims layout_c{0, 2, 1, 3};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<tpp::op::BrgemmTPP>(data_a.second, data_b.second, 0, 0, 0,
                                                               layout_a, layout_b, layout_c, m_blk, k_blk, n_blk);
        init_expr_descriptors(*brgemm.first, {}, {layout_a, layout_b, layout_c});
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<tpp::op::BrgemmTPP>(data_a.second, data_b.second, 0, 0, 0,
                                                                   layout_a, layout_b, layout_c, m_blk, k_blk, n_blk);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}}, {layout_a, layout_b, layout_c});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, 384, m_blk, 64, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}
#endif // SNIPPETS_LIBXSMM_TPP

}  // namespace snippets
}  // namespace test
}  // namespace ov
