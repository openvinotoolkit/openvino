// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/aarch64/pass/lowered/gemm_cpu_blocking.hpp"

#include "lir_test_utils.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::test::snippets {

using namespace ov::intel_cpu;
using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using namespace ov::snippets;
using PortType = LoopPort::Type;
using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;

namespace {


void create_gemm_loop_infos(const LinearIRPtr& linear_ir,
                             const ExpressionPtr& gemm_expr,
                             size_t m = 0, size_t m_blk = 0,
                             size_t k = 0, size_t k_blk = 0,
                             size_t n = 0, size_t n_blk = 0) {
    const bool n_block = n != 0 && n_blk != 0;
    const bool m_block = m != 0 && m_blk != 0;
    if (n_block) {
        linear_ir->get_loop_manager()->add_loop_info(
            std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(n, n_blk,
                std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>(gemm_expr->get_input_port(0)),
                                      LoopPort::create<PortType::Incremented>(gemm_expr->get_input_port(1))},
                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(gemm_expr->get_output_port(0))},
                BrgemmBlockingBase::get_default_blocking_loop_handlers(n, n_blk)));
    }
    if (m_block) {
        std::vector<LoopPort> entries{LoopPort::create<PortType::Incremented>(gemm_expr->get_input_port(0), 1)};
        for (size_t i = 1; i < gemm_expr->get_input_count(); ++i)
            entries.push_back(LoopPort::create<PortType::NotProcessed>(gemm_expr->get_input_port(i)));
        linear_ir->get_loop_manager()->add_loop_info(
            std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(m, m_blk,
                entries,
                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(gemm_expr->get_output_port(0), 1)},
                BrgemmBlockingBase::get_default_blocking_loop_handlers(m, m_blk)));
    }
}

} // namespace

class GemmBlockingTest : public LoweredPassTestsF {
public:
    GemmBlockingTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

protected:
    size_t m_blk = 32;
    size_t k_blk = full_dim;
    size_t n_blk = 64;

    static const size_t full_dim = ov::snippets::utils::get_full_dim_value();
};

class GemmCPUBlockingTest : public GemmBlockingTest {
public:
    GemmCPUBlockingTest() = default;

    void SetUp() override {
        pipeline.register_pass<ov::intel_cpu::pass::GemmCPUBlocking>();
    }
};

TEST_F(GemmCPUBlockingTest, Floating) {
    const ov::PartialShape input_shape_a{1, 384, 16, 1024};
    const ov::PartialShape input_shape_b{1, 384, 16, 1024};
    const auto precision = ov::element::f32;
    const VectorDims layout_a{0, 2, 1, 3};
    const VectorDims layout_b{0, 2, 3, 1};
    const VectorDims layout_c{0, 2, 1, 3};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir->push_node<aarch64::GemmCPU>(data_a.second,
                                                           data_b.second,
                                                           PortDescriptor{},
                                                           PortDescriptor{},
                                                           PortDescriptor{0, 0},
                                                           layout_a,
                                                           layout_b,
                                                           layout_c);
        init_expr_descriptors(*gemm.first, {}, {layout_a, layout_b, layout_c});
        auto result = linear_ir->push_node<ov::opset10::Result>(gemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir_ref->push_node<aarch64::GemmCPU>(data_a.second,
                                                               data_b.second,
                                                               PortDescriptor{},
                                                               PortDescriptor{},
                                                               PortDescriptor{0, 0},
                                                               layout_a,
                                                               layout_b,
                                                               layout_c);
        const auto& gemm_expr = *gemm.first;
        init_expr_descriptors(gemm_expr, {{m_blk, full_dim}, {full_dim, n_blk}, {m_blk, n_blk}}, {layout_a, layout_b, layout_c});
        create_gemm_loop_infos(linear_ir_ref, gemm_expr, 384, m_blk, 0, 0, 384, n_blk);
        gemm_expr->set_loop_ids({1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(gemm.second);
    }
}

TEST_F(GemmCPUBlockingTest, Floating_LargeK) {
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type n = 384;
    const ov::Dimension::value_type k = 2048;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::f32;
    k_blk = full_dim;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                           PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        init_expr_descriptors(*gemm.first, {});
        auto result = linear_ir->push_node<ov::opset10::Result>(gemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir_ref->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                               PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        const auto& gemm_expr = *gemm.first;
        init_expr_descriptors(gemm_expr, {{m_blk, full_dim}, {full_dim, n_blk}, {m_blk, n_blk}});
        create_gemm_loop_infos(linear_ir_ref, gemm_expr, m, m_blk, 0, 0, n, n_blk);
        gemm_expr->set_loop_ids({1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(gemm.second);
    }
}

TEST_F(GemmCPUBlockingTest, Float_FC) {
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type k = 1024;
    const ov::Dimension::value_type n = 384;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::f32;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);

        auto gemm = linear_ir->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                           PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        init_expr_descriptors(*gemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(gemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);

        auto gemm = linear_ir_ref->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                               PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        const auto& gemm_expr = *gemm.first;
        init_expr_descriptors(gemm_expr, {{m_blk, full_dim}, {full_dim, n_blk}, {m_blk, n_blk}});
        create_gemm_loop_infos(linear_ir_ref, gemm_expr, m, m_blk, 0, 0, n, n_blk);
        gemm_expr->set_loop_ids({1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(gemm.second);
    }
}

TEST_F(GemmCPUBlockingTest, BlockingIsNotNeeded) {
    const auto precision = ov::element::f32;
    const ov::Dimension::value_type n = 64;
    const ov::Dimension::value_type m = 32;
    const ov::Dimension::value_type k = 16;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                           PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        init_expr_descriptors(*gemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(gemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto gemm = linear_ir_ref->push_node<aarch64::GemmCPU>(data_a.second, data_b.second,
                                                               PortDescriptor{}, PortDescriptor{}, PortDescriptor{});
        const auto full_subtensor = VectorDims(2, ov::snippets::utils::get_full_dim_value());
        init_expr_descriptors(*gemm.first, std::vector<VectorDims>(3, full_subtensor));
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(gemm.second);
    }
}

}  // namespace ov::test::snippets
