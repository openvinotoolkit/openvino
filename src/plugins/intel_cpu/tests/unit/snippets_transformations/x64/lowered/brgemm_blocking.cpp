// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"
#ifdef SNIPPETS_LIBXSMM_TPP
    #include "transformations/tpp/common/pass/lowered/brgemm_tpp_blocking.hpp"
#endif

#include "lir_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace test {
namespace snippets {
using namespace ov::intel_cpu;
using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using namespace ov::snippets;
using BRGEMM_TYPE = intel_cpu::brgemm_utils::BRGEMM_TYPE;
using PortType = LoopPort::Type;

namespace {
enum class BACKEND_TYPE{CPU, TPP};
SpecificIterationHandlers get_k_loop_handlers(size_t work_amount, size_t block_size, BACKEND_TYPE backend = BACKEND_TYPE::CPU) {
    auto handlers = BrgemmBlockingBase::get_default_blocking_loop_handlers(work_amount, block_size);
    switch (backend) {
#ifdef SNIPPETS_LIBXSMM_TPP
        case BACKEND_TYPE::TPP:
            handlers.register_pass<SpecificLoopIterType::FIRST_ITER, ov::intel_cpu::tpp::pass::BrgemmTPPBlocking::SetBrgemmBeta>();
            break;
#endif
        case BACKEND_TYPE::CPU:
            handlers.register_pass<SpecificLoopIterType::FIRST_ITER, ov::intel_cpu::pass::BrgemmCPUBlocking::DummyPass>();
            break;
        default:
            OPENVINO_THROW("Unsupported code generator backend type");
    }
    return handlers;
}

void create_brgemm_loop_infos(const LinearIRPtr& linear_ir,
                              const ExpressionPtr& brgemm_expr,
                              size_t m = 0, size_t m_blk = 0,
                              size_t k = 0, size_t k_blk = 0,
                              size_t n = 0, size_t n_blk = 0,
                              BACKEND_TYPE backend = BACKEND_TYPE::CPU) {
    const bool k_block = k != 0 && k_blk != 0;
    const bool n_block = k != 0 && k_blk != 0;
    const bool m_block = m != 0 && m_blk != 0;
    if (k_block) {
        const auto loop_info =
            std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(k, k_blk,
                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(0), 0),
                                      LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(1), 1)},
                std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_output_port(0))},
                get_k_loop_handlers(k, k_block, backend));
        linear_ir->get_loop_manager()->add_loop_info(loop_info);
    }
    if (n_block) {
        linear_ir->get_loop_manager()->add_loop_info(
            std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(n, n_blk,
                std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_input_port(0)),
                                      LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(1))},
                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(brgemm_expr->get_output_port(0))},
                BrgemmBlockingBase::get_default_blocking_loop_handlers(n, n_block)));
    }
    if (m_block) {
        std::vector<LoopPort> entries{LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(0), 1)};
        for (size_t i = 1; i < brgemm_expr->get_input_count(); ++i)
            entries.push_back(LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_input_port(i)));
        linear_ir->get_loop_manager()->add_loop_info(
            std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(m, m_blk,
                entries,
                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(brgemm_expr->get_output_port(0), 1)},
                BrgemmBlockingBase::get_default_blocking_loop_handlers(m, m_block)));
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

protected:
    size_t m_blk = 32;
    size_t k_blk = 512;
    size_t n_blk = 64;

    static const size_t full_dim = ov::snippets::utils::get_full_dim_value();
};
class BrgemmCPUBlockingTest : public BrgemmBlockingTest {
public:
    BrgemmCPUBlockingTest() : BrgemmBlockingTest() {
        n_blk = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
    }

    void SetUp() override {
        pipeline.register_pass<ov::intel_cpu::pass::BrgemmCPUBlocking>();
    }
};

TEST_F(BrgemmCPUBlockingTest, Floating) {
    const ov::PartialShape input_shape_a{1, 384, 16, 1024};
    const ov::PartialShape input_shape_b{1, 384, 16, 1024};
    const auto precision = ov::element::f32;
    const VectorDims layout_a{0, 2, 1, 3};
    const VectorDims layout_b{0, 2, 3, 1};
    const VectorDims layout_c{0, 2, 1, 3};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE,
                                                      0, 0, 0, layout_a, layout_b, layout_c);
        init_expr_descriptors(*brgemm.first, {}, {layout_a, layout_b, layout_c});
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE,
                                                          0, 0, 0, layout_a, layout_b, layout_c);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}}, {layout_a, layout_b, layout_c});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, 384, m_blk, 1024, k_blk, 384, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, Floating_LargeK) {
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type n = 384;
    const ov::Dimension::value_type k = 2048;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::f32;
    k_blk = 1024;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE);
        init_expr_descriptors(*brgemm.first, {});
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, m, m_blk, k, k_blk, n, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, BlockingIsNotNeeded) {
    const ov::Dimension::value_type m = 32;
    const ov::Dimension::value_type k = 16;
    const ov::Dimension::value_type n = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::f32;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, data_b.second, BRGEMM_TYPE::STAND_ALONE);
        const auto full_subtensor = VectorDims(2, ov::snippets::utils::get_full_dim_value());
        init_expr_descriptors(*brgemm.first, std::vector<VectorDims>(3, full_subtensor));
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, WithTransposeB) {
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type k = 1024;
    const ov::Dimension::value_type n = 384;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, n, k};
    const auto precision_a = ov::element::f32;
    const auto precision_b = ov::element::f32;
    const std::vector<size_t> layout_input{0, 1, 3, 2};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision_a, BRGEMM_TYPE::REPACKING_ONLY, 0, 0, 0, layout_input);
        init_expr_descriptors(*copy_b.first);

        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, BRGEMM_TYPE::REPACKING_ONLY);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision_a, BRGEMM_TYPE::REPACKING_ONLY, 0, 0, 0, layout_input);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{full_dim, full_dim}, {full_dim, full_dim}});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, BRGEMM_TYPE::REPACKING_ONLY);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, m, m_blk, k, k_blk, n, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, WithDataRepacking) {
    // Skipped because K,N blocking is disabled until heuristic is updated (ticket: 156014)
    GTEST_SKIP();
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type k = 1024;
    const ov::Dimension::value_type n = 384;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision_a = ov::element::u8;
    const auto precision_b = ov::element::i8;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision_a, BRGEMM_TYPE::REPACKING_ONLY);
        init_expr_descriptors(*copy_b.first);

        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, BRGEMM_TYPE::REPACKING_ONLY);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_a, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision_b, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision_a, BRGEMM_TYPE::REPACKING_ONLY);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{full_dim, full_dim}, {full_dim, full_dim}});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, BRGEMM_TYPE::REPACKING_ONLY);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, m, m_blk, k, k_blk, n, n_blk);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, WithCompensations) {
    // Skipped because K,N blocking is disabled until heuristic is updated (ticket: 156014)
    GTEST_SKIP();
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type k = 1024;
    const ov::Dimension::value_type n = 384;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::i8;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision, BRGEMM_TYPE::WITH_COMPENSATIONS);
        init_expr_descriptors(*copy_b.first);
        const auto& copy_b_n = copy_b.second;
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b_n->output(0), copy_b_n->output(1), BRGEMM_TYPE::WITH_COMPENSATIONS);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision, BRGEMM_TYPE::WITH_COMPENSATIONS);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{full_dim, full_dim}, {full_dim, full_dim}, {1, full_dim}});

        const auto& copy_b_n = copy_b.second;
        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b_n->output(0), copy_b_n->output(1), BRGEMM_TYPE::WITH_COMPENSATIONS);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {1, full_dim},  {m_blk, n_blk}});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, m, m_blk, k, k_blk, n, n_blk);

        const auto loop_manager = linear_ir_ref->get_loop_manager();
        auto update_loop_info = [&loop_manager](size_t loop_id, LoopPort&& new_port) {
            const auto& loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
            const auto& in_ports = loop_info->get_input_ports();
            loop_info->replace_with_new_ports(in_ports[1], {in_ports[1], new_port});
        };
        const auto& compens_port = brgemm_expr->get_input_port(2);
        update_loop_info(1, LoopPort::create<PortType::Incremented>(compens_port, 0));
        update_loop_info(0, LoopPort::create<PortType::NotIncremented>(compens_port, 1));

        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}

TEST_F(BrgemmCPUBlockingTest, AMX) {
    // Skipped because K,N blocking is disabled until heuristic is updated (ticket: 156014)
    GTEST_SKIP();
    const ov::Dimension::value_type m = 384;
    const ov::Dimension::value_type k = 1024;
    const ov::Dimension::value_type n = 384;
    const ov::PartialShape input_shape_a{1, 16, m, k};
    const ov::PartialShape input_shape_b{1, 16, k, n};
    const auto precision = ov::element::bf16;

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto scratch = linear_ir->push_node<snippets::op::Buffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
        auto copy_b = linear_ir->push_node<BrgemmCopyB>(data_b.second, precision, BRGEMM_TYPE::REPACKING_ONLY);
        init_expr_descriptors(*copy_b.first);
        auto brgemm = linear_ir->push_node<BrgemmCPU>(data_a.second, copy_b.second, scratch.second, BRGEMM_TYPE::WITH_AMX);
        init_expr_descriptors(*brgemm.first);
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto copy_b = linear_ir_ref->push_node<BrgemmCopyB>(data_b.second, precision, BRGEMM_TYPE::REPACKING_ONLY);
        const auto copy_b_expr = *copy_b.first;
        init_expr_descriptors(copy_b_expr, {{full_dim, full_dim},
                                            {full_dim, full_dim}});

        auto scratch = linear_ir_ref->push_node<snippets::op::Buffer>(ov::Shape {BrgemmCPU::SCRATCH_BYTE_SIZE});
        scratch.first->get()->set_loop_ids({2, 1, 0});

        auto brgemm = linear_ir_ref->push_node<BrgemmCPU>(data_a.second, copy_b.second, scratch.second,
                                                          BRGEMM_TYPE::WITH_AMX);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, get_default_subtensor(), {m_blk, n_blk}});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, m, 0, k, k_blk, n, n_blk);

        std::vector<LoopPort> entries {LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(0), 1),
                                       LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_input_port(1))};
        std::vector<LoopPort> exits {LoopPort::create<PortType::Incremented>(brgemm_expr->get_output_port(0), 1)};
        auto handlers = BrgemmBlockingBase::get_default_blocking_loop_handlers(m, m_blk);
        linear_ir_ref->get_loop_manager()->
            add_loop_info(std::make_shared<ov::snippets::lowered::UnifiedLoopInfo>(m, m_blk, entries, exits, handlers));
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}


#ifdef SNIPPETS_LIBXSMM_TPP
class BrgemmTPPBlockingTest : public BrgemmBlockingTest {
public:
    BrgemmTPPBlockingTest() : BrgemmBlockingTest() {}

    void SetUp() override {
        pipeline.register_pass<ov::intel_cpu::tpp::pass::BrgemmTPPBlocking>();
    }
};

TEST_F(BrgemmTPPBlockingTest, TPPFloating) {
    const ov::PartialShape input_shape_a{1, 384, 16, 1024};
    const ov::PartialShape input_shape_b{1, 384, 16, 1024};
    const auto precision = ov::element::f32;
    const VectorDims layout_a{0, 2, 1, 3};
    const VectorDims layout_b{0, 2, 3, 1};
    const VectorDims layout_c{0, 2, 1, 3};

    {
        auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir->push_node<tpp::op::BrgemmTPP>(data_a.second, data_b.second, 0, 0, 0,
                                                               layout_a, layout_b, layout_c);
        init_expr_descriptors(*brgemm.first, {}, {layout_a, layout_b, layout_c});
        auto result = linear_ir->push_node<ov::opset10::Result>(brgemm.second);
    }
    {
        auto data_a = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_a);
        auto data_b = linear_ir_ref->push_node<ov::opset10::Parameter>(precision, input_shape_b);
        auto brgemm = linear_ir_ref->push_node<tpp::op::BrgemmTPP>(data_a.second, data_b.second, 0, 0, 0,
                                                                   layout_a, layout_b, layout_c);
        const auto& brgemm_expr = *brgemm.first;
        init_expr_descriptors(brgemm_expr, {{m_blk, k_blk}, {k_blk, n_blk}, {m_blk, n_blk}}, {layout_a, layout_b, layout_c});
        create_brgemm_loop_infos(linear_ir_ref, brgemm_expr, 384, m_blk, 1024, k_blk, 384, n_blk, BACKEND_TYPE::TPP);
        brgemm_expr->set_loop_ids({2, 1, 0});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm.second);
    }
}
#endif // SNIPPETS_LIBXSMM_TPP

}  // namespace snippets
}  // namespace test
}  // namespace ov
