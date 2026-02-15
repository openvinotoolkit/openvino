// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"

#include "lir_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using PortType = LoopPort::Type;

class ComputeBufferAllocationSizeTest : public LoweredPassTestsF {
public:
    ComputeBufferAllocationSizeTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::EXPR_ATTRS);
    }

    void SetUp() override {
        pipeline.register_pass<ComputeBufferAllocationSize>();
    }

    size_t vector_size = 16;
    ov::element::Type input_precision = ov::element::f32;
    size_t m_block = 32;
    size_t n_block = 64;
};

/*
 * Control Flow Graph:
 * LoopBegin (blocking_m_loop)
 * |  LoopBegin (brgemm1_n_loop)
 * |  |  Brgemm1
 * |  LoopEnd (brgemm1_n_loop)
 * |
 * |  Buffer1
 * |
 * |  LoopBegin (add_m_split_loop)
 * |  |  LoopBegin (inner_add_loop)
 * |  |    Add
 * |  |  LoopEnd (inner_add_loop)
 * |  LoopEnd (add_m_split_loop)
 * |
 * |  Buffer2
 * |
 * |  LoopBegin (brgemm2_n2_loop)
 * |  |  Brgemm2
 * |  LoopEnd (brgemm2_n2_loop)
 * LoopEnd (blocking_m_loop)
 */
TEST_F(ComputeBufferAllocationSizeTest, BrgemmAddBrgemmSplitM) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const size_t n2 = 256;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::Shape input_shape_2{1, 1, m, n};
    const ov::Shape input_shape_3{1, 1, n, n2};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<ov::snippets::lowered::LinearIR>& lir) {
        lir->set_loop_depth(2);
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto param3 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_3);
        auto brgemm1 = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto buffer1 = lir->push_node<ov::snippets::op::Buffer>(brgemm1.second);
        auto add = lir->push_node<ov::op::v1::Add>(buffer1.second, param2.second);
        auto buffer2 = lir->push_node<ov::snippets::op::Buffer>(add.second);
        auto brgemm2 = lir->push_node<ov::snippets::op::Brgemm>(buffer2.second, param3.second);
        init_expr_descriptors(*brgemm2.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto result = lir->push_node<ov::op::v0::Result>(brgemm2.second);

        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 n-blocking loop
        const auto brgemm1_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_output_port(0), 0)},
            false);
        const auto brgemm1_blocking_n_loop_id = loop_manager->add_loop_info(brgemm1_n_loop);

        // Brgemm2 n2-blocking loop
        const auto brgemm2_n2_loop = std::make_shared<UnifiedLoopInfo>(
            n2,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            false);
        const auto brgemm2_blocking_n2_loop_id = loop_manager->add_loop_info(brgemm2_n2_loop);

        // Shared m-blocking loop for brgemm1, add, and brgemm2
        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            false);
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        const auto inner_add_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)},
            false);
        const auto inner_add_loop_id = loop_manager->add_loop_info(inner_add_loop);

        const auto add_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)},
            blocking_m_loop);
        const auto add_m_split_loop_id = loop_manager->add_loop_info(add_m_split_loop);
        (*brgemm1.first)->set_loop_ids({blocking_m_loop_id, brgemm1_blocking_n_loop_id});
        (*buffer1.first)->set_loop_ids({blocking_m_loop_id});
        (*add.first)->set_loop_ids({blocking_m_loop_id, add_m_split_loop_id, inner_add_loop_id});
        (*buffer2.first)->set_loop_ids({blocking_m_loop_id});
        (*brgemm2.first)->set_loop_ids({blocking_m_loop_id, brgemm2_blocking_n2_loop_id});

        const auto buffer_after_brgemm1_expr = ov::as_type_ptr<BufferExpression>(*buffer1.first);
        const auto buffer_before_brgemm2_expr = ov::as_type_ptr<BufferExpression>(*buffer2.first);
        return std::make_tuple(buffer_after_brgemm1_expr, buffer_before_brgemm2_expr);
    };

    build_lir(linear_ir);
    const auto& [buffer1, buffer2] = build_lir(linear_ir_ref);
    buffer1->set_allocation_size(m_block * n);
    buffer2->set_allocation_size(m_block * n);
}


/*
 * Control Flow Graph:
 * LoopBegin (blocking_m_loop)
 * |  LoopBegin (blocking_n_loop)
 * |  |  Brgemm1
 * |  |  Buffer1
 * |  |  LoopBegin (add_m_split_loop)
 * |  |  |  LoopBegin (add_n_split_loop)
 * |  |  |  |  Add
 * |  |  |  LoopEnd (add_n_split_loop)
 * |  |  LoopEnd (add_m_split_loop)
 * |  LoopEnd (blocking_n_loop)
 * |
 * |  Buffer2
 * |
 * |  LoopBegin (brgemm2_n2_loop)
 * |  |  Brgemm2
 * |  LoopEnd (brgemm2_n2_loop)
 * LoopEnd (blocking_m_loop)
 */
TEST_F(ComputeBufferAllocationSizeTest, BrgemmAddBrgemmSplitMN) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const size_t n2 = 256;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::Shape input_shape_2{1, 1, m, n};
    const ov::Shape input_shape_3{1, 1, n, n2};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<ov::snippets::lowered::LinearIR>& lir) {
        lir->set_loop_depth(2);
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto param3 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_3);
        auto brgemm1 = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto buffer1 = lir->push_node<ov::snippets::op::Buffer>(brgemm1.second);
        auto add = lir->push_node<ov::op::v1::Add>(buffer1.second, param2.second);
        auto buffer2 = lir->push_node<ov::snippets::op::Buffer>(add.second);
        auto brgemm2 = lir->push_node<ov::snippets::op::Brgemm>(buffer2.second, param3.second);
        init_expr_descriptors(*brgemm2.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto result = lir->push_node<ov::op::v0::Result>(brgemm2.second);

        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 n-blocking loop
        const auto blocking_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)},
        false);
        const auto blocking_n_loop_id = loop_manager->add_loop_info(blocking_n_loop);

        const auto add_n_split_loop = make_inner_split_loop_info(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)},
            blocking_n_loop);
        const auto add_n_split_loop_id = loop_manager->add_loop_info(add_n_split_loop);

        // Brgemm2 n2-blocking loop
        const auto brgemm2_n2_loop = std::make_shared<UnifiedLoopInfo>(
            n2,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            false);
        const auto brgemm2_blocking_n2_loop_id = loop_manager->add_loop_info(brgemm2_n2_loop);

        // Shared m-blocking loop for brgemm1, add, and brgemm2
        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            false);
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        const auto add_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)},
            blocking_m_loop);
        const auto add_m_split_loop_id = loop_manager->add_loop_info(add_m_split_loop);

        (*brgemm1.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id});
        (*buffer1.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id});
        (*add.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id, add_m_split_loop_id, add_n_split_loop_id});
        (*buffer2.first)->set_loop_ids({blocking_m_loop_id});
        (*brgemm2.first)->set_loop_ids({blocking_m_loop_id, brgemm2_blocking_n2_loop_id});

        const auto buffer_after_brgemm1_expr = ov::as_type_ptr<BufferExpression>(*buffer1.first);
        const auto buffer_before_brgemm2_expr = ov::as_type_ptr<BufferExpression>(*buffer2.first);
        return std::make_tuple(buffer_after_brgemm1_expr, buffer_before_brgemm2_expr);
    };

    build_lir(linear_ir);
    const auto& [buffer1, buffer2] = build_lir(linear_ir_ref);
    buffer1->set_allocation_size(m_block * n_block);
    buffer2->set_allocation_size(m_block * n);
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
