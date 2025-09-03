// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/split_loops.hpp"

#include "lir_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using PortType = LoopPort::Type;

namespace {
InnerSplittedUnifiedLoopInfoPtr make_inner_split_loop_info(size_t work_amount,
                                                           size_t increment,
                                                           const std::vector<LoopPort>& entries,
                                                           const std::vector<LoopPort>& exits,
                                                           const LoopInfoPtr& outer_split_loop_info) {
    // Note: this temporary loop is needed to easily create InnerSplittedUnifiedLoopInfo:
    // we extract all automatically calculated parameters from it such as LoopPortDesc and SpecificIterationHandlers
    const auto tmp_unified_loop = std::make_shared<UnifiedLoopInfo>(work_amount, increment, entries, exits);
    return std::make_shared<InnerSplittedUnifiedLoopInfo>(tmp_unified_loop->get_increment(),
                                                          tmp_unified_loop->get_input_ports(),
                                                          tmp_unified_loop->get_output_ports(),
                                                          tmp_unified_loop->get_input_port_descs(),
                                                          tmp_unified_loop->get_output_port_descs(),
                                                          tmp_unified_loop->get_handlers(),
                                                          outer_split_loop_info);
}
}  // namespace

class SplitLoopsTest : public LoweredPassTestsF {
public:
    SplitLoopsTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<SplitLoops>();
    }

    size_t vector_size = 16;
    ov::element::Type input_precision = ov::element::f32;
    size_t m_block = 32;
    size_t n_block = 64;
};

/*
 *      Param0   Param1
 *         \      /
 *          Brgemm
 *             |    Param2
 *             \    /
 *               Add
 *                |
 *              Result
 */
TEST_F(SplitLoopsTest, BrgemmAdd) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::Shape input_shape_2{1, 1, m, n};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};
    {
        auto param0 = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto brgemm = linear_ir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto add = linear_ir->push_node<ov::op::v1::Add>(brgemm.second, param2.second);
        auto result = linear_ir->push_node<ov::op::v0::Result>(add.second);

        const auto& loop_manager = linear_ir->get_loop_manager();
        const auto inner_add_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)});
        const auto inner_add_loop_id = loop_manager->add_loop_info(inner_add_loop);

        const auto outer_add_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)});
        const auto outer_add_loop_id = loop_manager->add_loop_info(outer_add_loop);

        const auto blocking_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 0)});
        const auto blocking_n_loop_id = loop_manager->add_loop_info(blocking_n_loop);

        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 1)});
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        (*add.first)->set_loop_ids({outer_add_loop_id, inner_add_loop_id});
        (*brgemm.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id});
    }

    {
        auto param0 = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto brgemm = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto add = linear_ir_ref->push_node<ov::op::v1::Add>(brgemm.second, param2.second);
        auto result = linear_ir_ref->push_node<ov::op::v0::Result>(add.second);

        const auto& loop_manager = linear_ir_ref->get_loop_manager();
        const auto brgemm_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 0)});
        const auto blocking_n_loop_id = loop_manager->add_loop_info(brgemm_n_loop);

        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)});
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        const auto inner_add_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)});
        const auto inner_add_loop_id = loop_manager->add_loop_info(inner_add_loop);

        const auto add_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)},
            blocking_m_loop);
        const auto add_m_split_loop_id = loop_manager->add_loop_info(add_m_split_loop);
        blocking_m_loop->register_pass_to_handler<SpecificLoopIterType::MAIN_BODY, SplitLoops::TransformInnerSplitLoop>();
        blocking_m_loop->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, SplitLoops::TransformInnerSplitLoop>();

        // Note: since it's impossible to set the desired loop id during loop info creation,
        // we have to reorder the loop ids in reference LIR to make them match the actual LIR
        const std::map<size_t, size_t> loop_ids_mapper = {
            {inner_add_loop_id, 0},
            {blocking_n_loop_id, 2},
            {blocking_m_loop_id, 3},
            {add_m_split_loop_id, 5}
        };
        loop_manager->reorder_identifiers(loop_ids_mapper);
        (*add.first)->set_loop_ids({3, 5, 0});
        (*brgemm.first)->set_loop_ids({3, 2});
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
