// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_specific_iterations.hpp"

#include "lir_test_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/lowered/pass/normalize_loop_ids.hpp"
#include "snippets/lowered/pass/validate_expanded_loops.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/store.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::op;
using namespace ov::snippets::utils;
using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

using PortType = LoopPort::Type;
using LoopPortDesc = UnifiedLoopInfo::LoopPortDesc;

class InsertSpecificIterationsTest : public LoweredPassTestsF {
public:
    InsertSpecificIterationsTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<InsertSpecificIterations>();
    }

    size_t vector_size = 16;
    ov::element::Type input_precision = ov::element::f32;
    size_t m_block = 32;
    size_t n_block = 64;
};

inline std::pair<LinearIR::constExprIt, std::shared_ptr<LoopEnd>> push_loop_end(const LinearIRPtr& linear_ir,
                                                                                const ExpressionPtr& loop_begin_expr,
                                                                                size_t loop_id) {
    const auto& loop_manager = linear_ir->get_loop_manager();
    const auto loop_info = loop_manager->get_loop_info(loop_id);
    auto get_ptr_increments = [&]() {
        if (auto unified_info = ov::as_type_ptr<UnifiedLoopInfo>(loop_info)) {
            return std::make_tuple(unified_info->get_ptr_increments(),
                                   unified_info->get_finalization_offsets(),
                                   unified_info->get_data_sizes());
        }
        if (auto expanded_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_info)) {
            return std::make_tuple(expanded_info->get_ptr_increments(),
                                   expanded_info->get_finalization_offsets(),
                                   expanded_info->get_data_sizes());
        }
        OPENVINO_THROW("Unsupported LoopInfo type for getting ptr increments");
    };

    std::vector<PortConnectorPtr> loop_end_inputs;
    loop_end_inputs.reserve(loop_info->get_input_count() + loop_info->get_output_count());
    loop_info->iterate_through_ports([&loop_end_inputs](const LoopPort& port) {
        loop_end_inputs.emplace_back(port.get_expr_port()->get_port_connector_ptr());
    });
    loop_end_inputs.emplace_back(loop_begin_expr->get_output_port_connector(0));

    const auto loop_begin = ov::as_type_ptr<LoopBegin>(loop_begin_expr->get_node());
    OPENVINO_ASSERT(loop_begin, "The expression is not LoopBegin");
    const auto [ptr_increments, finalization_offsets, data_sizes] = get_ptr_increments();
    const auto loop_end = std::make_shared<LoopEnd>(loop_begin,
                                                    loop_info->get_work_amount(),
                                                    loop_info->get_increment(),
                                                    loop_info->get_is_incremented(),
                                                    ptr_increments,
                                                    finalization_offsets,
                                                    data_sizes,
                                                    loop_info->get_input_count(),
                                                    loop_info->get_output_count(),
                                                    loop_id,
                                                    loop_info->is_parallel());
    auto expr_it = linear_ir->insert_node(loop_end, loop_end_inputs, {}, false, linear_ir->cend());
    return {expr_it, loop_end};
}

/*
 * Expected Control Flow Graph:
 * LoopBegin (outer_m_loop_main)
 * |  LoopBegin (inner_n_loop_1_main)
 * |  |  Brgemm1
 * |  LoopEnd (inner_n_loop_1_main)
 * LoopEnd (outer_m_loop_main)
 * LoopBegin (outer_m_loop_tail)
 * |  LoopBegin (inner_n_loop_2_main)
 * |  |  Brgemm2
 * |  LoopEnd (inner_n_loop_2_main)
 * LoopEnd (outer_m_loop_tail)
 */
TEST_F(InsertSpecificIterationsTest, InnerExpandedLoopsCloning) {
    const size_t m = 70;
    const size_t n = 256;
    const size_t k = 16;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};

    {
        auto param0 = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto outer_m_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto inner_n_loop_begin = linear_ir->push_node<LoopBegin>(false);

        auto brgemm = linear_ir->push_node<Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});

        const auto& loop_manager = linear_ir->get_loop_manager();
        const auto inner_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{LoopPortDesc(0, 0, 4), LoopPortDesc(1, -256, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -256, 4)},
            false);
        const auto inner_n_loop_id = loop_manager->add_loop_info(inner_n_loop);
        auto inner_n_loop_end = push_loop_end(linear_ir, *inner_n_loop_begin.first, inner_n_loop_id);

        const auto outer_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(16, -1120, 4), LoopPortDesc(0, 0, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(256, -17920, 4)},
            false);
        const auto outer_m_loop_id = loop_manager->add_loop_info(outer_m_loop);
        auto outer_m_loop_end = push_loop_end(linear_ir, *outer_m_loop_begin.first, outer_m_loop_id);
        auto result = linear_ir->push_node<ov::snippets::op::Result>(brgemm.second);

        (*inner_n_loop_begin.first)->set_loop_ids({outer_m_loop_id});
        (*brgemm.first)->set_loop_ids({outer_m_loop_id, inner_n_loop_id});
        (*inner_n_loop_end.first)->set_loop_ids({outer_m_loop_id});
    }

    {
        const size_t m_tail_work_amount = m % m_block;
        const size_t m_main_work_amount = m - m_tail_work_amount;
        auto param0 = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto outer_m_loop_begin_main = linear_ir_ref->push_node<LoopBegin>(false);
        auto inner_n_loop_1_begin = linear_ir_ref->push_node<LoopBegin>(false);

        auto brgemm1 = linear_ir_ref->push_node<Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});

        const std::vector<LoopPort> inner_n_loop_1_inputs{
            LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
            LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0)};
        const std::vector<LoopPort> inner_n_loop_1_outputs{
            LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_output_port(0), 0)};
        const IOLoopPortDescs n_loop_descs{{LoopPortDesc(0, 0, 4), LoopPortDesc(1, -256, 4)},
                                           {LoopPortDesc(1, -256, 4)}};

        // To these unified loops will be connected decomposed loop iterations
        const auto inner_n_loop_1_unified = std::make_shared<UnifiedLoopInfo>(n,
                                                                              n_block,
                                                                              inner_n_loop_1_inputs,
                                                                              inner_n_loop_1_outputs,
                                                                              n_loop_descs.first,
                                                                              n_loop_descs.second,
                                                                              false);
        const auto outer_m_loop_unified = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(16, -1120, 4), LoopPortDesc(0, 0, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(256, -17920, 4)},
            false);

        const auto& loop_manager = linear_ir_ref->get_loop_manager();
        const auto inner_n_loop_1_main = std::make_shared<ExpandedLoopInfo>(n,
                                                                            n_block,
                                                                            inner_n_loop_1_inputs,
                                                                            inner_n_loop_1_outputs,
                                                                            std::vector<int64_t>{0, 1, 1},
                                                                            std::vector<int64_t>{0, -256, -256},
                                                                            std::vector<int64_t>{4, 4, 4},
                                                                            SpecificLoopIterType::MAIN_BODY,
                                                                            inner_n_loop_1_unified);

        const auto inner_n_loop_1_id = loop_manager->add_loop_info(inner_n_loop_1_main);
        auto inner_n_loop_1_end = push_loop_end(linear_ir_ref, *inner_n_loop_1_begin.first, inner_n_loop_1_id);

        const auto outer_m_loop_main = std::make_shared<ExpandedLoopInfo>(
            m_main_work_amount,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_output_port(0), 1)},
            std::vector<int64_t>{16, 0, 256},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{4, 4, 4},
            SpecificLoopIterType::MAIN_BODY,
            outer_m_loop_unified);
        const auto outer_m_loop_id_main = loop_manager->add_loop_info(outer_m_loop_main);
        auto outer_m_loop_end_main = push_loop_end(linear_ir_ref, *outer_m_loop_begin_main.first, outer_m_loop_id_main);

        auto outer_m_loop_begin_tail = linear_ir_ref->push_node<LoopBegin>(false);
        auto inner_n_loop_2_begin = linear_ir_ref->push_node<LoopBegin>(false);

        auto brgemm2 = linear_ir_ref->push_node<Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm2.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});

        const auto inner_n_loop_2_unified = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            n_loop_descs.first,
            n_loop_descs.second,
            false);

        const auto inner_n_loop_2_main = std::make_shared<ExpandedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            std::vector<int64_t>{0, 1, 1},
            std::vector<int64_t>{0, -256, -256},
            std::vector<int64_t>{4, 4, 4},
            SpecificLoopIterType::MAIN_BODY,
            inner_n_loop_2_unified);

        const auto inner_n_loop_2_id = loop_manager->add_loop_info(inner_n_loop_2_main);
        auto inner_n_loop_2_end = push_loop_end(linear_ir_ref, *inner_n_loop_2_begin.first, inner_n_loop_2_id);

        const auto outer_m_loop_tail = std::make_shared<ExpandedLoopInfo>(
            m_tail_work_amount,
            m_tail_work_amount,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            std::vector<int64_t>{16, 0, 256},
            std::vector<int64_t>{-1120, 0, -17920},
            std::vector<int64_t>{4, 4, 4},
            SpecificLoopIterType::LAST_ITER,
            outer_m_loop_unified,
            true);
        const auto outer_m_loop_id_tail = loop_manager->add_loop_info(outer_m_loop_tail);
        auto outer_m_loop_end_tail = push_loop_end(linear_ir_ref, *outer_m_loop_begin_tail.first, outer_m_loop_id_tail);
        auto result = linear_ir_ref->push_node<ov::snippets::op::Result>(brgemm1.second, OutputVector{brgemm2.second});

        const std::map<ExpressionPtr, std::vector<size_t>> expr_to_loop_ids = {
            {*inner_n_loop_1_begin.first, {outer_m_loop_id_main}},
            {*brgemm1.first, {outer_m_loop_id_main, inner_n_loop_1_id}},
            {*inner_n_loop_1_end.first, {outer_m_loop_id_main}},
            {*inner_n_loop_2_begin.first, {outer_m_loop_id_tail}},
            {*brgemm2.first, {outer_m_loop_id_tail, inner_n_loop_2_id}},
            {*inner_n_loop_2_end.first, {outer_m_loop_id_tail}}};
        const std::map<size_t, size_t> loop_ids_mapper = {{inner_n_loop_1_id, 3},
                                                          {outer_m_loop_id_main, 4},
                                                          {inner_n_loop_2_id, 2},
                                                          {outer_m_loop_id_tail, 5}};
        assign_loop_ids(expr_to_loop_ids, loop_ids_mapper);
    }
}

/*
 * Control Flow Graph before transformation:
 * LoopBegin (m_outer_loop)
 * |  Load
 * |  LoopBegin (n_loop)
 * |  |  LoopBegin (m_split_loop)
 * |  |  |  Store
 * |  |  LoopEnd (m_split_loop)
 * |  LoopEnd (n_loop) - last iteration should be inserted
 * LoopEnd (m_outer_loop)
 */
TEST_F(InsertSpecificIterationsTest, OuterSplitLoopOutsideClonedLoop_InputPortOutside) {
    const size_t m = 64;
    const size_t n = 70;
    const ov::Shape input_shape{1, 1, m, n};
    {
        auto param = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape);

        auto m_outer_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto load = linear_ir->push_node<ov::snippets::op::Load>(param.second);

        auto n_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto m_split_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto store = linear_ir->push_node<ov::snippets::op::Store>(load.second);

        const auto& loop_manager = linear_ir->get_loop_manager();

        const auto n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            false,
            SpecificIterationHandlers(n, n_block, 0));
        const auto n_loop_id = loop_manager->add_loop_info(n_loop);

        const auto m_outer_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            false,
            SpecificIterationHandlers(m, m_block, 0));
        const auto m_outer_loop_id = loop_manager->add_loop_info(m_outer_loop);

        IOLoopPortDescs m_split_loop_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto load_convert_store_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            m_outer_loop,
            m_split_loop_descs);
        const auto m_split_loop_id = loop_manager->add_loop_info(load_convert_store_m_split_loop);

        auto m_split_loop_end = push_loop_end(linear_ir, *m_split_loop_begin.first, m_split_loop_id);
        auto n_loop_end = push_loop_end(linear_ir, *n_loop_begin.first, n_loop_id);
        auto m_outer_loop_end = push_loop_end(linear_ir, *m_outer_loop_begin.first, m_outer_loop_id);
        auto result = linear_ir->push_node<ov::snippets::op::Result>(store.second);

        (*load.first)->set_loop_ids({m_outer_loop_id});
        (*n_loop_begin.first)->set_loop_ids({m_outer_loop_id});
        (*m_split_loop_begin.first)->set_loop_ids({m_outer_loop_id, n_loop_id});
        (*store.first)->set_loop_ids({m_outer_loop_id, n_loop_id, m_split_loop_id});
        (*m_split_loop_end.first)->set_loop_ids({m_outer_loop_id, n_loop_id});
        (*n_loop_end.first)->set_loop_ids({m_outer_loop_id});
    }

    {
        const size_t n_tail_work_amount = n % n_block;
        const size_t n_main_work_amount = n - n_tail_work_amount;
        auto param = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape);

        auto m_outer_loop_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto load = linear_ir_ref->push_node<ov::snippets::op::Load>(param.second);

        // n_loop main iteration
        auto n_loop_main_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto m_split_loop_main_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto store_main = linear_ir_ref->push_node<ov::snippets::op::Store>(load.second);

        const auto& loop_manager = linear_ir_ref->get_loop_manager();

        // Unified loop infos for connection
        const auto n_loop_unified = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            false,
            SpecificIterationHandlers(n, n_block, 0));

        const auto m_outer_loop_unified = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            false,
            SpecificIterationHandlers(m, m_block, 0));

        // n_loop main expanded info
        const auto n_loop_main = std::make_shared<ExpandedLoopInfo>(
            n_main_work_amount,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 0)},
            std::vector<int64_t>{1, 1},
            std::vector<int64_t>{0, 0},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            n_loop_unified);
        const auto n_loop_main_id = loop_manager->add_loop_info(n_loop_main);

        // m_outer_loop main (only main since m=64 is divisible by m_block=32)
        const auto m_outer_loop_main = std::make_shared<ExpandedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-4480, -4480},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_outer_loop_unified);
        const auto m_outer_loop_main_id = loop_manager->add_loop_info(m_outer_loop_main);

        // m_split_loop for main n_loop
        IOLoopPortDescs m_split_loop_main_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto m_split_loop_main_unified = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 1)},
            m_outer_loop_unified,
            m_split_loop_main_descs);
        const auto m_split_loop_main_expanded = std::make_shared<ExpandedLoopInfo>(
            m_block,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_main.first)->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-2240, -2240},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_split_loop_main_unified);
        const auto m_split_loop_main_id = loop_manager->add_loop_info(m_split_loop_main_expanded);

        auto m_split_loop_main_end = push_loop_end(linear_ir_ref, *m_split_loop_main_begin.first, m_split_loop_main_id);
        auto n_loop_main_end = push_loop_end(linear_ir_ref, *n_loop_main_begin.first, n_loop_main_id);

        // n_loop tail iteration
        auto n_loop_tail_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto m_split_loop_tail_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto store_tail = linear_ir_ref->push_node<ov::snippets::op::Store>(load.second);

        // n_loop tail expanded info
        const auto n_loop_tail = std::make_shared<ExpandedLoopInfo>(
            n_tail_work_amount,
            n_tail_work_amount,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_output_port(0), 0)},
            std::vector<int64_t>{1, 1},
            std::vector<int64_t>{-70, -70},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::LAST_ITER,
            n_loop_unified,
            true);
        const auto n_loop_tail_id = loop_manager->add_loop_info(n_loop_tail);

        // m_split_loop for tail n_loop
        IOLoopPortDescs m_split_loop_tail_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto m_split_loop_tail_unified = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_output_port(0), 1)},
            m_outer_loop_unified,
            m_split_loop_tail_descs,
            false);
        const auto m_split_loop_cloned_expanded = std::make_shared<ExpandedLoopInfo>(
            m_block,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store_tail.first)->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-2240, -2240},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_split_loop_tail_unified,
            true);
        const auto m_split_loop_cloned_id = loop_manager->add_loop_info(m_split_loop_cloned_expanded);

        auto m_split_loop_tail_end = push_loop_end(linear_ir_ref, *m_split_loop_tail_begin.first, m_split_loop_cloned_id);
        auto n_loop_tail_end = push_loop_end(linear_ir_ref, *n_loop_tail_begin.first, n_loop_tail_id);

        // Note: outer loop is connected to the store's output from the last loop iteration of an inner loop
        m_outer_loop_main->replace_with_new_ports((*store_main.first)->get_output_port(0),
                                                  {(*store_tail.first)->get_output_port(0)});
        auto m_outer_loop_end = push_loop_end(linear_ir_ref, *m_outer_loop_begin.first, m_outer_loop_main_id);
        auto result = linear_ir_ref->push_node<ov::snippets::op::Result>(store_main.second, OutputVector{store_tail.second});

        const std::map<ExpressionPtr, std::vector<size_t>> expr_to_loop_ids = {
            {*load.first, {m_outer_loop_main_id}},
            {*n_loop_main_begin.first, {m_outer_loop_main_id}},
            {*m_split_loop_main_begin.first, {m_outer_loop_main_id, n_loop_main_id}},
            {*store_main.first, {m_outer_loop_main_id, n_loop_main_id, m_split_loop_main_id}},
            {*m_split_loop_main_end.first, {m_outer_loop_main_id, n_loop_main_id}},
            {*n_loop_main_end.first, {m_outer_loop_main_id}},
            {*n_loop_tail_begin.first, {m_outer_loop_main_id}},
            {*m_split_loop_tail_begin.first, {m_outer_loop_main_id, n_loop_tail_id}},
            {*store_tail.first, {m_outer_loop_main_id, n_loop_tail_id, m_split_loop_cloned_id}},
            {*m_split_loop_tail_end.first, {m_outer_loop_main_id, n_loop_tail_id}},
            {*n_loop_tail_end.first, {m_outer_loop_main_id}}};
        const std::map<size_t, size_t> loop_ids_mapper = {{n_loop_main_id, 5},
                                                          {m_outer_loop_main_id, 7},
                                                          {m_split_loop_main_id, 8},
                                                          {n_loop_tail_id, 6},
                                                          {m_split_loop_cloned_id, 9}};
        assign_loop_ids(expr_to_loop_ids, loop_ids_mapper);
    }
}

/*
 * Control Flow Graph before transformation:
 * LoopBegin (m_outer_loop)
 * |  LoopBegin (n_loop)
 * |  |  LoopBegin (m_split_loop)
 * |  |  |  Load
 * |  |  LoopEnd (m_split_loop)
 * |  LoopEnd (n_loop) - last iteration should be inserted
 * |  Store
 * LoopEnd (m_outer_loop)
 */
TEST_F(InsertSpecificIterationsTest, OuterSplitLoopOutsideClonedLoop_OutputPortOutside) {
    const size_t m = 64;
    const size_t n = 70;
    const ov::Shape input_shape{1, 1, m, n};

    {
        auto param = linear_ir->push_node<ov::op::v0::Parameter>(input_precision, input_shape);

        auto m_outer_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto n_loop_begin = linear_ir->push_node<LoopBegin>(false);
        auto m_split_loop_begin = linear_ir->push_node<LoopBegin>(false);

        auto load = linear_ir->push_node<ov::snippets::op::Load>(param.second);

        // Create store temporarily to get its ports for loop info creation
        auto store_temp = linear_ir->push_node<ov::snippets::op::Store>(load.second);
        // Store the expression locally and erase it from LIR
        auto store_expr = *store_temp.first;
        linear_ir->erase(store_temp.first);

        const auto& loop_manager = linear_ir->get_loop_manager();

        const auto n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            false);
        const auto n_loop_id = loop_manager->add_loop_info(n_loop);

        const auto m_outer_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(store_expr->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            false);
        const auto m_outer_loop_id = loop_manager->add_loop_info(m_outer_loop);

        IOLoopPortDescs m_split_loop_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto load_convert_store_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load.first)->get_output_port(0), 1)},
            m_outer_loop,
            m_split_loop_descs);
        const auto m_split_loop_id = loop_manager->add_loop_info(load_convert_store_m_split_loop);

        auto m_split_loop_end = push_loop_end(linear_ir, *m_split_loop_begin.first, m_split_loop_id);
        auto n_loop_end = push_loop_end(linear_ir, *n_loop_begin.first, n_loop_id);

        // Now re-insert the store at the correct position
        linear_ir->insert(linear_ir->cend(), store_expr);
        load.first->get()->get_output_port_connector(0)->add_consumer(store_expr->get_input_port(0));

        auto m_outer_loop_end = push_loop_end(linear_ir, *m_outer_loop_begin.first, m_outer_loop_id);
        auto result = linear_ir->push_node<ov::snippets::op::Result>(store_expr->get_node()->output(0));

        (*n_loop_begin.first)->set_loop_ids({m_outer_loop_id});
        (*m_split_loop_begin.first)->set_loop_ids({m_outer_loop_id, n_loop_id});
        (*load.first)->set_loop_ids({m_outer_loop_id, n_loop_id, m_split_loop_id});
        (*m_split_loop_end.first)->set_loop_ids({m_outer_loop_id, n_loop_id});
        (*n_loop_end.first)->set_loop_ids({m_outer_loop_id});
        store_expr->set_loop_ids({m_outer_loop_id});
    }

    {
        const size_t n_tail_work_amount = n % n_block;
        const size_t n_main_work_amount = n - n_tail_work_amount;
        auto param = linear_ir_ref->push_node<ov::op::v0::Parameter>(input_precision, input_shape);

        auto m_outer_loop_begin = linear_ir_ref->push_node<LoopBegin>(false);

        // n_loop main iteration
        auto n_loop_main_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto m_split_loop_main_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto load_main = linear_ir_ref->push_node<ov::snippets::op::Load>(param.second);

        const auto& loop_manager = linear_ir_ref->get_loop_manager();

        // Unified loop infos for connection
        const auto n_loop_unified = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(1, -70, 4)},
            false);

        // Create store temporarily to get its ports for unified loop info creation
        auto store_temp = linear_ir_ref->push_node<ov::snippets::op::Store>(load_main.second);
        auto store_temp_expr = *store_temp.first;
        linear_ir_ref->erase(store_temp.first);

        const auto m_outer_loop_unified = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(store_temp_expr->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            std::vector<LoopPortDesc>{LoopPortDesc(70, -4480, 4)},
            false);

        // n_loop main expanded info
        const auto n_loop_main = std::make_shared<ExpandedLoopInfo>(
            n_main_work_amount,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_output_port(0), 0)},
            std::vector<int64_t>{1, 1},
            std::vector<int64_t>{0, 0},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            n_loop_unified);
        const auto n_loop_main_id = loop_manager->add_loop_info(n_loop_main);

        // m_outer_loop main (only main since m=64 is divisible by m_block=32)
        const auto m_outer_loop_main = std::make_shared<ExpandedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>(store_temp_expr->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-4480, -4480},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_outer_loop_unified);
        const auto m_outer_loop_main_id = loop_manager->add_loop_info(m_outer_loop_main);

        // m_split_loop for main n_loop
        IOLoopPortDescs m_split_loop_main_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto m_split_loop_main_unified = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_output_port(0), 1)},
            m_outer_loop_unified,
            m_split_loop_main_descs);
        const auto m_split_loop_main_expanded = std::make_shared<ExpandedLoopInfo>(
            m_block,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_main.first)->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-2240, -2240},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_split_loop_main_unified);
        const auto m_split_loop_main_id = loop_manager->add_loop_info(m_split_loop_main_expanded);

        auto m_split_loop_main_end = push_loop_end(linear_ir_ref, *m_split_loop_main_begin.first, m_split_loop_main_id);
        auto n_loop_main_end = push_loop_end(linear_ir_ref, *n_loop_main_begin.first, n_loop_main_id);

        // n_loop tail iteration
        auto n_loop_tail_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto m_split_loop_tail_begin = linear_ir_ref->push_node<LoopBegin>(false);
        auto load_tail = linear_ir_ref->push_node<ov::snippets::op::Load>(param.second);

        // n_loop tail expanded info
        const auto n_loop_tail = std::make_shared<ExpandedLoopInfo>(
            n_tail_work_amount,
            n_tail_work_amount,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_output_port(0), 0)},
            std::vector<int64_t>{1, 1},
            std::vector<int64_t>{-70, -70},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::LAST_ITER,
            n_loop_unified,
            true);
        const auto n_loop_tail_id = loop_manager->add_loop_info(n_loop_tail);

        // m_split_loop for tail n_loop
        IOLoopPortDescs m_split_loop_tail_descs{{LoopPortDesc(70, -2240, 4)}, {LoopPortDesc(70, -2240, 4)}};
        const auto m_split_loop_tail_unified = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_output_port(0), 1)},
            m_outer_loop_unified,
            m_split_loop_tail_descs,
            false);
        const auto m_split_loop_cloned_expanded = std::make_shared<ExpandedLoopInfo>(
            m_block,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_tail.first)->get_output_port(0), 1)},
            std::vector<int64_t>{70, 70},
            std::vector<int64_t>{-2240, -2240},
            std::vector<int64_t>{4, 4},
            SpecificLoopIterType::MAIN_BODY,
            m_split_loop_tail_unified,
            true);
        const auto m_split_loop_cloned_id = loop_manager->add_loop_info(m_split_loop_cloned_expanded);

        auto m_split_loop_tail_end = push_loop_end(linear_ir_ref, *m_split_loop_tail_begin.first, m_split_loop_cloned_id);
        auto n_loop_tail_end = push_loop_end(linear_ir_ref, *n_loop_tail_begin.first, n_loop_tail_id);

        // Store operation placed after the loops (key difference from InputPortOutside test)
        auto store = linear_ir_ref->push_node<ov::snippets::op::Store>(load_tail.second);

        // Note: outer loop is connected to the store's output
        m_outer_loop_main->replace_with_new_ports(store_temp_expr->get_output_port(0),
                                                  {(*store.first)->get_output_port(0)});
        m_outer_loop_unified->replace_with_new_ports(store_temp_expr->get_output_port(0),
                                                     {(*store.first)->get_output_port(0)});
        auto m_outer_loop_end = push_loop_end(linear_ir_ref, *m_outer_loop_begin.first, m_outer_loop_main_id);
        auto result = linear_ir_ref->push_node<ov::snippets::op::Result>(store.second);

        const std::map<ExpressionPtr, std::vector<size_t>> expr_to_loop_ids = {
            {*n_loop_main_begin.first, {m_outer_loop_main_id}},
            {*m_split_loop_main_begin.first, {m_outer_loop_main_id, n_loop_main_id}},
            {*load_main.first, {m_outer_loop_main_id, n_loop_main_id, m_split_loop_main_id}},
            {*m_split_loop_main_end.first, {m_outer_loop_main_id, n_loop_main_id}},
            {*n_loop_main_end.first, {m_outer_loop_main_id}},
            {*n_loop_tail_begin.first, {m_outer_loop_main_id}},
            {*m_split_loop_tail_begin.first, {m_outer_loop_main_id, n_loop_tail_id}},
            {*load_tail.first, {m_outer_loop_main_id, n_loop_tail_id, m_split_loop_cloned_id}},
            {*m_split_loop_tail_end.first, {m_outer_loop_main_id, n_loop_tail_id}},
            {*n_loop_tail_end.first, {m_outer_loop_main_id}},
            {*store.first, {m_outer_loop_main_id}}};
        const std::map<size_t, size_t> loop_ids_mapper = {{n_loop_main_id, 5},
                                                          {m_outer_loop_main_id, 7},
                                                          {m_split_loop_main_id, 8},
                                                          {n_loop_tail_id, 6},
                                                          {m_split_loop_cloned_id, 9}};
        assign_loop_ids(expr_to_loop_ids, loop_ids_mapper);
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov