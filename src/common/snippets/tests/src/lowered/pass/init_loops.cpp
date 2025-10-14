// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "lir_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using PortType = LoopPort::Type;

class InitLoopsTest : public LoweredPassTestsF {
public:
    InitLoopsTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<InitLoops>();
    }

    size_t vector_size = 16;
    ov::element::Type input_precision = ov::element::f32;
    size_t m_block = 32;
    size_t n_block = 64;
};

/*
 * Control Flow Graph:
 * LoopBegin (blocking_m_loop)
 * |  LoopBegin (brgemm_n_loop)
 * |  |  Brgemm1
 * |  LoopEnd (brgemm_n_loop)
 * |
 * |  Buffer
 * |
 * |  LoopBegin (add_m_split_loop)
 * |  |  LoopBegin (inner_add_loop)
 * |  | Load  Load
 * |  |    Add
 * |  |   Store
 * |  |  LoopEnd (inner_add_loop)
 * |  LoopEnd (add_m_split_loop)
 * LoopEnd (blocking_m_loop)
 */
TEST_F(InitLoopsTest, BrgemmAddSplitM) {
    const size_t m = 64;
    const size_t n = 256;
    const size_t k = 16;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::Shape input_shape_2{1, 1, m, n};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<ov::snippets::lowered::LinearIR>& lir,
                         const IOLoopPortDescs& m_loop_descs,
                         const IOLoopPortDescs& n_loop_descs,
                         const IOLoopPortDescs& m_split_loop_descs,
                         const IOLoopPortDescs& inner_add_loop_descs) {
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto brgemm = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto buffer = lir->push_node<ov::snippets::op::Buffer>(brgemm.second);
        auto load1 = lir->push_node<ov::snippets::op::Load>(buffer.second, vector_size);
        auto load2 = lir->push_node<ov::snippets::op::Load>(param2.second, vector_size);
        auto add = lir->push_node<ov::op::v1::Add>(load1.second, load2.second);
        auto store = lir->push_node<ov::snippets::op::Store>(add.second, vector_size);
        auto result = lir->push_node<ov::op::v0::Result>(store.second);

        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 n-blocking loop
        const auto brgemm_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_output_port(0), 0)},
            n_loop_descs.first,
            n_loop_descs.second,
            false);
        const auto brgemm_blocking_n_loop_id = loop_manager->add_loop_info(brgemm_n_loop);

        // Shared m-blocking loop for brgemm and add
        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            m_loop_descs.first,
            m_loop_descs.second,
            false);
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        const auto inner_add_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load1.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 0)},
            inner_add_loop_descs.first,
            inner_add_loop_descs.second,
            false);
        const auto inner_add_loop_id = loop_manager->add_loop_info(inner_add_loop);

        const auto add_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            blocking_m_loop,
            m_split_loop_descs);
        const auto add_m_split_loop_id = loop_manager->add_loop_info(add_m_split_loop);
        (*brgemm.first)->set_loop_ids({blocking_m_loop_id, brgemm_blocking_n_loop_id});
        (*buffer.first)->set_loop_ids({blocking_m_loop_id});
        (*load1.first)->set_loop_ids({blocking_m_loop_id, add_m_split_loop_id, inner_add_loop_id});
        (*load2.first)->set_loop_ids({blocking_m_loop_id, add_m_split_loop_id, inner_add_loop_id});
        (*add.first)->set_loop_ids({blocking_m_loop_id, add_m_split_loop_id, inner_add_loop_id});
        (*store.first)->set_loop_ids({blocking_m_loop_id, add_m_split_loop_id, inner_add_loop_id});
    };

    {
        auto gen_empty_descs = [](const size_t i_size, size_t o_size) {
            return std::make_pair(std::vector<UnifiedLoopInfo::LoopPortDesc>(i_size),
                                  std::vector<UnifiedLoopInfo::LoopPortDesc>(o_size));
        };
        build_lir(linear_ir,
                  {gen_empty_descs(3, 1)},
                  {gen_empty_descs(2, 1)},
                  {gen_empty_descs(2, 1)},
                  {gen_empty_descs(2, 1)});
    }
    {
        using LoopPortDesc = UnifiedLoopInfo::LoopPortDesc;
        const auto data_size = input_precision.size();

        const IOLoopPortDescs n_loop_descs = {
            {LoopPortDesc(0, 0, data_size), LoopPortDesc(1, -static_cast<int>(n), data_size)},
            {LoopPortDesc(1, -static_cast<int>(n), data_size)}};

        const IOLoopPortDescs m_loop_descs = {{LoopPortDesc(k, -static_cast<int>(k * m), data_size),
                                               LoopPortDesc(0, 0, data_size),
                                               LoopPortDesc(n, -static_cast<int>(n * m), data_size)},
                                              {LoopPortDesc(n, -static_cast<int>(n * m), data_size)}};

        const IOLoopPortDescs inner_add_loop_descs = {
            {LoopPortDesc(1, -static_cast<int>(n), data_size), LoopPortDesc(1, -static_cast<int>(n), data_size)},
            {LoopPortDesc(1, -static_cast<int>(n), data_size)}};

        const IOLoopPortDescs m_split_loop_descs = {{LoopPortDesc(n, -static_cast<int>(n * m_block), data_size),
                                                     LoopPortDesc(n, -static_cast<int>(n * m_block), data_size)},
                                                    {LoopPortDesc(n, -static_cast<int>(n * m_block), data_size)}};

        build_lir(linear_ir_ref, m_loop_descs, n_loop_descs, m_split_loop_descs, inner_add_loop_descs);
    }
}
/*
 * Control Flow Graph:
 * LoopBegin (blocking_m_loop)
 * |  LoopBegin (blocking_n_loop)
 * |  |  Brgemm
 * |  |  Buffer
 * |  |  LoopBegin (add_m_split_loop)
 * |  |  |  LoopBegin (add_n_split_loop)
 * |  |  |  |Load  Load
 * |  |  |  |  Add
 * |  |  |  | Store
 * |  |  |  LoopEnd (add_n_split_loop)
 * |  |  LoopEnd (add_m_split_loop)
 * |  LoopEnd (blocking_n_loop)
 * LoopEnd (blocking_m_loop)
 */
TEST_F(InitLoopsTest, BrgemmAddSplitMN) {
    const size_t m = 64;
    const size_t n = 256;
    const size_t k = 16;
    const ov::Shape input_shape_0{1, 1, m, k};
    const ov::Shape input_shape_1{1, 1, k, n};
    const ov::Shape input_shape_2{1, 1, m, n};
    const ov::snippets::VectorDims brgemm_a_subtensor{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims brgemm_b_subtensor{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims brgemm_c_subtensor{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<ov::snippets::lowered::LinearIR>& lir,
                         const IOLoopPortDescs& m_loop_descs,
                         const IOLoopPortDescs& n_loop_descs,
                         const IOLoopPortDescs& m_split_loop_descs,
                         const IOLoopPortDescs& n_split_loop_descs) {
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_0);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_1);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, input_shape_2);
        auto brgemm = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm.first, {brgemm_a_subtensor, brgemm_b_subtensor, brgemm_c_subtensor});
        auto buffer = lir->push_node<ov::snippets::op::Buffer>(brgemm.second);
        auto load1 = lir->push_node<ov::snippets::op::Load>(buffer.second, vector_size);
        auto load2 = lir->push_node<ov::snippets::op::Load>(param2.second, vector_size);
        auto add = lir->push_node<ov::op::v1::Add>(load1.second, load2.second);
        auto store = lir->push_node<ov::snippets::op::Store>(add.second, vector_size);
        auto result = lir->push_node<ov::op::v0::Result>(store.second);

        const auto& loop_manager = lir->get_loop_manager();

        // Shared n-blocking loop for brgemm and add
        const auto blocking_n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(1), 0),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 0)},
            n_loop_descs.first,
            n_loop_descs.second,
            false);
        const auto blocking_n_loop_id = loop_manager->add_loop_info(blocking_n_loop);

        // Shared m-blocking loop for brgemm and add
        const auto blocking_m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            m_loop_descs.first,
            m_loop_descs.second,
            false);
        const auto blocking_m_loop_id = loop_manager->add_loop_info(blocking_m_loop);

        const auto add_n_split_loop = make_inner_split_loop_info(
            n,
            vector_size,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load1.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 0)},
            blocking_n_loop,
            n_split_loop_descs);
        const auto add_n_split_loop_id = loop_manager->add_loop_info(add_n_split_loop);

        const auto add_m_split_loop = make_inner_split_loop_info(
            m,
            1,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::Incremented>((*load2.first)->get_input_port(0), 1)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)},
            blocking_m_loop,
            m_split_loop_descs);
        const auto add_m_split_loop_id = loop_manager->add_loop_info(add_m_split_loop);

        (*brgemm.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id});
        (*buffer.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id});
        (*load1.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id, add_m_split_loop_id, add_n_split_loop_id});
        (*load2.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id, add_m_split_loop_id, add_n_split_loop_id});
        (*add.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id, add_m_split_loop_id, add_n_split_loop_id});
        (*store.first)->set_loop_ids({blocking_m_loop_id, blocking_n_loop_id, add_m_split_loop_id, add_n_split_loop_id});
    };

    {
        auto gen_empty_descs = [](const size_t i_size, size_t o_size) {
            return std::make_pair(std::vector<UnifiedLoopInfo::LoopPortDesc>(i_size),
                                  std::vector<UnifiedLoopInfo::LoopPortDesc>(o_size));
        };
        build_lir(linear_ir,
                  {gen_empty_descs(3, 1)},
                  {gen_empty_descs(3, 1)},
                  {gen_empty_descs(2, 1)},
                  {gen_empty_descs(2, 1)});
    }
    {
        using LoopPortDesc = UnifiedLoopInfo::LoopPortDesc;
        const auto data_size = input_precision.size();

        const IOLoopPortDescs n_loop_descs = {{LoopPortDesc(0, 0, data_size),
                                               LoopPortDesc(1, -static_cast<int>(n), data_size),
                                               LoopPortDesc(1, -static_cast<int>(n), data_size)},
                                              {LoopPortDesc(1, -static_cast<int>(n), data_size)}};

        const IOLoopPortDescs m_loop_descs = {{LoopPortDesc(k, -static_cast<int>(k * m), data_size),
                                               LoopPortDesc(0, 0, data_size),
                                               LoopPortDesc(n, -static_cast<int>(n * m), data_size)},
                                              {LoopPortDesc(n, -static_cast<int>(n * m), data_size)}};

        const IOLoopPortDescs n_split_loop_descs = {{LoopPortDesc(1, -static_cast<int>(n_block), data_size),
                                                     LoopPortDesc(1, -static_cast<int>(n_block), data_size)},
                                                    {LoopPortDesc(1, -static_cast<int>(n_block), data_size)}};

        const IOLoopPortDescs m_split_loop_descs = {
            // Note: when m split loop is placed inside n blocking loop,
            // loop ports, which are inside the blocking loop, must have increment equal to n_block, not n
            // If the LoopPort is also blocking loop loop port, it still has increment equal to n
            {LoopPortDesc(n_block, -static_cast<int>(n_block * m_block), data_size),
             LoopPortDesc(n, -static_cast<int>(n * m_block), data_size)},
            {LoopPortDesc(n, -static_cast<int>(n * m_block), data_size)}};
        build_lir(linear_ir_ref, m_loop_descs, n_loop_descs, m_split_loop_descs, n_split_loop_descs);
    }
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
