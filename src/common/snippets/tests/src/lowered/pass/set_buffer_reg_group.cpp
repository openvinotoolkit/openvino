// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_buffer_reg_group.hpp"

#include "lir_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/lowered/pass/mark_invariant_shape_path.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/result.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using ov::snippets::op::LoopBegin;
using PortType = LoopPort::Type;
using LoopPortDesc = UnifiedLoopInfo::LoopPortDesc;

class SetBufferRegGroupTest : public LoweredPassTestsF {
public:
    SetBufferRegGroupTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::EXPR_ATTRS);
    }

    void SetUp() override {
        pipeline.register_pass<MarkInvariantShapePath>();
        pipeline.register_pass<SetBufferRegGroup>();
    }

    ov::element::Type input_precision = ov::element::f32;
    static constexpr size_t data_size = 4;  // float32

    static LoopPortDesc inc_desc(size_t work_amount) {
        return LoopPortDesc(1, -static_cast<int64_t>(work_amount), data_size);
    }
    static LoopPortDesc not_proc_desc() {
        return LoopPortDesc(0, 0, data_size);
    }
};

/*
 * Expected Control Flow Graph:
 * LoopBegin (m)
 * |  LoopBegin (n1)
 * |  |  Brgemm1
 * |  LoopEnd (n1)
 * |  Buffer1
 * |  LoopBegin (n_add)
 * |  |  Add
 * |  LoopEnd (n_add)
 * |  Buffer2
 * |  LoopBegin (n2)
 * |  |  Brgemm2
 * |  LoopEnd (n2)
 * LoopEnd (m)
 *
 * Buffer1 and Buffer2 are both outside inner loops and share register group 0.
 */
TEST_F(SetBufferRegGroupTest, TwoBuffersSameRegGroup) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const size_t m_block = 32;
    const size_t n_block = 64;
    const ov::Shape shape_a{1, 1, m, k};
    const ov::Shape shape_b{1, 1, k, n};
    const ov::Shape shape_c{1, 1, m, n};
    const ov::Shape shape_d{1, 1, n, n};
    const ov::snippets::VectorDims a_sub{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims b_sub{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims c_sub{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<LinearIR>& lir) {
        lir->set_loop_depth(2);
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_a);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_b);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_c);
        auto param3 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_d);
        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 + n-loop
        auto brgemm1 = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {a_sub, b_sub, c_sub});
        const auto n1_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{not_proc_desc(), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n1_id = loop_manager->add_loop_info(n1_loop);
        auto lb1 = lir->insert_node(std::make_shared<LoopBegin>(false),
                                    std::vector<PortConnectorPtr>{},
                                    {},
                                    false,
                                    brgemm1.first);
        auto le1 = insert_loop_end(lir, lb1, n1_id, lir->cend());

        // Buffer1 at m-loop level
        auto buffer1 = lir->push_node<ov::snippets::op::Buffer>(brgemm1.second);

        // Add + n-loop
        auto add = lir->push_node<ov::op::v1::Add>(buffer1.second, param2.second);
        const auto an_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto an_id = loop_manager->add_loop_info(an_loop);
        auto lb_an =
            lir->insert_node(std::make_shared<LoopBegin>(false), std::vector<PortConnectorPtr>{}, {}, false, add.first);
        auto le_an = insert_loop_end(lir, lb_an, an_id, lir->cend());

        // Buffer2 at m-loop level
        auto buffer2 = lir->push_node<ov::snippets::op::Buffer>(add.second);

        // Brgemm2 + n-loop
        auto brgemm2 = lir->push_node<ov::snippets::op::Brgemm>(buffer2.second, param3.second);
        init_expr_descriptors(*brgemm2.first, {c_sub, b_sub, c_sub});
        const auto n2_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n2_id = loop_manager->add_loop_info(n2_loop);
        auto lb2 = lir->insert_node(std::make_shared<LoopBegin>(false),
                                    std::vector<PortConnectorPtr>{},
                                    {},
                                    false,
                                    brgemm2.first);
        auto le2 = insert_loop_end(lir, lb2, n2_id, lir->cend());

        auto result = lir->push_node<ov::snippets::op::Result>(brgemm2.second);

        // m-loop
        const auto m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{inc_desc(m), not_proc_desc(), inc_desc(m), not_proc_desc()},
            std::vector<LoopPortDesc>{inc_desc(m)},
            false);
        const auto m_id = loop_manager->add_loop_info(m_loop);
        auto lb_m = lir->insert_node(std::make_shared<LoopBegin>(false),
                                     std::vector<PortConnectorPtr>{},
                                     {},
                                     false,
                                     lir->cend());
        auto le_m = insert_loop_end(lir, lb_m, m_id, lir->cend());

        // Set loop IDs
        (*lb_m)->set_loop_ids({m_id});
        (*lb1)->set_loop_ids({m_id, n1_id});
        (*brgemm1.first)->set_loop_ids({m_id, n1_id});
        (*le1)->set_loop_ids({m_id, n1_id});
        (*buffer1.first)->set_loop_ids({m_id});
        (*lb_an)->set_loop_ids({m_id, an_id});
        (*add.first)->set_loop_ids({m_id, an_id});
        (*le_an)->set_loop_ids({m_id, an_id});
        (*buffer2.first)->set_loop_ids({m_id});
        (*lb2)->set_loop_ids({m_id, n2_id});
        (*brgemm2.first)->set_loop_ids({m_id, n2_id});
        (*le2)->set_loop_ids({m_id, n2_id});
        (*le_m)->set_loop_ids({m_id});

        return std::make_pair(ov::as_type_ptr<BufferExpression>(*buffer1.first),
                              ov::as_type_ptr<BufferExpression>(*buffer2.first));
    };

    build_lir(linear_ir);
    const auto& [buffer1, buffer2] = build_lir(linear_ir_ref);
    buffer1->set_reg_group(0);
    buffer2->set_reg_group(0);
}

/*
 * Expected Control Flow Graph:
 * LoopBegin (m)
 * |  LoopBegin (n1)
 * |  |  Brgemm1
 * |  |  Buffer1
 * |  |  Add
 * |  LoopEnd (n1)
 * |  Buffer2
 * |  LoopBegin (n2)
 * |  |  Brgemm2
 * |  LoopEnd (n2)
 * LoopEnd (m)
 *
 * Buffer1 is inside n1 while Buffer2 is outside it, so they use groups 0 and 1.
 */
TEST_F(SetBufferRegGroupTest, BufferInsideVsOutsideLoop) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const size_t m_block = 32;
    const size_t n_block = 64;
    const ov::Shape shape_a{1, 1, m, k};
    const ov::Shape shape_b{1, 1, k, n};
    const ov::Shape shape_c{1, 1, m, n};
    const ov::Shape shape_d{1, 1, n, n};
    const ov::snippets::VectorDims a_sub{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims b_sub{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims c_sub{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<LinearIR>& lir) {
        lir->set_loop_depth(2);
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_a);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_b);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_c);
        auto param3 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_d);
        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 + Buffer1 + Add inside n-loop
        auto brgemm1 = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {a_sub, b_sub, c_sub});
        auto buffer1 = lir->push_node<ov::snippets::op::Buffer>(brgemm1.second);
        auto add = lir->push_node<ov::op::v1::Add>(buffer1.second, param2.second);

        const auto n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{not_proc_desc(), inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n_id = loop_manager->add_loop_info(n_loop);
        auto lb_n = lir->insert_node(std::make_shared<LoopBegin>(false),
                                     std::vector<PortConnectorPtr>{},
                                     {},
                                     false,
                                     brgemm1.first);
        auto le_n = insert_loop_end(lir, lb_n, n_id, lir->cend());

        // Buffer2 outside n-loop, at m-loop level
        auto buffer2 = lir->push_node<ov::snippets::op::Buffer>(add.second);

        // Brgemm2 + n-loop
        auto brgemm2 = lir->push_node<ov::snippets::op::Brgemm>(buffer2.second, param3.second);
        init_expr_descriptors(*brgemm2.first, {c_sub, b_sub, c_sub});
        const auto n2_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n2_id = loop_manager->add_loop_info(n2_loop);
        auto lb_n2 = lir->insert_node(std::make_shared<LoopBegin>(false),
                                      std::vector<PortConnectorPtr>{},
                                      {},
                                      false,
                                      brgemm2.first);
        auto le_n2 = insert_loop_end(lir, lb_n2, n2_id, lir->cend());

        auto result = lir->push_node<ov::snippets::op::Result>(brgemm2.second);

        // m-loop
        const auto m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{inc_desc(m), not_proc_desc(), inc_desc(m), not_proc_desc()},
            std::vector<LoopPortDesc>{inc_desc(m)},
            false);
        const auto m_id = loop_manager->add_loop_info(m_loop);
        auto lb_m = lir->insert_node(std::make_shared<LoopBegin>(false),
                                     std::vector<PortConnectorPtr>{},
                                     {},
                                     false,
                                     lir->cend());
        auto le_m = insert_loop_end(lir, lb_m, m_id, lir->cend());

        (*lb_m)->set_loop_ids({m_id});
        (*lb_n)->set_loop_ids({m_id, n_id});
        (*brgemm1.first)->set_loop_ids({m_id, n_id});
        (*buffer1.first)->set_loop_ids({m_id, n_id});
        (*add.first)->set_loop_ids({m_id, n_id});
        (*le_n)->set_loop_ids({m_id, n_id});
        (*buffer2.first)->set_loop_ids({m_id});
        (*lb_n2)->set_loop_ids({m_id, n2_id});
        (*brgemm2.first)->set_loop_ids({m_id, n2_id});
        (*le_n2)->set_loop_ids({m_id, n2_id});
        (*le_m)->set_loop_ids({m_id});

        return std::make_pair(ov::as_type_ptr<BufferExpression>(*buffer1.first),
                              ov::as_type_ptr<BufferExpression>(*buffer2.first));
    };

    build_lir(linear_ir);
    const auto& [buffer1, buffer2] = build_lir(linear_ir_ref);
    buffer1->set_reg_group(0);
    buffer2->set_reg_group(1);
}

/*
 * Expected Control Flow Graph:
 * LoopBegin (m)
 * |  LoopBegin (n1)
 * |  |  Brgemm1
 * |  |  Buffer1
 * |  |  Add1
 * |  |  Buffer2
 * |  |  Add2
 * |  LoopEnd (n1)
 * |  Buffer3
 * |  LoopBegin (n2)
 * |  |  Brgemm2
 * |  LoopEnd (n2)
 * LoopEnd (m)
 *
 * Buffer1 and Buffer2 share group 0 inside n1. Buffer3 is outside n1 and uses group 1.
 */
TEST_F(SetBufferRegGroupTest, ThreeBuffersInsideAndOutsideLoop) {
    const size_t m = 64;
    const size_t n = 128;
    const size_t k = 512;
    const size_t m_block = 32;
    const size_t n_block = 64;
    const ov::Shape shape_a{1, 1, m, k};
    const ov::Shape shape_b{1, 1, k, n};
    const ov::Shape shape_c{1, 1, m, n};
    const ov::Shape shape_d{1, 1, n, n};
    const ov::snippets::VectorDims a_sub{m_block, ov::snippets::utils::get_full_dim_value()};
    const ov::snippets::VectorDims b_sub{ov::snippets::utils::get_full_dim_value(), n_block};
    const ov::snippets::VectorDims c_sub{m_block, n_block};

    auto build_lir = [&](const std::shared_ptr<LinearIR>& lir) {
        lir->set_loop_depth(2);
        auto param0 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_a);
        auto param1 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_b);
        auto param2 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_c);
        auto param3 = lir->push_node<ov::op::v0::Parameter>(input_precision, shape_d);
        const auto& loop_manager = lir->get_loop_manager();

        // Brgemm1 + Buffer1 + Add1 + Buffer2 + Add2 all inside n-loop
        auto brgemm1 = lir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        init_expr_descriptors(*brgemm1.first, {a_sub, b_sub, c_sub});
        auto buffer1 = lir->push_node<ov::snippets::op::Buffer>(brgemm1.second);
        auto add1 = lir->push_node<ov::op::v1::Add>(buffer1.second, param2.second);
        auto buffer2 = lir->push_node<ov::snippets::op::Buffer>(add1.second);
        auto add2 = lir->push_node<ov::op::v1::Add>(buffer2.second, param2.second);

        const auto n_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(0)),
                                  LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(1), 0),
                                  LoopPort::create<PortType::Incremented>((*add1.first)->get_input_port(1), 0),
                                  LoopPort::create<PortType::Incremented>((*add2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add2.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{not_proc_desc(), inc_desc(n), inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n_id = loop_manager->add_loop_info(n_loop);
        auto lb_n = lir->insert_node(std::make_shared<LoopBegin>(false),
                                     std::vector<PortConnectorPtr>{},
                                     {},
                                     false,
                                     brgemm1.first);
        auto le_n = insert_loop_end(lir, lb_n, n_id, lir->cend());

        // Buffer3 outside n-loop, at m-loop level
        auto buffer3 = lir->push_node<ov::snippets::op::Buffer>(add2.second);

        // Brgemm2 + n-loop
        auto brgemm2 = lir->push_node<ov::snippets::op::Brgemm>(buffer3.second, param3.second);
        init_expr_descriptors(*brgemm2.first, {c_sub, b_sub, c_sub});
        const auto n2_loop = std::make_shared<UnifiedLoopInfo>(
            n,
            n_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(0), 0),
                                  LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_input_port(1), 0)},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 0)},
            std::vector<LoopPortDesc>{inc_desc(n), inc_desc(n)},
            std::vector<LoopPortDesc>{inc_desc(n)},
            false);
        const auto n2_id = loop_manager->add_loop_info(n2_loop);
        auto lb_n2 = lir->insert_node(std::make_shared<LoopBegin>(false),
                                      std::vector<PortConnectorPtr>{},
                                      {},
                                      false,
                                      brgemm2.first);
        auto le_n2 = insert_loop_end(lir, lb_n2, n2_id, lir->cend());

        auto result = lir->push_node<ov::snippets::op::Result>(brgemm2.second);

        // m-loop
        const auto m_loop = std::make_shared<UnifiedLoopInfo>(
            m,
            m_block,
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm1.first)->get_input_port(0), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm1.first)->get_input_port(1)),
                                  LoopPort::create<PortType::Incremented>((*add1.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::Incremented>((*add2.first)->get_input_port(1), 1),
                                  LoopPort::create<PortType::NotProcessed>((*brgemm2.first)->get_input_port(1))},
            std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*brgemm2.first)->get_output_port(0), 1)},
            std::vector<LoopPortDesc>{inc_desc(m), not_proc_desc(), inc_desc(m), inc_desc(m), not_proc_desc()},
            std::vector<LoopPortDesc>{inc_desc(m)},
            false);
        const auto m_id = loop_manager->add_loop_info(m_loop);
        auto lb_m = lir->insert_node(std::make_shared<LoopBegin>(false),
                                     std::vector<PortConnectorPtr>{},
                                     {},
                                     false,
                                     lir->cend());
        auto le_m = insert_loop_end(lir, lb_m, m_id, lir->cend());

        (*lb_m)->set_loop_ids({m_id});
        (*lb_n)->set_loop_ids({m_id, n_id});
        (*brgemm1.first)->set_loop_ids({m_id, n_id});
        (*buffer1.first)->set_loop_ids({m_id, n_id});
        (*add1.first)->set_loop_ids({m_id, n_id});
        (*buffer2.first)->set_loop_ids({m_id, n_id});
        (*add2.first)->set_loop_ids({m_id, n_id});
        (*le_n)->set_loop_ids({m_id, n_id});
        (*buffer3.first)->set_loop_ids({m_id});
        (*lb_n2)->set_loop_ids({m_id, n2_id});
        (*brgemm2.first)->set_loop_ids({m_id, n2_id});
        (*le_n2)->set_loop_ids({m_id, n2_id});
        (*le_m)->set_loop_ids({m_id});

        return std::make_tuple(ov::as_type_ptr<BufferExpression>(*buffer1.first),
                               ov::as_type_ptr<BufferExpression>(*buffer2.first),
                               ov::as_type_ptr<BufferExpression>(*buffer3.first));
    };

    build_lir(linear_ir);
    const auto& [buffer1, buffer2, buffer3] = build_lir(linear_ir_ref);
    buffer1->set_reg_group(0);
    buffer2->set_reg_group(0);
    buffer3->set_reg_group(1);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
