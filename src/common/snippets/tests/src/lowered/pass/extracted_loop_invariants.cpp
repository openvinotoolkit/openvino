// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/pass/extract_loop_invariants.hpp"
#include "snippets/lowered/pass/normalize_loop_ids.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;
using PortType = LoopPort::Type;

class ExtractLoopInvariantsTest : public LoweredPassTestsF {
public:
    ExtractLoopInvariantsTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<ExtractLoopInvariants>();
    }
};

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsWithParams) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape0{1};
    const ov::Shape input_shape1{512};
    const std::vector<ov::snippets::VectorDims> layout_1d{{0}, {0}, {0}};
    const std::vector<ov::snippets::VectorDims> mul_subtensor{{1}, {1}, {1}};
    const std::vector<ov::snippets::VectorDims> sub_subtensor{{512}, {1}, {512}};
    /*
     *      Param00    Param01
     *         \       /
     *          Multiply(loopBegin)
     *              |
     *          Broadcast     Param1
     *               \         /
     *                Substract(loopBeginRef)
     *                    |
     *                  Result(LoopEnd and LoopEndRef)
    */
    {
        auto param00 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param01 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto multiply = linear_ir->push_node<ov::opset10::Multiply>(param00.second, param01.second);
        init_expr_descriptors(*multiply.first, mul_subtensor, layout_1d);
        auto broadcastmove = linear_ir->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 512);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(param1.second, broadcastmove.second);
        init_expr_descriptors(*sub.first, sub_subtensor, layout_1d);
        auto result = linear_ir->push_node<ov::opset10::Result>(sub.second);
        auto begin = multiply.first;
        auto end = result.first;
        linear_ir->get_loop_manager()->mark_loop(begin, end, 512, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0)),
                                                                       LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(1)),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0))},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0))});
        linear_ir->set_loop_depth(1);
    }
    {
        auto param00 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param01 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto multiply = linear_ir_ref->push_node<ov::opset10::Multiply>(param00.second, param01.second);
        init_expr_descriptors(*multiply.first, mul_subtensor, layout_1d);
        auto broadcastmove = linear_ir_ref->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 512);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(param1.second, broadcastmove.second);
        init_expr_descriptors(*sub.first, sub_subtensor, layout_1d);
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(sub.second);
        auto begin = sub.first;
        auto end = result.first;
        linear_ir_ref->get_loop_manager()->mark_loop(begin, end, 512, vector_size,
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0)),
                                                                           LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(1))},
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0))});
    }
}

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsWithScalar) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape scalar_shape{1};
    const ov::Shape input_shape0{1};
    const ov::Shape input_shape1{512};
    const std::vector<ov::snippets::VectorDims> layout_1d{{0}, {0}, {0}};
    const std::vector<ov::snippets::VectorDims> mul_subtensor{{1}, {1}, {1}};
    const std::vector<ov::snippets::VectorDims> sub_subtensor{{512}, {1}, {512}};
    /*
     *      Param0    Scalar(loopBegin)
     *         \       /
     *          Multiply
     *              |
     *          Broadcast     Param1
     *               \         /
     *                Substract(loopBeginRef)
     *                    |
     *                  Result(LoopEnd and LoopEndRef)
    */
    {
        auto param0 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto scalar = linear_ir->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 3.8f);
        auto multiply = linear_ir->push_node<ov::opset10::Multiply>(param0.second, scalar.second);
        init_expr_descriptors(*multiply.first, mul_subtensor, layout_1d);
        auto broadcastmove = linear_ir->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 512);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(param1.second, broadcastmove.second);
        init_expr_descriptors(*sub.first, sub_subtensor, layout_1d);
        auto result = linear_ir->push_node<ov::opset10::Result>(sub.second);
        auto begin = scalar.first;
        auto end = result.first;
        linear_ir->get_loop_manager()->mark_loop(begin, end, 512, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0)),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0))},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0))});
        linear_ir->set_loop_depth(1);
    }
    {
        auto param0 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape0);
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto scalar = linear_ir_ref->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 3.8f);
        auto multiply = linear_ir_ref->push_node<ov::opset10::Multiply>(param0.second, scalar.second);
        init_expr_descriptors(*multiply.first, mul_subtensor, layout_1d);
        auto broadcastmove = linear_ir_ref->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 512);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(param1.second, broadcastmove.second);
        init_expr_descriptors(*sub.first, sub_subtensor, layout_1d);
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(sub.second);
        auto begin = sub.first;
        auto end = result.first;
        linear_ir_ref->get_loop_manager()->mark_loop(begin, end, 512, vector_size,
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0)),
                                                                           LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(1))},
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0))});
    }
}

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsOutputLoopUpdateNotNeed) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape_a{3, 1};
    const ov::Shape input_shape_b{3, 16};
    const std::vector<ov::snippets::VectorDims> layout{{0, 1}, {0, 1}, {0, 1}};
    const std::vector<ov::snippets::VectorDims> subtensor_mul{{3, 1}, {3, 1}, {3, 1}};
    const std::vector<ov::snippets::VectorDims> subtensor_add{{3, 16}, {3, 16}, {3, 16}};
    /*
     *  Before: Param0, Param1, Param2, [[Multiply, Broadcast, Add, Sub]], Result0, Result1
     *  After:  Param0, Param1, Param2, [Multiply, Broadcast, [Add, Sub]], Result0, Result1
     *      Param0(3,1)    Param1(3,1)
     *             \       /
     *              Multiply
     *                 |
     *             Broadcast  Param2(3,16)
     *                    \   /
     *                     Add --- Result0
     *                      |
     *    Param3(3,16) --- Sub
     *                      |
     *                     Result1
    */
    {
        auto param0 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_a);
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_a);
        auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_b);
        auto param3 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_b);
        auto multiply = linear_ir->push_node<ov::opset10::Multiply>(param0.second, param1.second);
        init_expr_descriptors(*multiply.first, subtensor_mul, layout);
        auto broadcastmove = linear_ir->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 16);
        auto add = linear_ir->push_node<ov::opset10::Add>(param2.second, broadcastmove.second);
        init_expr_descriptors(*add.first, subtensor_add, layout);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(param3.second, add.second);
        auto result0 = linear_ir->push_node<ov::opset10::Result>(add.second);
        auto result1 = linear_ir->push_node<ov::opset10::Result>(sub.second);
        auto begin = multiply.first;
        auto end = result1.first;
        linear_ir->get_loop_manager()->mark_loop(begin, end, 16, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0)),
                                                                       LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(1)),
                                                                       LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0)),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0))},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0)),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0))});
        linear_ir->get_loop_manager()->mark_loop(begin, end, 3, 1,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0), 1),
                                                                       LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(1), 1),
                                                                       LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0), 1)},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0), 1)});
        linear_ir->set_loop_depth(2);
    }
    {
        auto param0 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_a);
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_a);
        auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_b);
        auto param3 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_b);
        auto multiply = linear_ir_ref->push_node<ov::opset10::Multiply>(param0.second, param1.second);
        init_expr_descriptors(*multiply.first, subtensor_mul, layout);
        auto broadcastmove = linear_ir_ref->push_node<ov::snippets::op::BroadcastMove>(multiply.second, 16);
        auto add = linear_ir_ref->push_node<ov::opset10::Add>(param2.second, broadcastmove.second);
        init_expr_descriptors(*add.first, subtensor_add, layout);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(param3.second, add.second);
        auto result0 = linear_ir_ref->push_node<ov::opset10::Result>(add.second);
        auto result1 = linear_ir_ref->push_node<ov::opset10::Result>(sub.second);
        auto begin_inner = add.first;
        auto end_inner = result1.first;
        {
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0),
                                                            LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0), 0)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0),
                                                          LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0), 0)};
            linear_ir_ref->get_loop_manager()->mark_loop(begin_inner, end_inner, 16, vector_size, entry_ports, exit_ports);
        }
        {
            auto begin_outer = multiply.first;
            auto end_outer = result1.first;
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0), 1),
                                                            LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(1), 1),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                                            LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0), 1)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1),
                                                          LoopPort::create<PortType::Incremented>((*sub.first)->get_output_port(0), 1)};
            linear_ir_ref->get_loop_manager()->mark_loop(begin_outer, end_outer, 3, 1, entry_ports, exit_ports);
        }
    }
}

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsFromInnermostToLoopOutside) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape_0{3, 512};
    const ov::Shape input_shape_1{1, 1};
    ov::snippets::VectorDims layout{0, 1};
    ov::snippets::VectorDims subtensor{3, 512};
    /*
     * before:       Param0, Param1, [[Broadcast, Add]], Result
     * intermediate: Param0, Param1, [Broadcast, [Add]], Result
     * after:        Param0, Param1, Broadcast, [[Add]], Result
     *      Param0(3,512)    Param1(1,1)
     *              \         /
     *               \    Broadcast
     *                \     /
     *                  Add
     *                   |
     *                 Result
    */
    {
        auto param_0 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_0);
        auto param_1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_1);
        auto broadcastmove = linear_ir->push_node<ov::snippets::op::BroadcastMove>(param_1.second, 512);
        init_expr_descriptors(*broadcastmove.first, {{1, 1}, subtensor}, {layout, layout});
        auto add = linear_ir->push_node<ov::opset10::Add>(param_0.second, broadcastmove.second);
        init_expr_descriptors(*add.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto result = linear_ir->push_node<ov::opset10::Result>(add.second);

        {
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*broadcastmove.first)->get_input_port(0), 0),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)};
            linear_ir->get_loop_manager()->mark_loop(broadcastmove.first, result.first, 512, vector_size, entry_ports, exit_ports);
        }
        {
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*broadcastmove.first)->get_input_port(0), 1),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)};
            linear_ir->get_loop_manager()->mark_loop(broadcastmove.first, result.first, 3, 1, entry_ports, exit_ports);
        }

        linear_ir->set_loop_depth(2);
    }
    {
        auto param_0 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_0);
        auto param_1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_1);
        auto broadcastmove = linear_ir_ref->push_node<ov::snippets::op::BroadcastMove>(param_1.second, 512);
        init_expr_descriptors(*broadcastmove.first, {{1, 1}, subtensor}, {layout, layout});
        auto add = linear_ir_ref->push_node<ov::opset10::Add>(param_0.second, broadcastmove.second);
        init_expr_descriptors(*add.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(add.second);

        {
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 0),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)};
            linear_ir_ref->get_loop_manager()->mark_loop(add.first, result.first, 512, vector_size, entry_ports, exit_ports);
        }
        {
            const auto entry_ports =  std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0), 1),
                                                            LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 1)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 1)};
            linear_ir_ref->get_loop_manager()->mark_loop(add.first, result.first, 3, 1, entry_ports, exit_ports);
        }
    }
}

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsImpossible) {
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape_0{32, 8, 1};
    ov::snippets::VectorDims order{1, 2, 0};
    ov::snippets::VectorDims layout{0, 1, 2};
    ov::snippets::VectorDims subtensor{1, 1};
    /* 
     *  < Transpose decomposition >
     *
     *        Param0(32,8,1)
     *             |
     *       LoadReorder with order (1,2,0)
     *             |
     *           Store
     *             |
     *           Result
    */
    {
        auto param = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_0);
        auto load_reshape = linear_ir->push_node<ov::snippets::op::LoadReorder>(param.second, 1, 0, layout);
        auto store = linear_ir->push_node<ov::snippets::op::Store>(load_reshape.second, 1, 0);
        init_expr_descriptors(*load_reshape.first, {subtensor, subtensor}, {order, layout});
        init_expr_descriptors(*store.first, {subtensor, subtensor}, {layout, layout});
        auto result = linear_ir->push_node<ov::opset10::Result>(store.second);

        {
            const auto entry_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_reshape.first)->get_input_port(0), 0)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 0)};
            linear_ir->get_loop_manager()->mark_loop(load_reshape.first, result.first, 32, 1, entry_ports, exit_ports);
        }
        {
            const auto entry_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*load_reshape.first)->get_input_port(0), 1)};
            const auto exit_ports = std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*store.first)->get_output_port(0), 1)};
            linear_ir->get_loop_manager()->mark_loop(load_reshape.first, result.first, 1, 1, entry_ports, exit_ports);
        }

        linear_ir->set_loop_depth(2);
    }
}

TEST_F(ExtractLoopInvariantsTest, ExtractedLoopInvariantsSplitLoops) {
    size_t vector_size = 16;
    size_t block_size = 32;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape_0{128, 512};
    const ov::Shape input_shape_1{512, 64};
    const ov::Shape input_shape_2{1, 1};
    const ov::snippets::VectorDims layout{0, 1};
    const ov::snippets::VectorDims subtensor{1, vector_size};
    /*
     *            Params    Param2(1,1)
     *              \         /
     *            MatMul   Broadcast
     *                \     /
     *                  Add
     *                   |
     *                 Result
    */
    {
        const auto param0 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_0);
        const auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_1);
        const auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape_2);
        const auto matmul = linear_ir->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        const auto broadcastmove = linear_ir->push_node<ov::snippets::op::BroadcastMove>(param2.second, input_shape_1.back());
        init_expr_descriptors(*broadcastmove.first, {{1, 1}, subtensor}, {layout, layout});
        const auto add = linear_ir->push_node<ov::opset10::Add>(matmul.second, broadcastmove.second);
        init_expr_descriptors(*add.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        const auto result = linear_ir->push_node<ov::opset10::Result>(add.second);
        const auto& loop_manager = linear_ir->get_loop_manager();
        loop_manager->mark_loop(matmul.first, broadcastmove.first, 128, block_size, 1,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*matmul.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::NotProcessed>((*matmul.first)->get_input_port(1))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*matmul.first)->get_output_port(0))});
        loop_manager->mark_loop(broadcastmove.first, result.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*broadcastmove.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0))});
        loop_manager->mark_loop(broadcastmove.first, result.first, 128, 1, 1,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*broadcastmove.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0))});
        ov::snippets::lowered::pass::SplitLoops().run(*linear_ir, linear_ir->begin(), linear_ir->end());
    }
    {
        const auto param0 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_0);
        const auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_1);
        const auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape_2);
        auto broadcastmove = linear_ir_ref->push_node<ov::snippets::op::BroadcastMove>(param2.second, input_shape_1.back());
        init_expr_descriptors(*broadcastmove.first, {{1, 1}, subtensor}, {layout, layout});
        const auto matmul = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param0.second, param1.second);
        const auto add = linear_ir_ref->push_node<ov::opset10::Add>(matmul.second, broadcastmove.second);
        init_expr_descriptors(*add.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(add.second);
        const auto& loop_manager = linear_ir_ref->get_loop_manager();
        loop_manager->mark_loop(matmul.first, add.first, 128, block_size, 1,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*matmul.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::NotProcessed>((*matmul.first)->get_input_port(1))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*matmul.first)->get_output_port(0))});
        loop_manager->mark_loop(add.first, result.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0))});
        loop_manager->mark_loop(add.first, result.first, 128, 1, 1,
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(0)),
                                                      LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1))},
                                std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0))});
        ov::snippets::lowered::pass::SplitLoops().run(*linear_ir_ref, linear_ir_ref->begin(), linear_ir_ref->end());
    }
}

class ExtractLoopInvariantsRemoveLoopsTest : public LoweredPassTestsF {
public:
    ExtractLoopInvariantsRemoveLoopsTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<ExtractLoopInvariants>();
        pipeline.register_pass<NormalizeLoopIDs>(); // loop could be removed and loop index could be different, normalize it
    }
};

// softmax with shape of 1 for innermost dimension.
// Cover multiple(all) exprs are extracted, and inner loops are removed.
TEST_F(ExtractLoopInvariantsRemoveLoopsTest, ExtractedLoopInvariantsAllExprsInLoopExtracted) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape{10, 1};
    ov::snippets::VectorDims layout{0, 1};
    ov::snippets::VectorDims subtensor{10, 1};
    /*
     *       Param  Vector
     *       |   |   |
     *       |  Maximum
     *       |   |
     *       | HorizonMax
     *       |   |
     *        Sub
     *         |
     * Vector Exp
     *   |   |   |
     *    Add    |
     *     |     |
     *    HAdd   |
     *     |     |
     *   Power   |
     *     |     |
     *    Multiply
     *       |
     *     Result
    */
    {
        auto param = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape);
        auto vector_max = linear_ir->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        auto vector_sum = linear_ir->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        auto max = linear_ir->push_node<ov::opset10::Maximum>(param.second, vector_max.second);
        init_expr_descriptors(*max.first, {{subtensor}, {1}, {subtensor}}, {layout, {0}, layout});
        auto hmax = linear_ir->push_node<ov::snippets::op::HorizonMax>(max.second);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(param.second, hmax.second);
        init_expr_descriptors(*sub.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto exp = linear_ir->push_node<ov::opset10::Exp>(sub.second);
        init_expr_descriptors(*exp.first, {subtensor, subtensor}, {layout, layout});
        auto add = linear_ir->push_node<ov::opset10::Add>(exp.second, vector_sum.second);
        init_expr_descriptors(*add.first, {subtensor, {1}, subtensor}, {layout, {0}, layout});
        auto hsum = linear_ir->push_node<ov::snippets::op::HorizonSum>(add.second);
        auto power_static = linear_ir->push_node<ov::snippets::op::PowerStatic>(hsum.second, -1.0f);
        init_expr_descriptors(*power_static.first, {subtensor, subtensor}, {layout, layout});
        auto multiply = linear_ir->push_node<ov::opset10::Multiply>(exp.second, power_static.second);
        init_expr_descriptors(*multiply.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto result = linear_ir->push_node<ov::opset10::Result>(multiply.second);
        // 3 inner loop
        linear_ir->get_loop_manager()->mark_loop(max.first, hmax.first, 1, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(0), 0),
                                                                       LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(1), 0)},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*max.first)->get_output_port(0), 0)});
        linear_ir->get_loop_manager()->mark_loop(sub.first, hsum.first, 1, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(0), 0),
                                                                       LoopPort::create<PortType::Incremented>((*sub.first)->get_input_port(1), 0),
                                                                       LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*exp.first)->get_output_port(0), 0),
                                                                       LoopPort::create<PortType::Incremented>((*add.first)->get_output_port(0), 0)});
        linear_ir->get_loop_manager()->mark_loop(multiply.first, result.first, 1, vector_size,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(0), 0),
                                                                       LoopPort::create<PortType::Incremented>((*multiply.first)->get_input_port(1), 0)},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_output_port(0), 0)});
        // outer loop info
        const auto loop_begin = std::make_shared<ov::snippets::op::LoopBegin>();
        auto loop_begin_expr = linear_ir->insert_node(loop_begin, std::vector<PortConnectorPtr>{}, {}, false, max.first);
        const auto loop_end = std::make_shared<ov::snippets::op::LoopEnd>();
        std::vector<PortConnectorPtr> loop_end_inputs{(*loop_begin_expr)->get_output_port_connector(0)};
        auto loop_end_expr = linear_ir->insert_node(loop_end, loop_end_inputs, {}, false, result.first);
        linear_ir->get_loop_manager()->mark_loop(loop_begin_expr, result.first, 10, 1,
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(0), 1),
                                                                       LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(1), 0),
                                                                       LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
                                                 std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_output_port(0), 1)});
        loop_end->set_id((*loop_end_expr)->get_loop_ids().back());
        linear_ir->set_loop_depth(2);
    }
    {
        auto param = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape);
        auto vector_max = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        auto vector_sum = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        auto max = linear_ir_ref->push_node<ov::opset10::Maximum>(param.second, vector_max.second);
        init_expr_descriptors(*max.first, {{subtensor}, {1}, {subtensor}}, {layout, {0}, layout});
        auto hmax = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(max.second);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(param.second, hmax.second);
        init_expr_descriptors(*sub.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto exp = linear_ir_ref->push_node<ov::opset10::Exp>(sub.second);
        init_expr_descriptors(*exp.first, {subtensor, subtensor}, {layout, layout});
        auto add = linear_ir_ref->push_node<ov::opset10::Add>(exp.second, vector_sum.second);
        init_expr_descriptors(*add.first, {subtensor, {1}, subtensor}, {layout, {0}, layout});
        auto hsum = linear_ir_ref->push_node<ov::snippets::op::HorizonSum>(add.second);
        auto power_static = linear_ir_ref->push_node<ov::snippets::op::PowerStatic>(hsum.second, -1.0f);
        init_expr_descriptors(*power_static.first, {subtensor, subtensor}, {layout, layout});
        auto multiply = linear_ir_ref->push_node<ov::opset10::Multiply>(exp.second, power_static.second);
        init_expr_descriptors(*multiply.first, {subtensor, subtensor, subtensor}, {layout, layout, layout});
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(multiply.second);
        // outer loop
        const auto loop_begin = std::make_shared<ov::snippets::op::LoopBegin>();
        auto loop_begin_expr = linear_ir_ref->insert_node(loop_begin, std::vector<PortConnectorPtr>{}, {}, false, max.first);
        const auto loop_end = std::make_shared<ov::snippets::op::LoopEnd>();
        std::vector<PortConnectorPtr> loop_end_inputs{(*loop_begin_expr)->get_output_port_connector(0)};
        auto loop_end_expr = linear_ir_ref->insert_node(loop_end, loop_end_inputs, {}, false, result.first);
        linear_ir_ref->get_loop_manager()->mark_loop(loop_begin_expr, result.first, 10, 1,
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(0), 1),
                                                                           LoopPort::create<PortType::Incremented>((*max.first)->get_input_port(1), 0),
                                                                           LoopPort::create<PortType::Incremented>((*add.first)->get_input_port(1), 0)},
                                                     std::vector<LoopPort>{LoopPort::create<PortType::Incremented>((*multiply.first)->get_output_port(0), 1)});
        loop_end->set_id((*loop_end_expr)->get_loop_ids().back());
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
