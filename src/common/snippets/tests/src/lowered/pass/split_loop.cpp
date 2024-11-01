// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

class SplitLoopTest : public LoweredPassTestsF {
public:
    void SetUp() override {
        pipeline.register_pass<SplitLoops>();
    }
};

TEST_F(SplitLoopTest, SplitLoopHmaxInInnerDimensionBlockLoopTest) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape1{512, 64};
    const ov::Shape input_shape2{64, 1024};
    const ov::Shape input_shape3{1024, 16};
    /* Brgemm1 and brgemm2 have two block loops.
     * HorizonMax, Fill1 and VectorBuffer should be included into block loop of inner most dimension.
     *
     *      Param1     Param2
     *         \        /
     *          Brgemm1     VectorBuffer
     *          |   |          |
     *          |  Fill0      Fill1
     *          |    \         /
     *          |      Maximum
     *          |         |
     *          |     HorizonMax
     *          \         /
     *           Substract       Param3
     *                \          /
     *                  Brgemm2
     *                     |
     *                   Result
    */
    {
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto brgemm1 = linear_ir->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        const auto vector_buffer = linear_ir->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value = 0xff7fffff;
        const auto initial_fill = linear_ir->push_node<ov::snippets::op::Fill>(vector_buffer.second, 0, fill_value);
        const auto fill = linear_ir->push_node<ov::snippets::op::Fill>(brgemm1.second, vector_size, fill_value);
        auto max = linear_ir->push_node<ov::opset10::Maximum>(initial_fill.second, fill.second);
        auto h_max = linear_ir->push_node<ov::snippets::op::HorizonMax>(max.second);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(brgemm1.second, h_max.second);
        auto brgemm2 = linear_ir->push_node<ov::snippets::op::Brgemm>(sub.second, param3.second);
        const auto result = linear_ir->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir->get_loop_manager();
        // two loops(N and M) for brgemm1. mark inner first as mark new loop inserted as outer loop as default.
        loop_manager->mark_loop(brgemm1.first, vector_buffer.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(brgemm1.first, vector_buffer.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 1)});
        // loops on column
        loop_manager->mark_loop(fill.first, h_max.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 0),
                                                      // skip (*max.first)->get_input_port(0) ? It is vector_buffer, not memory.
                                                      LoopPort((*max.first)->get_input_port(0), false, 0)},
                                std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(sub.first, brgemm2.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0),
                                                      LoopPort((*sub.first)->get_input_port(1), false, 0)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});
        // loop on row
        loop_manager->mark_loop(vector_buffer.first, brgemm2.first, 512, 1, 1,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1)});
        // two loops(K and M) for brgemm2
        loop_manager->mark_loop(brgemm2.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        loop_manager->mark_loop(brgemm2.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
    }
    {
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto brgemm1 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        const auto vector_buffer = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value = 0xff7fffff;
        const auto initial_fill = linear_ir_ref->push_node<ov::snippets::op::Fill>(vector_buffer.second, 0, fill_value);
        const auto fill = linear_ir_ref->push_node<ov::snippets::op::Fill>(brgemm1.second, vector_size, fill_value);
        auto max = linear_ir_ref->push_node<ov::opset10::Maximum>(initial_fill.second, fill.second);
        auto h_max = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(max.second);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(brgemm1.second, h_max.second);
        auto brgemm2 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(sub.second, param3.second);
        const auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir_ref->get_loop_manager();

        // block inner loops for dimension 0
        loop_manager->mark_loop(fill.first, h_max.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 0),
                                                      // skip (*max.first)->get_input_port(0) ? It is vector_buffer, not memory.
                                                      LoopPort((*max.first)->get_input_port(0), false, 0)},
                                std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(sub.first, brgemm2.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0),
                                                      LoopPort((*sub.first)->get_input_port(1), false, 0)},  // sub:brgemm1-Hmax
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});
        // block inner loop for dimension 1.
        loop_manager->mark_loop(vector_buffer.first, brgemm2.first, 32, 1, 1,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1)});
        // two block loops. All exprs between two brgemm including h_max should be in both block loops.
        loop_manager->mark_loop(brgemm1.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        loop_manager->mark_loop(brgemm1.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
    }
}

// extend child of Hsum, and move up as exprs with same loop id should be in a line together to execute.
TEST_F(SplitLoopTest, SplitLoopExtendHsumChildChain) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape1{512, 64};
    const ov::Shape input_shape2{64, 1024};
    const ov::Shape input_shape3{1024, 16};
    const ov::Shape buf_shape{512, 1};
    /* Brgemm1 and brgemm2 have two block loops.
     * HorizonMax and Multiply extended, Multiply is moved up. Divide not moved as data dependency.
     *
     *      Param1     Param2
     *         \        /
     *          Brgemm1
     *          |     |
     *          |    Relu
     *          |     |
     *          |   HorizonMax   Buffer
     *          \     |     \     /
     *           Substract   Multiply
     *             |    |          |
     *             |  HorizonSum   |
     *             |    \         /
     *             |      Divide     param3
     *              \        |       /
     *                    Brgemm2
     *                       |
     *                     Result
    */
    {
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto buffer = linear_ir->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto brgemm1 = linear_ir->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        auto relu = linear_ir->push_node<ov::opset10::Relu>(brgemm1.second);
        auto h_max = linear_ir->push_node<ov::snippets::op::HorizonMax>(relu.second);
        auto sub = linear_ir->push_node<ov::opset10::Subtract>(brgemm1.second, h_max.second);
        auto h_sum = linear_ir->push_node<ov::snippets::op::HorizonMax>(sub.second);
        auto mul = linear_ir->push_node<ov::opset10::Subtract>(buffer.second, h_max.second);
        auto div = linear_ir->push_node<ov::opset10::Subtract>(mul.second, h_sum.second);
        auto brgemm2 = linear_ir->push_node<ov::snippets::op::Brgemm>(sub.second, param3.second, div.second);
        const auto result = linear_ir->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir->get_loop_manager();
        // two loops for brgemm1
        loop_manager->mark_loop(brgemm1.first, relu.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0)},
                                // after split, result in reused buffer, should not increment
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(brgemm1.first, relu.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 1)});
        // loops on column
        loop_manager->mark_loop(relu.first, h_max.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(sub.first, h_sum.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});
        // loop on row
        loop_manager->mark_loop(relu.first, brgemm2.first, 512, 1, 1,
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub.first)->get_input_port(0), true, 1),
                                                      LoopPort((*mul.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1),
                                                      LoopPort((*div.first)->get_output_port(0), true, 1)});
        // two loops for brgemm2
        loop_manager->mark_loop(brgemm2.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(2), false, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        loop_manager->mark_loop(brgemm2.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(2), true, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
    }
    {
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto buffer = linear_ir_ref->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto brgemm1 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        auto relu = linear_ir_ref->push_node<ov::opset10::Relu>(brgemm1.second);
        auto h_max = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(relu.second);
        auto mul = linear_ir_ref->push_node<ov::opset10::Subtract>(h_max.second, buffer.second);
        auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(brgemm1.second, h_max.second);
        auto h_sum = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(sub.second);
        auto div = linear_ir_ref->push_node<ov::opset10::Subtract>(mul.second, h_sum.second);
        auto brgemm2 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(sub.second, param3.second, div.second);
        const auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir_ref->get_loop_manager();

        // block inner loops for dimension 0
        loop_manager->mark_loop(relu.first, h_max.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(sub.first, h_sum.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0),
                                                      LoopPort((*sub.first)->get_input_port(1), false, 0)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});

        // block inner loop for dimension 1.
        loop_manager->mark_loop(relu.first, brgemm2.first, 32, 1, 1,
                                std::vector<LoopPort>{LoopPort((*relu.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub.first)->get_input_port(0), true, 1),
                                                      LoopPort((*mul.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1),
                                                      LoopPort((*div.first)->get_output_port(0), true, 1)});
        // two block loops. All exprs between two brgemm including h_max should be in both block loops.
        loop_manager->mark_loop(brgemm1.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0),
                                                      LoopPort((*mul.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        loop_manager->mark_loop(brgemm1.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1),
                                                      LoopPort((*mul.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
    }
}

// full falsh attention case
TEST_F(SplitLoopTest, SplitLoopFlashAttentionTest) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape1{512, 64};
    const ov::Shape input_shape2{64, 1024};
    const ov::Shape input_shape3{1024, 16};
    const ov::Shape buf_shape{512, 1};
    {
        auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto buffer_max = linear_ir->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto buffer_sum = linear_ir->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto brgemm1 = linear_ir->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        // softmax max
        const auto vector_buffer_max = linear_ir->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value_max = 0xff7fffff;
        const auto initial_fill_max = linear_ir->push_node<ov::snippets::op::Fill>(vector_buffer_max.second, 0, fill_value_max);
        const auto fill_max = linear_ir->push_node<ov::snippets::op::Fill>(brgemm1.second, vector_size, fill_value_max);
        auto max = linear_ir->push_node<ov::opset10::Maximum>(initial_fill_max.second, fill_max.second);
        auto h_max = linear_ir->push_node<ov::snippets::op::HorizonMax>(max.second);
        // scale
        auto sub_scale = linear_ir->push_node<ov::opset10::Subtract>(buffer_max.second, h_max.second);
        auto exp_scale = linear_ir->push_node<ov::opset10::Exp>(sub_scale.second);

        // softmax sum
        const auto vector_buffer_sum = linear_ir->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value_sum = 0x00000000;
        const auto initial_fill_sum = linear_ir->push_node<ov::snippets::op::Fill>(vector_buffer_sum.second, 0, fill_value_sum);

        auto max_new = linear_ir->push_node<ov::opset10::Maximum>(buffer_max.second, h_max.second);  // max of old and new
        auto sub_softmax = linear_ir->push_node<ov::opset10::Subtract>(brgemm1.second, max_new.second);
        auto exp = linear_ir->push_node<ov::opset10::Exp>(sub_softmax.second);
        const auto fill_sum = linear_ir->push_node<ov::snippets::op::Fill>(exp.second, vector_size, fill_value_max);
        auto add = linear_ir->push_node<ov::opset10::Add>(initial_fill_sum.second, fill_sum.second);
        auto h_sum = linear_ir->push_node<ov::snippets::op::HorizonSum>(add.second);

        // softmax multiply
        auto power_static = linear_ir->push_node<ov::snippets::op::PowerStatic>(h_sum.second, -1);
        auto mul_softmax = linear_ir->push_node<ov::opset10::Multiply>(exp.second, power_static.second);
        // scale
        auto mul_scale = linear_ir->push_node<ov::opset10::Multiply>(buffer_sum.second, exp_scale.second);
        auto scale = linear_ir->push_node<ov::opset10::Divide>(mul_scale.second, h_sum.second);

        auto brgemm2 = linear_ir->push_node<ov::snippets::op::Brgemm>(mul_softmax.second, param3.second, scale.second);
        const auto result = linear_ir->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir->get_loop_manager();
        // two loops for brgemm1. mark inner first as mark new loop inserted as outer loop as default.
        size_t brgemm1_n = loop_manager->mark_loop(brgemm1.first, vector_buffer_max.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 0)});
        size_t brgemm1_m = loop_manager->mark_loop(brgemm1.first, vector_buffer_max.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 1)});

        // three loops on column-1024 [512,1024]
        size_t column_loop1 = loop_manager->mark_loop(fill_max.first, h_max.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*fill_max.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
        size_t column_loop2 = loop_manager->mark_loop(max_new.first, h_sum.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*max_new.first)->get_input_port(0), false, 0),
                                                      LoopPort((*max_new.first)->get_input_port(1), false, 0),
                                                      LoopPort((*sub_softmax.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*exp.first)->get_output_port(0), true, 0),
                                                      LoopPort((*add.first)->get_output_port(0), true, 0)});
        size_t column_loop3 = loop_manager->mark_loop(power_static.first, mul_scale.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*power_static.first)->get_input_port(0), false, 0),
                                                      LoopPort((*mul_softmax.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*mul_softmax.first)->get_output_port(0), true, 0)});
        // one loop on row-512 [512,1024]
        size_t row_loop = loop_manager->mark_loop(vector_buffer_max.first, brgemm2.first, 512, 1, 1,
                                std::vector<LoopPort>{LoopPort((*fill_max.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub_softmax.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub_scale.first)->get_input_port(0), true, 1),
                                                      LoopPort((*max_new.first)->get_input_port(0), true, 1),
                                                      LoopPort((*mul_scale.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*mul_softmax.first)->get_output_port(0), true, 1),
                                                      LoopPort((*scale.first)->get_output_port(0), true, 1)});
        // two loops for brgemm2
        size_t brgemm2_k = loop_manager->mark_loop(brgemm2.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(2), false, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        size_t brgemm2_m = loop_manager->mark_loop(brgemm2.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(2), true, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
        std::cout << "brgemm1_m:" << brgemm1_m << std::endl;
        std::cout << "brgemm1_n:" << brgemm1_n << std::endl;
        std::cout << "column_loop1:" << column_loop1 << std::endl;
        std::cout << "column_loop2:" << column_loop2 << std::endl;
        std::cout << "column_loop3:" << column_loop3 << std::endl;
        std::cout << "row_loop:" << row_loop << std::endl;
        std::cout << "brgemm2_m:" << brgemm2_m << std::endl;
        std::cout << "brgemm2_k:" << brgemm2_k << std::endl;
    }
    {
        auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
        auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
        auto param3 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
        auto buffer_max = linear_ir_ref->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto buffer_sum = linear_ir_ref->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
        auto brgemm1 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
        // softmax max
        const auto vector_buffer_max = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value_max = 0xff7fffff;
        const auto initial_fill_max = linear_ir_ref->push_node<ov::snippets::op::Fill>(vector_buffer_max.second, 0, fill_value_max);
        const auto fill_max = linear_ir_ref->push_node<ov::snippets::op::Fill>(brgemm1.second, vector_size, fill_value_max);
        auto max = linear_ir_ref->push_node<ov::opset10::Maximum>(initial_fill_max.second, fill_max.second);
        auto h_max = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(max.second);
        // scale
        auto sub_scale = linear_ir_ref->push_node<ov::opset10::Subtract>(buffer_max.second, h_max.second);
        auto exp_scale = linear_ir_ref->push_node<ov::opset10::Exp>(sub_scale.second);
        auto mul_scale = linear_ir_ref->push_node<ov::opset10::Multiply>(buffer_sum.second, exp_scale.second); // moved up to here

        // softmax sum
        const auto vector_buffer_sum = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
        uint32_t fill_value_sum = 0x00000000;
        const auto initial_fill_sum = linear_ir_ref->push_node<ov::snippets::op::Fill>(vector_buffer_sum.second, 0, fill_value_sum);

        auto max_new = linear_ir_ref->push_node<ov::opset10::Maximum>(buffer_max.second, h_max.second);  // max of old and new
        auto sub_softmax = linear_ir_ref->push_node<ov::opset10::Subtract>(brgemm1.second, max_new.second);
        auto exp = linear_ir_ref->push_node<ov::opset10::Exp>(sub_softmax.second);
        const auto fill_sum = linear_ir_ref->push_node<ov::snippets::op::Fill>(exp.second, vector_size, fill_value_max);
        auto add = linear_ir_ref->push_node<ov::opset10::Add>(initial_fill_sum.second, fill_sum.second);
        auto h_sum = linear_ir_ref->push_node<ov::snippets::op::HorizonSum>(add.second);
        // scale, moved up here
        auto scale = linear_ir_ref->push_node<ov::opset10::Divide>(mul_scale.second, h_sum.second);

        // softmax multiply
        auto power_static = linear_ir_ref->push_node<ov::snippets::op::PowerStatic>(h_sum.second, -1);
        auto mul_softmax = linear_ir_ref->push_node<ov::opset10::Multiply>(exp.second, power_static.second);

        auto brgemm2 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(mul_softmax.second, param3.second, scale.second);
        const auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm2.second);
        const auto& loop_manager = linear_ir_ref->get_loop_manager();

        // three block inner loops for dimension 0(0 means inner most dimension)
        loop_manager->mark_loop(fill_max.first, h_max.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*fill_max.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(max_new.first, h_sum.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*max_new.first)->get_input_port(0), false, 0),
                                                      LoopPort((*max_new.first)->get_input_port(1), false, 0),
                                                      LoopPort((*sub_softmax.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*exp.first)->get_output_port(0), true, 0),
                                                      // result of add should not store, just Hsum on vec reg.
                                                      LoopPort((*add.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(power_static.first, brgemm2.first, 64, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*power_static.first)->get_input_port(0), false, 0),
                                                      LoopPort((*mul_softmax.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*mul_softmax.first)->get_output_port(0), true, 0)});

        // block inner loop for dimension 1.
        loop_manager->mark_loop(vector_buffer_max.first, brgemm2.first, 32, 1, 1,
                                std::vector<LoopPort>{LoopPort((*fill_max.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub_softmax.first)->get_input_port(0), true, 1),
                                                      // three buffers inc 1 on each row loop
                                                      LoopPort((*sub_scale.first)->get_input_port(0), true, 1),
                                                      LoopPort((*max_new.first)->get_input_port(0), true, 1),
                                                      LoopPort((*mul_scale.first)->get_input_port(0), true, 1)},
                                                      // inc 64. store to buffer, inc based on buffer shape.
                                std::vector<LoopPort>{LoopPort((*mul_softmax.first)->get_output_port(0), true, 1),
                                                      // one row get one scale.
                                                      LoopPort((*scale.first)->get_output_port(0), true, 1)});

        // two block loops. All exprs between two brgemm are in both block loops.
        size_t block_nk = loop_manager->mark_loop(brgemm1.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0),
                                                      // three buffers no increased on N/K block loop.
                                                      LoopPort((*max_new.first)->get_input_port(0), false, 0),
                                                      LoopPort((*sub_scale.first)->get_input_port(0), false, 0),
                                                      LoopPort((*mul_scale.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},  // V matrix inc
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
        size_t block_m = loop_manager->mark_loop(brgemm1.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1),
                                                      // three buffers increased 32 on M block loop
                                                      LoopPort((*max_new.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub_scale.first)->get_input_port(0), true, 1),
                                                      LoopPort((*mul_scale.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},  // V matrix not inc
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
    }
}

// TEST_F(SplitLoopTest, SplitLoopTestReduceOp) {
//     size_t vector_size = 16;
//     const auto input_precision = ov::element::f32;
//     const ov::Shape input_shape1{512, 64};
//     const ov::Shape input_shape2{64, 1024};
//     const ov::Shape input_shape3{1024, 16};
//     const ov::Shape buf_shape{512, 1};
//     /*
//      *     Param1    Param2
//      *        \     /
//      *        Brgemm1
//      *         |   |
//      *         | ReduceMax  Buffer
//      *         |    \       /  |
//      *         |     Maximum   |
//      *         |     |    |    |
//      *        Substract   |    |
//      *             |      \    /
//      *             |       Add
//      *             \       /
//      *              Multiply    Param3
//      *                   \       /
//      *                    Brgemm2
//      *                       |
//      *                     Result
//     */
//    // Maximum have one loop, add moved after Max. expr in same loop together.
//     {
//         auto param1 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
//         auto param2 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
//         auto param3 = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
//         auto buffer = linear_ir->push_node<ov::snippets::op::Buffer>(buf_shape, input_precision);
//         auto brgemm1 = linear_ir->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
//         auto reduce_max = linear_ir->push_node<ov::snippets::op::ReduceMax>(brgemm1.second, 1);
//         auto max = linear_ir->push_node<ov::opset10::Maximum>(reduce_max.second, buffer.second);
//         auto sub = linear_ir->push_node<ov::opset10::Subtract>(brgemm1.second, max.second);
//         auto add = linear_ir->push_node<ov::opset10::Add>(max.second, buffer.second);
//         auto mul = linear_ir->push_node<ov::opset10::Divide>(sub.second, add.second);
//         auto brgemm2 = linear_ir->push_node<ov::snippets::op::Brgemm>(mul.second, param3.second);
//         const auto result = linear_ir->push_node<ov::opset10::Result>(brgemm2.second);
//         const auto& loop_manager = linear_ir->get_loop_manager();
//         // two loops for brgemm1
//         loop_manager->mark_loop(brgemm1.first, reduce_max.first, 512, 32, 1,
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*brgemm1.first)->get_input_port(1), false, 1)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 1)});
//         loop_manager->mark_loop(brgemm1.first, reduce_max.first, 1024, 64, 0,
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
//                                                       LoopPort((*brgemm1.first)->get_input_port(1), true, 0)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 0)});
//         // loops on column
//         loop_manager->mark_loop(reduce_max.first, max.first, 1024, vector_size, 0,
//                                 std::vector<LoopPort>{LoopPort((*reduce_max.first)->get_input_port(0), true, 0)},
//                                 std::vector<LoopPort>{LoopPort((*reduce_max.first)->get_output_port(0), false, 0)});
//         loop_manager->mark_loop(sub.first, brgemm2.first, 1024, vector_size, 0,
//                                 std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0),
//                                                       LoopPort((*sub.first)->get_input_port(1), false, 0),
//                                                       LoopPort((*mul.first)->get_input_port(1), false, 0)},
//                                 std::vector<LoopPort>{LoopPort((*mul.first)->get_output_port(0), true, 0)});
//         // loop on row
//         loop_manager->mark_loop(reduce_max.first, brgemm2.first, 512, 1, 1,
//                                 std::vector<LoopPort>{LoopPort((*reduce_max.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*max.first)->get_input_port(1), true, 1),
//                                                       LoopPort((*sub.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*add.first)->get_input_port(1), true, 1)},
//                                 std::vector<LoopPort>{LoopPort((*mul.first)->get_output_port(0), true, 1)});
//         // two loops for brgemm2
//         loop_manager->mark_loop(brgemm2.first, result.first, 512, 32, 1,
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
//         loop_manager->mark_loop(brgemm2.first, result.first, 1024, 64, 0,
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 0),
//                                                       LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
//     }
//     {
//         auto param1 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape1);
//         auto param2 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape2);
//         auto param3 = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape3);
//         auto brgemm1 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(param1.second, param2.second);
//         const auto vector_buffer = linear_ir_ref->push_node<ov::snippets::op::VectorBuffer>(input_precision);
//         uint32_t fill_value = 0xff7fffff;
//         const auto initial_fill = linear_ir_ref->push_node<ov::snippets::op::Fill>(vector_buffer.second, 0, fill_value);
//         const auto fill = linear_ir_ref->push_node<ov::snippets::op::Fill>(brgemm1.second, vector_size, fill_value);
//         auto max = linear_ir_ref->push_node<ov::opset10::Maximum>(initial_fill.second, fill.second);
//         auto h_max = linear_ir_ref->push_node<ov::snippets::op::HorizonMax>(max.second);
//         auto sub = linear_ir_ref->push_node<ov::opset10::Subtract>(brgemm1.second, h_max.second);
//         auto brgemm2 = linear_ir_ref->push_node<ov::snippets::op::Brgemm>(sub.second, param3.second);
//         const auto result = linear_ir_ref->push_node<ov::opset10::Result>(brgemm2.second);
//         const auto& loop_manager = linear_ir_ref->get_loop_manager();

//         // block inner loops for dimension 0
//         loop_manager->mark_loop(fill.first, h_max.first, 64, vector_size, 0,
//                                 std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 0),
//                                                       LoopPort((*max.first)->get_input_port(0), false, 0)},  // Max(initial_fill, fill)
//                                 std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
//         loop_manager->mark_loop(sub.first, brgemm2.first, 64, vector_size, 0,
//                                 std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0),
//                                                       LoopPort((*sub.first)->get_input_port(1), false, 0)},  // sub:brgemm1-Hmax
//                                 std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});
//         // block inner loop for dimension 1.
//         loop_manager->mark_loop(vector_buffer.first, brgemm2.first, 32, 1, 1,
//                                 std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*sub.first)->get_input_port(0), true, 1)},
//                                 std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1)});
//         // two block loops. All exprs between two brgemm including h_max should be in both block loops.
//         loop_manager->mark_loop(brgemm1.first, result.first, 512, 32, 1,
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
//                                                       LoopPort((*brgemm1.first)->get_input_port(1), false, 1),
//                                                       LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
//         loop_manager->mark_loop(brgemm1.first, result.first, 1024, 64, 0,
//                                 std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
//                                                       LoopPort((*brgemm1.first)->get_input_port(1), true, 0),
//                                                       LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
//                                 std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
//     }
// }

}  // namespace snippets
}  // namespace test
}  // namespace ov