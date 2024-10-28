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
    SplitLoopTest() : LoweredPassTestsF() {
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_INDICES);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_DESCRIPTORS);
        comparator.enable(LIRComparator::LIRCmpValues::PORT_CONNECTORS);
        comparator.enable(LIRComparator::LIRCmpValues::LOOP_MANAGER);
    }

    void SetUp() override {
        pipeline.register_pass<SplitLoops>();
    }
};

TEST_F(SplitLoopTest, SplitLoopTestTwoDepthBlocks) {
    size_t vector_size = 16;
    const auto input_precision = ov::element::f32;
    const ov::Shape input_shape1{512, 64};
    const ov::Shape input_shape2{64, 1024};
    const ov::Shape input_shape3{1024, 16};
    /*
     *      Param1     Param2
     *         \        /
     *          Brgemm1     VectorBuffer
     *          |   |          |
     *          |  Fill       Fill
     *          |    \         /
     *          |      Maximum
     *          |         |
     *          |     HorizonMax
     *          |         |
     *           Substract       Param3
     *                |          /
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
        // two loops for brgemm1
        loop_manager->mark_loop(brgemm1.first, vector_buffer.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 1)});
        loop_manager->mark_loop(brgemm1.first, vector_buffer.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_output_port(0), true, 0)});
        // loops on column
        loop_manager->mark_loop(fill.first, h_max.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*max.first)->get_output_port(0), true, 0)});
        loop_manager->mark_loop(sub.first, brgemm2.first, 1024, vector_size, 0,
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_input_port(0), true, 0)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 0)});
        // loop on row
        loop_manager->mark_loop(vector_buffer.first, brgemm2.first, 512, 1, 1,
                                std::vector<LoopPort>{LoopPort((*fill.first)->get_input_port(0), true, 1),
                                                      LoopPort((*sub.first)->get_input_port(0), true, 1)},
                                std::vector<LoopPort>{LoopPort((*sub.first)->get_output_port(0), true, 1)});
        // two loops for brgemm2
        loop_manager->mark_loop(brgemm2.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
        loop_manager->mark_loop(brgemm2.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_input_port(0), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
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
                                                      LoopPort((*max.first)->get_input_port(0), false, 0)},  // Max(initial_fill, fill)
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
        loop_manager->mark_loop(brgemm1.first, result.first, 512, 32, 1,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), true, 1),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), false, 1),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), false, 1)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), true, 1)});
        loop_manager->mark_loop(brgemm1.first, result.first, 1024, 64, 0,
                                std::vector<LoopPort>{LoopPort((*brgemm1.first)->get_input_port(0), false, 0),
                                                      LoopPort((*brgemm1.first)->get_input_port(1), true, 0),
                                                      LoopPort((*brgemm2.first)->get_input_port(1), true, 0)},
                                std::vector<LoopPort>{LoopPort((*brgemm2.first)->get_output_port(0), false, 0)});
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov