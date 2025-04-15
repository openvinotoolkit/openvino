// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <transformations/cpu_opset/common/pass/move_readvalue_inputs_to_subgraph.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/read_value.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"

using namespace testing;
/****************************************************************
 * Pattern 1 (From whisper decoder):
 *  input                       input
 *    |                           |
 *  MatMul                      ReadValueWithSubgraph (MatMul)
 *    |                ->         |   \
 *  ReadValue                  Result  Assign
 *    |     \
 *  Result  Assign
 ****************************************************************/
static std::shared_ptr<ov::intel_cpu::ReadValueWithSubgraph> constructRVWithSubGraph(
    std::shared_ptr<ov::op::v0::Parameter> input,
    const ov::element::Type& type,
    std::shared_ptr<ov::op::util::Variable> variable) {
    auto mm_weights = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{2, 2}, std::vector<float>{1, 2, 3, 4});

    auto func_input =
        std::make_shared<ov::op::v0::Parameter>(input->get_element_type(), input->get_output_partial_shape(0));

    auto matmul = std::make_shared<ov::op::v0::MatMul>(func_input, mm_weights, false, false);

    auto func_output = std::make_shared<ov::op::v0::Result>(matmul);

    auto func = std::make_shared<ov::Model>(ov::NodeVector({func_output}),
                                            ov::ParameterVector{func_input},
                                            "state_init_submodel");

    auto readvalue = std::make_shared<ov::intel_cpu::ReadValueWithSubgraph>(variable, func);
    readvalue->set_input(input->output(0), func_input);
    readvalue->set_output(func_output);
    readvalue->validate_and_infer_types();

    return readvalue;
}

TEST(TransformationTests, ReadValueWithSubgraph_1) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        const ov::PartialShape shape{1, 1, 2};
        const ov::element::Type type = ov::element::f32;
        std::shared_ptr<ov::op::util::Variable> variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{1, 1, 2}, type, "var_id"});

        {
            auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);

            auto mm_weights =
                std::make_shared<ov::op::v0::Constant>(type, ov::Shape{2, 2}, std::vector<float>{1, 2, 3, 4});

            auto matmul = std::make_shared<ov::op::v0::MatMul>(input, mm_weights, false, false);

            auto readvalue = std::make_shared<ov::op::v6::ReadValue>(matmul, variable);

            auto assign = std::make_shared<ov::op::v6::Assign>(readvalue, variable);

            auto result = std::make_shared<ov::op::v0::Result>(readvalue);
            model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::SinkVector{assign},
                                                ov::ParameterVector{input});

            ov::pass::Manager manager;
            manager.register_pass<ov::intel_cpu::MoveReadValueInputsToSubgraph>();
            manager.run_passes(model);
        }
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);

            auto readvalue = constructRVWithSubGraph(input, type, variable);

            auto assign = std::make_shared<ov::op::v6::Assign>(readvalue, variable);

            auto result = std::make_shared<ov::op::v0::Result>(readvalue);

            model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                    ov::SinkVector{assign},
                                                    ov::ParameterVector{input});
        }
        auto res = compare_functions(model, model_ref, 0, 0, 0, 0, 0, 0);
        ASSERT_TRUE(res.first) << res.second;
    }
}

/***************************************************************************************************
 * Pattern 2 (Complex pattern):
 *           input                                  input
 *             |                                      |
 *          Convert                                Convert
 *         /   |   \                              /   |    \
 *        /    |    \                            /   Add2   \
 *     Add1   Add2   \                          |     |  \   |
 *      |      |  \   |         --->            |     |   Add3
 *       \     |   Add3                         |     |   /   \
 *        \    |   /   \               ReadValueWithSubgraph  Result2     Subgraph(Add1, Add4, Add5)
 *         \  Add4      \                           /   \
 *          \  |         \                     Result1  Assign
 *           Add5       Result2
 *             |
 *          ReadValue
 *           /   \
 *      Result1  Assign
 *
 ***************************************************************************************************/

static std::shared_ptr<ov::op::v0::Constant> create_const_node(ov::Shape shape) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i32, shape, std::vector<int32_t>{1});
}

static std::shared_ptr<ov::intel_cpu::ReadValueWithSubgraph> constructRVWithSubGraph2(
    ov::NodeVector inputs,
    const ov::element::Type& type,
    std::shared_ptr<ov::op::util::Variable> variable) {
    ov::ParameterVector func_inputs;
    for (auto input : inputs) {
        auto func_input =
            std::make_shared<ov::op::v0::Parameter>(input->get_element_type(), input->get_output_partial_shape(0));
        func_inputs.push_back(func_input);
    }

    auto add1 = std::make_shared<ov::op::v1::Add>(func_inputs[0], create_const_node(ov::Shape{4}));

    auto add4 = std::make_shared<ov::op::v1::Add>(func_inputs[1], func_inputs[2]);

    auto add5 = std::make_shared<ov::op::v1::Add>(add1, add4);

    auto func_output = std::make_shared<ov::op::v0::Result>(add5);

    auto func = std::make_shared<ov::Model>(ov::NodeVector({func_output}), func_inputs, "state_init_submodel");

    auto readvalue = std::make_shared<ov::intel_cpu::ReadValueWithSubgraph>(variable, func);
    for (size_t i = 0; i < inputs.size(); i++) {
        readvalue->set_input(inputs[i]->output(0), func_inputs[i]);
    }
    readvalue->set_output(func_output);
    readvalue->validate_and_infer_types();

    return readvalue;
}

TEST(TransformationTests, ReadValueWithSubgraph_2) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        const ov::PartialShape shape{1, 2, 4};
        const ov::element::Type in_type = ov::element::f32;
        const ov::element::Type out_type = ov::element::i32;

        std::shared_ptr<ov::op::util::Variable> variable =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{shape, out_type, "var_id"});

        {
            auto input = std::make_shared<ov::op::v0::Parameter>(in_type, shape);
            input->set_friendly_name("input");

            auto convert = std::make_shared<ov::op::v0::Convert>(input, out_type);
            convert->set_friendly_name("convert");

            auto add1 = std::make_shared<ov::op::v1::Add>(convert, create_const_node(ov::Shape{4}));
            add1->set_friendly_name("add1");

            auto add2 = std::make_shared<ov::op::v1::Add>(convert, create_const_node(ov::Shape{4}));
            add2->set_friendly_name("add2");

            auto add3 = std::make_shared<ov::op::v1::Add>(add2, convert);
            add3->set_friendly_name("add3");

            auto add4 = std::make_shared<ov::op::v1::Add>(add2, add3);
            add4->set_friendly_name("add4");

            auto add5 = std::make_shared<ov::op::v1::Add>(add1, add4);
            add5->set_friendly_name("add5");

            auto readvalue = std::make_shared<ov::op::v6::ReadValue>(add5, variable);
            readvalue->set_friendly_name("readvalue");

            auto assign = std::make_shared<ov::op::v6::Assign>(readvalue, variable);
            assign->set_friendly_name("assign");

            auto result1 = std::make_shared<ov::op::v0::Result>(readvalue);
            result1->set_friendly_name("result1");

            auto result2 = std::make_shared<ov::op::v0::Result>(add3);
            result2->set_friendly_name("result2");

            model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2},
                                                ov::SinkVector{assign},
                                                ov::ParameterVector{input});

            ov::pass::Manager manager;
            manager.register_pass<ov::intel_cpu::MoveReadValueInputsToSubgraph>();
            manager.run_passes(model);
        }
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(in_type, shape);

            auto convert = std::make_shared<ov::op::v0::Convert>(input, out_type);

            auto add2 = std::make_shared<ov::op::v1::Add>(convert, create_const_node(ov::Shape{4}));

            auto add3 = std::make_shared<ov::op::v1::Add>(add2, convert);

            auto readvalue = constructRVWithSubGraph2({convert, add2, add3}, out_type, variable);

            auto assign = std::make_shared<ov::op::v6::Assign>(readvalue, variable);

            auto result1 = std::make_shared<ov::op::v0::Result>(readvalue);

            auto result2 = std::make_shared<ov::op::v0::Result>(add3);

            model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2},
                                                    ov::SinkVector{assign},
                                                    ov::ParameterVector{input});
        }
        auto res = compare_functions(model, model_ref, 0, 0, 0, 0, 0, 0);
        ASSERT_TRUE(res.first) << res.second;
    }
}