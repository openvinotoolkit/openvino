// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <iostream>

#include <openvino/pass/manager.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/rt_info/decompression.hpp>
#include "transformations/rt_info/fused_names_attribute.hpp"
#include <ngraph/pass/constant_folding.hpp>
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

std::ostream& operator<<(std::ostream& os, const std::unordered_set<std::string>& s) {
    for (auto it = s.begin(); it != s.end(); ++it) {
        if (it != s.begin()) {
            os << ", " << *it;
        } else {
            os << *it;
        }
    }
    return os;
}

class GetSupportedNodesTest : public ::testing::Test {
protected:
    ov::Shape m_shape{1, 84};
    std::shared_ptr<ov::Model> m_function;

public:
    void Run(std::function<void(std::shared_ptr<ov::Model>&)> transform,
             std::function<bool(const std::shared_ptr<ngraph::Node>)> is_node_supported,
             const std::unordered_set<std::string>& expected) {
        auto supported = InferenceEngine::GetSupportedNodes(m_function, transform, is_node_supported);
        ASSERT_TRUE(supported == expected) << "Expected list of supported nodes '" << expected
            << "' but actually received '" << supported << "'";
    }
};

TEST_F(GetSupportedNodesTest, UnsupportedCompressedConstantCF) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant_compressed = ngraph::op::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ngraph::opset9::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        ov::mark_as_decompression(convert);
        auto add = std::make_shared<ngraph::opset9::Add>(param, convert);
        add->set_friendly_name("add");
        auto result = std::make_shared<ngraph::op::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
    Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConstantFolding>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op);
    }, {});
}

TEST_F(GetSupportedNodesTest, ConstantSubgraphCF) {
    {
        auto constant_compressed1 = ngraph::op::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed1->set_friendly_name("constant_compressed1");
        auto convert1 = std::make_shared<ngraph::opset9::Convert>(constant_compressed1, ov::element::f32);
        convert1->set_friendly_name("constant1");
        ov::mark_as_decompression(convert1);
        auto constant_compressed2 = ngraph::op::Constant::create(ov::element::f16, m_shape, {2});
        constant_compressed2->set_friendly_name("constant_compressed2");
        auto convert2 = std::make_shared<ngraph::opset9::Convert>(constant_compressed2, ov::element::f32);
        convert2->set_friendly_name("constant2");
        ov::mark_as_decompression(convert2);
        auto add = std::make_shared<ngraph::opset9::Add>(convert1, convert2);
        add->set_friendly_name("add");
        auto const_reshape = ngraph::opset9::Constant::create(ngraph::element::i64, ov::Shape{1}, {84});
        const_reshape->set_friendly_name("const_reshape");
        auto reshape = std::make_shared<ngraph::opset9::Reshape>(add, const_reshape, false);
        reshape->set_friendly_name("reshape");
        auto result = std::make_shared<ngraph::op::Result>(reshape);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{});
    }
    Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConstantFolding>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op);
    }, {"constant_compressed1", "constant1", "constant_compressed2", "constant2", "add", "const_reshape", "reshape", "result"});
}

TEST_F(GetSupportedNodesTest, SupportedCompressedConstantNop) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant_compressed = ngraph::op::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ngraph::opset9::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        auto add = std::make_shared<ngraph::opset9::Add>(param, convert);
        add->set_friendly_name("add");
        auto result = std::make_shared<ngraph::op::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
    Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConvertPrecision>(precisions_array{{ngraph::element::f16, ngraph::element::f32}});
            m.register_pass<ngraph::pass::NopElimination>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op) ||
        (std::dynamic_pointer_cast<ngraph::opset9::Add>(op) != nullptr);
    }, {"input", "constant_compressed", "constant", "add", "result"});
}

TEST_F(GetSupportedNodesTest, SupportedConstantInsertAdditionalOp) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto mul_const = ngraph::op::Constant::create(ov::element::f32, m_shape, {1});
        mul_const->set_friendly_name("constant");
        auto mul = std::make_shared<ngraph::opset9::Multiply>(param, mul_const);
        mul->set_friendly_name("output_operation");
        auto result = std::make_shared<ngraph::op::Result>(mul);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
    Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.run_passes(model);
            for (auto& op : model->get_ops()) {
                if (std::dynamic_pointer_cast<ngraph::opset9::Multiply>(op) != nullptr) {
                    // Add one more dummy operation
                    auto consumers = op->output(0).get_target_inputs();
                    auto shape = op->get_shape();
                    auto add_const = ngraph::op::Constant::create(ov::element::f32, m_shape, {0});
                    auto add = std::make_shared<ngraph::opset9::Add>(op, add_const);
                    add->set_friendly_name(op->get_friendly_name());
                    op->set_friendly_name(op->get_friendly_name() + "/previous");
                    copy_runtime_info(op, add);
                    for (auto& consumer : consumers) {
                        consumer.replace_source_output(add);
                    }
                }
            }
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op) ||
        (std::dynamic_pointer_cast<ngraph::opset9::Multiply>(op) != nullptr) ||
        (std::dynamic_pointer_cast<ngraph::opset9::Add>(op) != nullptr);
    }, {"input", "constant", "output_operation", "result"});
}


TEST_F(GetSupportedNodesTest, PartiallySupportedCompressedConstant) {
    {
        auto param1 = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param1->set_friendly_name("input1");
        auto param2 = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param2->set_friendly_name("input2");
        auto constant_compressed = ngraph::op::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ngraph::opset9::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        ov::mark_as_decompression(convert);
        auto add = std::make_shared<ngraph::opset9::Add>(param1, convert);
        add->set_friendly_name("add");
        auto result1 = std::make_shared<ngraph::op::Result>(add);
        result1->set_friendly_name("result1");
        auto mul = std::make_shared<ngraph::opset9::Multiply>(param2, convert);
        mul->set_friendly_name("mul");
        auto result2 = std::make_shared<ngraph::op::Result>(mul);
        result2->set_friendly_name("result2");

        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result1, result2},
                                                 ngraph::ParameterVector{param1, param2});
    }
    Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConstantFolding>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op) ||
        (std::dynamic_pointer_cast<ngraph::opset9::Multiply>(op) != nullptr);
    }, {"input2", "constant_compressed", "constant", "mul", "result2"});
}

TEST_F(GetSupportedNodesTest, ConstantSubgraphSupported) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto weights =
                ngraph::opset9::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        weights->set_friendly_name("weights");
        auto shapeOf = std::make_shared<ngraph::opset9::ShapeOf>(weights);
        shapeOf->set_friendly_name("shapeof");
        auto const1 = ngraph::opset9::Constant::create(ov::element::Type_t::i32, {1}, {1});
        const1->set_friendly_name("const1");
        auto const2 = ngraph::opset9::Constant::create(ov::element::Type_t::i64, {}, {0});
        const2->set_friendly_name("const2");
        auto gather = std::make_shared<ngraph::opset9::Gather>(shapeOf, const1, const2);
        gather->set_friendly_name("gather");
        auto const3 = ngraph::opset9::Constant::create(ov::element::Type_t::i64, {1}, {1});
        const3->set_friendly_name("const3");
        auto concat = std::make_shared<ngraph::opset9::Concat>(ov::NodeVector{const3, gather}, 0);
        concat->set_friendly_name("concat");
        auto reshape = std::make_shared<ngraph::opset9::Reshape>(param, concat, false);
        reshape->set_friendly_name("reshape");
        auto matmul = std::make_shared<ngraph::opset9::MatMul>(reshape, weights, false, true);
        matmul->set_friendly_name("matmul");
        auto result = std::make_shared<ngraph::op::Result>(matmul);
        result->set_friendly_name("result");

        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
        Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConstantFolding>();
            m.register_pass<ngraph::pass::NopElimination>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op) ||
        (std::dynamic_pointer_cast<ngraph::opset9::MatMul>(op) != nullptr);
    }, {"input", "weights", "shapeof", "const1", "const2", "gather", "const3", "concat", "reshape", "matmul", "result"});
}

TEST_F(GetSupportedNodesTest, UnmarkedSupportedInputsOutputs) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant = ngraph::op::Constant::create(ov::element::f32, ov::Shape{m_shape[1]}, {1});
        constant->set_friendly_name("constant");
        auto const_reshape = ngraph::opset9::Constant::create(ngraph::element::i64, ov::Shape{2}, m_shape);
        const_reshape->set_friendly_name("const_reshape");
        auto reshape = std::make_shared<ngraph::opset9::Reshape>(constant, const_reshape, false);
        reshape->set_friendly_name("reshape");
        auto add = std::make_shared<ngraph::opset9::Add>(param, reshape);
        add->set_friendly_name("add");
        auto result = std::make_shared<ngraph::op::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
        Run([&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ngraph::pass::ConstantFolding>();
            m.run_passes(model);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        // Plugin don't mark input, constant and result as supported
        return (std::dynamic_pointer_cast<ngraph::opset9::Add>(op) != nullptr);
    }, {"input", "constant", "const_reshape", "reshape", "add", "result"});
}

TEST_F(GetSupportedNodesTest, WrongFusedNamesInOriginalModel) {
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto weights =
                ngraph::opset9::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        weights->set_friendly_name("weights");
        auto matmul = std::make_shared<ngraph::opset9::MatMul>(param, weights, false, true);
        matmul->get_rt_info()[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames("add");
        matmul->set_friendly_name("matmul");
        auto constant = ngraph::op::Constant::create(ov::element::f32, {1, 10}, {1});
        constant->set_friendly_name("constant");
        auto add = std::make_shared<ngraph::opset9::Add>(matmul, constant);
        add->get_rt_info()[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames("matmul");
        add->set_friendly_name("add");
        auto result = std::make_shared<ngraph::op::Result>(add);
        result->set_friendly_name("result");

        m_function = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                                 ngraph::ParameterVector{param});
    }
        Run([&](std::shared_ptr<ov::Model>& model) {
            return;
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::op::is_parameter(op) || ngraph::op::is_constant(op) || ngraph::op::is_output(op) ||
               (std::dynamic_pointer_cast<ngraph::opset9::MatMul>(op) != nullptr);
    }, {"input", "weights", "matmul"});
}