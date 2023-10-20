// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <iostream>
#include <openvino/core/rt_info.hpp>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

std::ostream& operator<<(std::ostream& os, const std::unordered_set<std::string>& s);

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
             std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported,
             const std::unordered_set<std::string>& expected) {
        auto supported = ov::get_supported_nodes(m_function, transform, is_node_supported);
        auto const is_in_expected = [&expected](const std::string& x) {
            return expected.find(x) != expected.end();
        };
        bool is_equal =
            (supported.size() == expected.size()) && std::all_of(supported.begin(), supported.end(), is_in_expected);
        std::stringstream ss;
        if (!is_equal) {
            ss << "Expected list of supported nodes '" << expected << "' but actually received '" << supported << "'";
        }
        ASSERT_TRUE(is_equal) << ss.str();
    }
};

TEST_F(GetSupportedNodesTest, UnsupportedCompressedConstantCF) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant_compressed = ov::op::v0::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ov::op::v0::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        ov::mark_as_decompression(convert);
        auto add = std::make_shared<ov::op::v1::Add>(param, convert);
        add->set_friendly_name("add");
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConstantFolding>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op);
        },
        {});
}

TEST_F(GetSupportedNodesTest, ConstantSubgraphCF) {
    {
        auto constant_compressed1 = ov::op::v0::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed1->set_friendly_name("constant_compressed1");
        auto convert1 = std::make_shared<ov::op::v0::Convert>(constant_compressed1, ov::element::f32);
        convert1->set_friendly_name("constant1");
        ov::mark_as_decompression(convert1);
        auto constant_compressed2 = ov::op::v0::Constant::create(ov::element::f16, m_shape, {2});
        constant_compressed2->set_friendly_name("constant_compressed2");
        auto convert2 = std::make_shared<ov::op::v0::Convert>(constant_compressed2, ov::element::f32);
        convert2->set_friendly_name("constant2");
        ov::mark_as_decompression(convert2);
        auto add = std::make_shared<ov::op::v1::Add>(convert1, convert2);
        add->set_friendly_name("add");
        auto const_reshape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {84});
        const_reshape->set_friendly_name("const_reshape");
        auto reshape = std::make_shared<ov::op::v1::Reshape>(add, const_reshape, false);
        reshape->set_friendly_name("reshape");
        auto result = std::make_shared<ov::op::v0::Result>(reshape);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConstantFolding>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op);
        },
        {"constant_compressed1",
         "constant1",
         "constant_compressed2",
         "constant2",
         "add",
         "const_reshape",
         "reshape",
         "result"});
}

TEST_F(GetSupportedNodesTest, SupportedCompressedConstantNop) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant_compressed = ov::op::v0::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ov::op::v0::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        auto add = std::make_shared<ov::op::v1::Add>(param, convert);
        add->set_friendly_name("add");
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::f16, ov::element::f32}});
            m.register_pass<ov::pass::NopElimination>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v1::Add>(op) != nullptr);
        },
        {"input", "constant_compressed", "constant", "add", "result"});
}

TEST_F(GetSupportedNodesTest, SupportedConstantInsertAdditionalOp) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto mul_const = ov::op::v0::Constant::create(ov::element::f32, m_shape, {1});
        mul_const->set_friendly_name("constant");
        auto mul = std::make_shared<ov::op::v1::Multiply>(param, mul_const);
        mul->set_friendly_name("output_operation");
        auto result = std::make_shared<ov::op::v0::Result>(mul);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
            for (auto& op : model->get_ops()) {
                if (std::dynamic_pointer_cast<ov::op::v1::Multiply>(op) != nullptr) {
                    // Add one more dummy operation
                    auto consumers = op->output(0).get_target_inputs();
                    auto shape = op->get_shape();
                    auto add_const = ov::op::v0::Constant::create(ov::element::f32, m_shape, {0});
                    auto add = std::make_shared<ov::op::v1::Add>(op, add_const);
                    add->set_friendly_name(op->get_friendly_name());
                    op->set_friendly_name(op->get_friendly_name() + "/previous");
                    ov::copy_runtime_info(op, add);
                    for (auto& consumer : consumers) {
                        consumer.replace_source_output(add);
                    }
                }
            }
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v1::Multiply>(op) != nullptr) ||
                   (std::dynamic_pointer_cast<ov::op::v1::Add>(op) != nullptr);
        },
        {"input", "constant", "output_operation", "result"});
}

TEST_F(GetSupportedNodesTest, PartiallySupportedCompressedConstant) {
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param1->set_friendly_name("input1");
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param2->set_friendly_name("input2");
        auto constant_compressed = ov::op::v0::Constant::create(ov::element::f16, m_shape, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ov::op::v0::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("constant");
        ov::mark_as_decompression(convert);
        auto add = std::make_shared<ov::op::v1::Add>(param1, convert);
        add->set_friendly_name("add");
        auto result1 = std::make_shared<ov::op::v0::Result>(add);
        result1->set_friendly_name("result1");
        auto mul = std::make_shared<ov::op::v1::Multiply>(param2, convert);
        mul->set_friendly_name("mul");
        auto result2 = std::make_shared<ov::op::v0::Result>(mul);
        result2->set_friendly_name("result2");

        m_function =
            std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConstantFolding>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v1::Multiply>(op) != nullptr);
        },
        {"input2", "constant_compressed", "constant", "mul", "result2"});
}

TEST_F(GetSupportedNodesTest, ConstantSubgraphSupported) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto weights = ov::op::v0::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        weights->set_friendly_name("weights");
        auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(weights);
        shapeOf->set_friendly_name("shapeof");
        auto const1 = ov::op::v0::Constant::create(ov::element::Type_t::i32, {1}, {1});
        const1->set_friendly_name("const1");
        auto const2 = ov::op::v0::Constant::create(ov::element::Type_t::i64, {}, {0});
        const2->set_friendly_name("const2");
        auto gather = std::make_shared<ov::op::v8::Gather>(shapeOf, const1, const2);
        gather->set_friendly_name("gather");
        auto const3 = ov::op::v0::Constant::create(ov::element::Type_t::i64, {1}, {1});
        const3->set_friendly_name("const3");
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{const3, gather}, 0);
        concat->set_friendly_name("concat");
        auto reshape = std::make_shared<ov::op::v1::Reshape>(param, concat, false);
        reshape->set_friendly_name("reshape");
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, weights, false, true);
        matmul->set_friendly_name("matmul");
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        result->set_friendly_name("result");

        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConstantFolding>();
            m.register_pass<ov::pass::NopElimination>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v0::MatMul>(op) != nullptr);
        },
        {"input",
         "weights",
         "shapeof",
         "const1",
         "const2",
         "gather",
         "const3",
         "concat",
         "reshape",
         "matmul",
         "result"});
}

TEST_F(GetSupportedNodesTest, UnmarkedSupportedInputsOutputs) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{m_shape[1]}, {1});
        constant->set_friendly_name("constant");
        auto const_reshape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, m_shape);
        const_reshape->set_friendly_name("const_reshape");
        auto reshape = std::make_shared<ov::op::v1::Reshape>(constant, const_reshape, false);
        reshape->set_friendly_name("reshape");
        auto add = std::make_shared<ov::op::v1::Add>(param, reshape);
        add->set_friendly_name("add");
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ConstantFolding>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            // Plugin don't mark input, constant and result as supported
            return (std::dynamic_pointer_cast<ov::op::v1::Add>(op) != nullptr);
        },
        {"add"});
}

TEST_F(GetSupportedNodesTest, WrongFusedNamesInOriginalModel) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto weights = ov::op::v0::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        weights->set_friendly_name("weights");
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, true);
        matmul->get_rt_info()[ov::FusedNames::get_type_info_static()] = ov::FusedNames("add");
        matmul->set_friendly_name("matmul");
        auto constant = ov::op::v0::Constant::create(ov::element::f32, {1, 10}, {1});
        constant->set_friendly_name("constant");
        auto add = std::make_shared<ov::op::v1::Add>(matmul, constant);
        add->get_rt_info()[ov::FusedNames::get_type_info_static()] = ov::FusedNames("matmul");
        add->set_friendly_name("add");
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result");

        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            return;
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v0::MatMul>(op) != nullptr);
        },
        {"input", "weights", "matmul"});
}

TEST_F(GetSupportedNodesTest, FusedNamesSupportedUnsupportedBoth) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto dummy_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        dummy_param->set_friendly_name("dummy_param");
        auto logsoftmax = std::make_shared<ov::op::v5::LogSoftmax>(param, 1);
        logsoftmax->set_friendly_name("logsoftmax");
        auto result = std::make_shared<ov::op::v0::Result>(logsoftmax);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, dummy_param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::LogSoftmaxDecomposition>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            // Exp is not supported and all constants are missing
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) ||
                   (std::dynamic_pointer_cast<ov::op::v1::ReduceMax>(op) != nullptr) ||
                   (std::dynamic_pointer_cast<ov::op::v1::Subtract>(op) != nullptr) ||
                   (std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(op) != nullptr) ||
                   (std::dynamic_pointer_cast<ov::op::v0::Log>(op) != nullptr);
        },
        {"dummy_param"});  // kepp dummy only since it has no unsupported consumers
}

TEST_F(GetSupportedNodesTest, ShuffleChannelFusion) {
    {
        ov::Shape input_shape = {1, 112, 56, 56};
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        input->set_friendly_name("input");

        ov::Shape reshape_before_shape = {1, 4, 28, 56, 56};
        auto shape_reshape_before = ov::op::v0::Constant::create(ov::element::i64,
                                                                 ov::Shape{reshape_before_shape.size()},
                                                                 reshape_before_shape);
        shape_reshape_before->set_friendly_name("shape_reshape_before");
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(input, shape_reshape_before, true);
        reshape_before->set_friendly_name("reshape_before");

        ov::Shape permute_order = {0, 2, 1, 3, 4};
        auto permutation =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{permute_order.size()}, permute_order);
        permutation->set_friendly_name("permutation");
        auto permute = std::make_shared<ov::op::v1::Transpose>(reshape_before, permutation);
        permute->set_friendly_name("permute");

        auto shape_reshape_after =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{input_shape.size()}, input_shape);
        shape_reshape_after->set_friendly_name("shape_reshape_after");
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(permute, shape_reshape_after, true);
        reshape_after->set_friendly_name("reshape_after");

        m_function = std::make_shared<ov::Model>(ov::NodeVector{reshape_after}, ov::ParameterVector{input});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::CommonOptimizations>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::op::util::is_constant(op);
        },
        {});  // Nothing is supported due to unsupported ShuffleChannels
}

TEST_F(GetSupportedNodesTest, FusedNameReduceL2Test) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 512});
        data->set_friendly_name("data");
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        axes->set_friendly_name("axes");
        auto reduce_l2 = std::make_shared<ov::op::v4::ReduceL2>(data, axes, true);
        reduce_l2->set_friendly_name("reduce_l2");

        m_function = std::make_shared<ov::Model>(ov::NodeVector{reduce_l2}, ov::ParameterVector{data});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::ReduceL2Decomposition>();
            m.register_pass<ov::pass::ConvertReduceToPooling>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            // Pooling is supported, but Sqrt is not
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::op::util::is_constant(op) ||
                   (std::dynamic_pointer_cast<ov::opset1::AvgPool>(op) != nullptr);
        },
        {});  // Check that constant axis is removed from supported
}
