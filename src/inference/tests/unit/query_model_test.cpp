// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <iostream>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using ConfigParams = std::tuple<float, std::unordered_set<std::string>>;

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

class GetSupportedNodesTest : public ::testing::TestWithParam<ConfigParams> {
protected:
    ov::Shape m_shape{1, 84};
    std::shared_ptr<ov::Model> m_function;

public:
    void Run(std::function<void(std::shared_ptr<ov::Model>&)> transform,
             std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported,
             const std::unordered_set<std::string>& expected,
             float query_model_ratio = 1.0f) {
        auto supported = ov::get_supported_nodes(m_function, transform, is_node_supported, query_model_ratio);
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
                   ov::is_type<ov::op::v1::Add>(op);
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
                if (ov::is_type<ov::op::v1::Multiply>(op)) {
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
                   ov::is_type<ov::op::v1::Multiply>(op) || ov::is_type<ov::op::v1::Add>(op);
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
                   ov::is_type<ov::op::v1::Multiply>(op);
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
                   ov::is_type<ov::op::v0::MatMul>(op);
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
            return ov::is_type<ov::op::v1::Add>(op);
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
                   ov::is_type<ov::op::v0::MatMul>(op);
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
                   ov::is_type<ov::op::v1::ReduceMax>(op) || ov::is_type<ov::op::v1::Subtract>(op) ||
                   ov::is_type<ov::op::v1::ReduceSum>(op) || ov::is_type<ov::op::v0::Log>(op);
        },
        {"dummy_param"});  // kepp dummy only since it has no unsupported consumers
}

TEST_F(GetSupportedNodesTest, ShapeOfNonConstantNode) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_shape);
        param->set_friendly_name("input");
        auto slope_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {-2.f});
        slope_compressed->set_friendly_name("slope_compressed");
        auto convert_slope = std::make_shared<ov::op::v0::Convert>(slope_compressed, ov::element::f32);
        convert_slope->set_friendly_name("slope");
        ov::mark_as_decompression(convert_slope);
        auto prelu = std::make_shared<ov::op::v0::PRelu>(param, convert_slope);
        prelu->set_friendly_name("prelu");
        auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(prelu);
        shapeOf->set_friendly_name("shapeof");
        auto convert_fp32 = std::make_shared<ov::op::v0::Convert>(shapeOf, ov::element::f32);
        convert_fp32->set_friendly_name("convert_fp32");
        auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        scale->set_friendly_name("scale");
        auto mul_scale = std::make_shared<ov::op::v1::Multiply>(convert_fp32, scale);
        mul_scale->set_friendly_name("mul_scale");
        auto convert_i64 = std::make_shared<ov::op::v0::Convert>(mul_scale, ov::element::i64);
        convert_i64->set_friendly_name("convert_i64");
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(prelu,
                                                                     convert_i64,
                                                                     scale,
                                                                     ov::op::v4::Interpolate::InterpolateAttrs());
        interpolate->set_friendly_name("interpolate");
        auto interpolate_result = std::make_shared<ov::op::v0::Result>(interpolate);
        interpolate_result->set_friendly_name("interpolate_result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{interpolate_result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::CommonOptimizations>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) || ov::op::util::is_output(op) ||
                   ov::is_type<ov::op::v0::PRelu>(op);
        },
        {"input", "slope_compressed", "slope", "prelu"});  // keep dummy only since it has no unsupported consumers
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
                   ov::is_type<ov::op::v1::AvgPool>(op);
        },
        {});  // Check that constant axis is removed from supported
}

TEST_F(GetSupportedNodesTest, AssignReadValueTest) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 512});
        const ov::op::util::VariableInfo variable_info{ov::Shape{1, 512}, ov::element::f32, "v0"};
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        m_function =
            std::make_shared<ov::Model>(ov::ResultVector{res}, ov::SinkVector{assign}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            // Assign is supported, but ReadValue is not
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::op::util::is_constant(op) ||
                   ov::is_type<ov::op::v6::Assign>(op);
        },
        {});
}

TEST_F(GetSupportedNodesTest, NoSupportedOpsTest) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
        add->set_friendly_name("add");
        auto res = std::make_shared<ov::op::v0::Result>(add);
        res->set_friendly_name("res");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return false;
        },
        {},
        0.9f);
}

TEST_F(GetSupportedNodesTest, NoConstOpTest) {
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 512});
        param1->set_friendly_name("input1");
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 512});
        param2->set_friendly_name("input2");
        auto add = std::make_shared<ov::op::v1::Add>(param1, param2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ov::op::v0::Result>(add);
        res->set_friendly_name("res");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param1, param2});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::is_type<ov::op::v1::Add>(op);
        },
        {"input1", "input2", "add", "res"},
        0.9f);
}

TEST_F(GetSupportedNodesTest, DivideWillRemoveConvertAndConstant) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto constant_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 3, 2, 2}, {1});
        constant_compressed->set_friendly_name("constant_compressed");
        auto convert = std::make_shared<ov::op::v0::Convert>(constant_compressed, ov::element::f32);
        convert->set_friendly_name("convert");
        auto divide = std::make_shared<ov::op::v1::Divide>(param, convert);
        divide->set_friendly_name("divide");
        auto const_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::op::v1::Add>(divide, const_value);
        add->set_friendly_name("add");
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            const bool keep_precision_sensitive_in_fp32_1 = true;
            const bool convert_input_output_precision = false;
            const bool store_original_precision_as_rt_attribute = true;
            type_to_fuse_map empty_fuse_map = {};
            precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
            m.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                        empty_fuse_map,
                                                        keep_precision_sensitive_in_fp32_1,
                                                        convert_input_output_precision,
                                                        store_original_precision_as_rt_attribute);
            m.register_pass<ov::pass::CommonOptimizations>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return true;
        },
        {"input", "constant_compressed", "divide", "const_val", "add", "convert", "result"},
        0.98f);
}

using GetSupportedNodesCommonTest = GetSupportedNodesTest;
using GetSupportedNodesOneConstOp = GetSupportedNodesTest;
using GetSupportedNodesStopSplit = GetSupportedNodesTest;

TEST_P(GetSupportedNodesCommonTest, SplitModelWithDifferentRatioTest) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value1->set_friendly_name("const_val1");
        auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value1);
        add1->set_friendly_name("add1");
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value2->set_friendly_name("const_val2");
        auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value2);
        add2->set_friendly_name("add2");
        auto result = std::make_shared<ov::op::v0::Result>(add2);
        result->set_friendly_name("res");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    float query_model_ratio;
    std::unordered_set<std::string> expected;
    std::tie(query_model_ratio, expected) = this->GetParam();
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::op::util::is_constant(op) ||
                   ov::is_type<ov::op::v1::Add>(op) || ov::is_type<ov::op::v1::Reshape>(op);
        },
        expected,
        query_model_ratio);
}

TEST_P(GetSupportedNodesOneConstOp, OneConstOpTest) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
        add->set_friendly_name("add");
        auto res = std::make_shared<ov::op::v0::Result>(add);
        res->set_friendly_name("res");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    }
    float query_model_ratio;
    std::unordered_set<std::string> expected;
    std::tie(query_model_ratio, expected) = this->GetParam();
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::op::util::is_constant(op) ||
                   ov::is_type<ov::op::v1::Add>(op);
        },
        expected,
        query_model_ratio);
}

TEST_P(GetSupportedNodesStopSplit, StopSplitTest) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value1->set_friendly_name("const_val1");
        auto add = std::make_shared<ov::op::v1::Add>(param, const_value1);
        add->set_friendly_name("add");
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 2, 2}, {1});
        const_value2->set_friendly_name("const_val2");
        auto mul_scale = std::make_shared<ov::op::v1::Multiply>(add, const_value2);
        mul_scale->set_friendly_name("mul_scale");
        auto result = std::make_shared<ov::op::v0::Result>(mul_scale);
        result->set_friendly_name("res");
        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    float query_model_ratio;
    std::unordered_set<std::string> expected;
    std::tie(query_model_ratio, expected) = this->GetParam();
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return ov::op::util::is_parameter(op) || ov::op::util::is_output(op) || ov::is_type<ov::op::v1::Add>(op) ||
                   ov::op::util::is_constant(op);
        },
        expected,
        query_model_ratio);
}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{0.0f, std::unordered_set<std::string>{}},
    ConfigParams{0.5f, std::unordered_set<std::string>{"input", "const_val1", "add1"}},
    ConfigParams{1.0f, std::unordered_set<std::string>{"input", "const_val1", "add1", "const_val2", "add2", "res"}}};

const std::vector<ConfigParams> testConfigs1 = {
    ConfigParams{0.0f, std::unordered_set<std::string>{}},
    ConfigParams{0.5f, std::unordered_set<std::string>{}},
    ConfigParams{1.0f, std::unordered_set<std::string>{"input", "const_val", "add", "res"}}};

const std::vector<ConfigParams> testConfigs2 = {
    ConfigParams{0.0f, std::unordered_set<std::string>{}},
    ConfigParams{0.3f, std::unordered_set<std::string>{}},
    ConfigParams{0.9f, std::unordered_set<std::string>{"input", "const_val1", "add"}},
    ConfigParams{1.0f, std::unordered_set<std::string>{"input", "const_val1", "add"}}};

INSTANTIATE_TEST_SUITE_P(GetSupportedNodesTest, GetSupportedNodesCommonTest, ::testing::ValuesIn(testConfigs));
INSTANTIATE_TEST_SUITE_P(GetSupportedNodesTest, GetSupportedNodesOneConstOp, ::testing::ValuesIn(testConfigs1));
INSTANTIATE_TEST_SUITE_P(GetSupportedNodesTest, GetSupportedNodesStopSplit, ::testing::ValuesIn(testConfigs2));

TEST_F(GetSupportedNodesTest, FilterShapeOf) {
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1});
        param->set_friendly_name("input");
        auto weights = ov::op::v0::Constant::create(ov::element::Type_t::f32, {1, 1}, {1});
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
        auto result = std::make_shared<ov::op::v0::Result>(reshape);
        result->set_friendly_name("result");

        m_function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    Run(
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            return true;
        },
        {"weights"},
        0.5f);
}