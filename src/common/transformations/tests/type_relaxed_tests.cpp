// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/convert_precision.hpp"

namespace element = ov::element;
using std::make_shared;
using TypeVector = element::TypeVector;

using TypeRelaxedTests = ov::test::TestsCommon;

TEST_F(TypeRelaxedTests, noOverrideCopyCtor) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        element::Type type(element::Type_t::f32);
        auto param = make_shared<ov::opset1::Parameter>(type, shape);
        auto op = ov::opset1::Relu(param);
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::Relu>>(op);
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = make_shared<ov::Model>(results, params);

        ASSERT_EQ(element::f32, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::f32, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, overrideOutputCopyCtor) {
    auto input_type = element::f32;
    auto overriden_type = element::i32;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param = make_shared<ov::opset1::Parameter>(input_type, shape);
        auto op = ov::opset1::Relu(param);
        auto relaxed_op =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Relu>>(op, TypeVector{}, TypeVector{overriden_type});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

        ASSERT_EQ(input_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, overrideInputCopyCtor) {
    auto input_type = element::f32;
    auto overriden_type = element::i32;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param = make_shared<ov::opset1::Parameter>(input_type, shape);
        auto op = ov::opset1::Relu(param);
        auto relaxed_op =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Relu>>(op, TypeVector{overriden_type}, TypeVector{});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

        ASSERT_EQ(input_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, mixedInputsAutoOutput) {
    auto input_type1 = element::u8;
    auto input_type2 = element::i8;
    auto overriden_type = element::i16;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(input_type1, shape);
        auto param2 = make_shared<ov::opset1::Parameter>(input_type2, shape);
        auto op = ov::opset1::Add(ov::op::TemporaryReplaceOutputType(param1->output(0), overriden_type).get(),
                                  ov::op::TemporaryReplaceOutputType(param2->output(0), overriden_type).get());
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(op,
                                                                            TypeVector{overriden_type, overriden_type},
                                                                            TypeVector{});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

        ASSERT_EQ(input_type1, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(input_type2, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, mixedInputsAutoOutputForwardCtor) {
    auto input_type1 = element::u8;
    auto input_type2 = element::i8;
    auto overriden_type = element::i16;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(input_type1, shape);
        auto param2 = make_shared<ov::opset1::Parameter>(input_type2, shape);
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(
            TypeVector{overriden_type, overriden_type},
            TypeVector{},
            ov::op::TemporaryReplaceOutputType(param1, overriden_type).get(),
            ov::op::TemporaryReplaceOutputType(param2, overriden_type).get());
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

        ASSERT_EQ(input_type1, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(input_type2, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, notSupportedTypeOverride) {
    auto overriden_type = element::u8;
    auto orig_type = element::boolean;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(overriden_type, shape);
        auto param2 = make_shared<ov::opset1::Parameter>(overriden_type, shape);
        auto op = ov::opset1::LogicalAnd(ov::op::TemporaryReplaceOutputType(param1, orig_type).get(),
                                         ov::op::TemporaryReplaceOutputType(param2, orig_type).get());
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::LogicalAnd>>(op,
                                                                                   TypeVector{orig_type, orig_type},
                                                                                   TypeVector{overriden_type});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, notSupportedTypeOverridePartially) {
    auto some_type = element::u8;
    auto overriden_type = element::f32;
    auto orig_type = element::i64;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(some_type, shape);
        auto param2 = make_shared<ov::opset1::Parameter>(overriden_type, ov::PartialShape{1});
        auto op = ov::opset1::Reshape(param1, ov::op::TemporaryReplaceOutputType(param2, orig_type).get(), false);
        auto relaxed_op =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Reshape>>(op,
                                                                  TypeVector{element::undefined, orig_type},
                                                                  TypeVector{});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

        ASSERT_EQ(some_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(some_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, multiOutputTypeOverride) {
    auto overriden_type = element::f16;
    auto orig_type = element::f32;
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(orig_type, shape);
        auto op = ov::opset1::Split(param1, ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {1}), 3);
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::Split>>(
            op,
            TypeVector{},
            TypeVector{overriden_type, overriden_type, overriden_type});
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1});

        for (size_t i = 0; i < 3; ++i) {
            ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(i));
            ASSERT_EQ(ov::Shape({1, 1, 22, 22}), relaxed_op->get_output_shape(i));
        }
    }
}

TEST_F(TypeRelaxedTests, setGetTypes) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ov::opset1::Parameter>(element::u8, shape);
        auto param2 = make_shared<ov::opset1::Parameter>(element::u8, shape);
        // create TypeRelaxed without any type adjustment, the same behaviour as for opset1::Add
        auto relaxed_op = make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(param1, param2);
        auto result = make_shared<ov::opset1::Result>(relaxed_op);

        model = make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::u8, relaxed_op->get_output_element_type(0));

        // internally set types for opset1::Add inference wasn't set when TypeRelaxed created, check it
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(0));
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(1));
        // if we access elements outside really existing inputs, it should give undefined as well
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(2));
        // number of inputs for the operation node shouldn't change after that
        ASSERT_EQ(2, relaxed_op->get_input_size());

        // similar checks for outputs
        ASSERT_EQ(element::undefined, relaxed_op->get_overridden_output_type(0));
        ASSERT_EQ(element::undefined, relaxed_op->get_overridden_output_type(1));
        ASSERT_EQ(1, relaxed_op->get_output_size());

        // previous checks for input/output indices that are out of number of real inputs/outputs
        // should resize internal vectors that hold orig/overridden types, it may affect
        // inference for the op, so here we check if the inference is still OK:
        model->validate_nodes_and_infer_types();

        // recheck basic statements about input/output types; they should be the same as we haven't changed anything
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::u8, relaxed_op->get_output_element_type(0));

        // now we are modifying input types and see if the output type reflects this change
        relaxed_op->set_origin_input_type(element::i8, 0);
        relaxed_op->set_origin_input_type(element::i8, 1);
        model->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::i8, relaxed_op->get_output_element_type(0));

        // override output type
        relaxed_op->set_overridden_output_type(element::f32, 0);
        model->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::f32, relaxed_op->get_output_element_type(0));

        // check if get methods reflect recent changes after set methods
        ASSERT_EQ(element::i8, relaxed_op->get_origin_input_type(0));
        ASSERT_EQ(element::i8, relaxed_op->get_origin_input_type(1));
        ASSERT_EQ(element::f32, relaxed_op->get_overridden_output_type(0));

        // Now, a more advanced trick: set real orig/overridden type for a not existing input/output
        // it shouldn't affect inference as corresponding inputs/outputs don't exist.
        // This scenario is tested for cases when we want to set new types for operation that will
        // be further modified in the code by adding new inputs (Concat) or outputs (Split) and this code
        // is not aware of TypeRelaxed and shouldn't bother about setting types for new items
        // (a bit hypothetical though).
        relaxed_op->set_origin_input_type(element::i32, 2);
        relaxed_op->set_overridden_output_type(element::i32, 1);
        model->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::f32, relaxed_op->get_output_element_type(0));
        ASSERT_EQ(2, relaxed_op->get_input_size());
        ASSERT_EQ(1, relaxed_op->get_output_size());

        // lets try to reset types to undefined again and make sure that all original types are restored
        relaxed_op->set_origin_input_type(element::undefined, 0);
        relaxed_op->set_origin_input_type(element::undefined, 1);
        relaxed_op->set_overridden_output_type(element::undefined, 0);
        model->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::u8, relaxed_op->get_output_element_type(0));

        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(0));
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(1));
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(0));
    }

    ASSERT_EQ(4, model->get_ops().size());
}

TEST_F(TypeRelaxedTests, OneOutputMultipleInputPorts) {
    std::shared_ptr<ov::Model> f;
    {
        auto param1 = make_shared<ov::opset1::Parameter>(element::boolean, ov::Shape{1, 3, 22, 22});
        auto op = ov::opset1::Select(param1, param1, param1);
        auto relaxed_op =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Select>>(op, TypeVector{}, TypeVector{element::i64});

        f = make_shared<ov::Model>(ov::OutputVector{relaxed_op}, ov::ParameterVector{param1});

        // Prepare relaxed op for input change
        relaxed_op->set_origin_input_type(element::boolean, 0);

        // Change Parameter element type
        param1->set_element_type(element::i64);
        param1->validate_and_infer_types();
        ASSERT_EQ(param1->output(0).get_element_type(), element::i64);

        // Check that after restoring original precisions inside validate_and_infer_types
        // function we do not corrupt original types
        relaxed_op->validate_and_infer_types();
        ASSERT_EQ(param1->output(0).get_element_type(), element::i64);
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck) {
    std::shared_ptr<ov::Model> f;
    {
        auto const1 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto const2 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto equal = ov::opset1::Equal(const1, const2);
        auto relaxed_equal =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Equal>>(equal, TypeVector{}, TypeVector{element::u8});

        f = make_shared<ov::Model>(ov::OutputVector{relaxed_equal}, ov::ParameterVector{});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        OV_ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<ov::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck1) {
    std::shared_ptr<ov::Model> f;
    {
        auto const1 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto const2 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto equal = ov::opset1::Equal(const1, const2);
        auto relaxed_equal =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Equal>>(equal, TypeVector{}, TypeVector{element::boolean});

        f = make_shared<ov::Model>(ov::OutputVector{relaxed_equal}, ov::ParameterVector{});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        OV_ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<ov::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck2) {
    std::shared_ptr<ov::Model> f;
    {
        auto const1 = ov::opset1::Constant::create(element::u8, ov::Shape{}, {2});
        auto const2 = ov::opset1::Constant::create(element::i8, ov::Shape{}, {2});

        auto original_input_types = TypeVector{element::i32, element::i32};
        auto relaxed_equal = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Equal>>(
            ov::element::TypeVector{element::i32, element::i32},
            ov::element::TypeVector{element::u8},
            ov::op::TemporaryReplaceOutputType(const1, element::i32).get(),
            ov::op::TemporaryReplaceOutputType(const2, element::i32).get());

        f = make_shared<ov::Model>(ov::OutputVector{relaxed_equal}, ov::ParameterVector{});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        OV_ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<ov::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck3) {
    std::shared_ptr<ov::Model> f;
    {
        auto const1 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto const2 = ov::opset1::Constant::create(element::i32, ov::Shape{}, {2});
        auto equal = ov::opset1::Equal(const1, const2);

        auto original_input_types = TypeVector{element::f32, element::f32};
        auto relaxed_equal =
            make_shared<ov::op::TypeRelaxed<ov::opset1::Equal>>(equal, original_input_types, TypeVector{element::u8});

        f = make_shared<ov::Model>(ov::OutputVector{relaxed_equal}, ov::ParameterVector{});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        OV_ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<ov::opset1::Constant>(layer_before_result));
    }
}

/* copied from CPU plugin to provide the same experience here */
bool fuse_type_to_convert_cpu(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    const auto& from = node->get_output_element_type(0);
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto convert = ov::as_type_ptr<ov::opset1::Convert>(node)) {
        // For Convert node, converting precision from floating point to boolean will lead to mathematical
        // error, because here the output precision boolean is replaced by u8:
        //  - floating point value 0.01 is converted to be 1 for boolean, but 0 for u8 - need to insert Ceil.
        //  - floating point value 256 is converted to be 1 for boolean, but 0 for u8 - need to insert Min(x, UINT8_MAX)
        //  - floating point value -256 is converted to be 1 for boolean, but 0 for u8 - need to insert Abs before Min.
        // Thus an Abs, Ceil and Min nodes should be added before the Convert node for this scenario.
        if (convert->input(0).get_element_type().is_real() &&
            convert->get_convert_element_type() == ov::element::boolean && to.is_integral_number()) {
            ov::pass::NodeRegistry reg;
            const auto& in_prec = convert->get_input_element_type(0);
            auto data = convert->input_value(0).get_node_shared_ptr();
            auto item = precisions.find(in_prec);
            if (item != precisions.end()) {
                // Add convert node for unsupported precision, such as FP64
                data = reg.make<ov::opset1::Convert>(data, item->second);
            }
            const auto abs = reg.make<ov::opset1::Abs>(data);
            const auto to_max_value = reg.make<ov::opset1::Constant>(ov::util::make_tensor_of_max_value(to));
            const auto to_max_convert = reg.make<ov::opset1::Convert>(to_max_value, abs->get_output_element_type(0));
            const auto min = reg.make<ov::opset1::Minimum>(abs, to_max_convert);
            const auto ceil = reg.make<ov::opset1::Ceiling>(min);
            const auto new_convert = reg.make<ov::opset1::Convert>(ceil, to);
            new_convert->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, reg.get());
            ov::replace_node(convert, new_convert);
            return true;
        } else {
            convert->set_convert_element_type(to);
            return true;
        }
    }
    return false;
}

TEST_F(TypeRelaxedTests, PartialValuePropagation) {
    std::shared_ptr<ov::Model> model;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(element::f32, ov::PartialShape{1, 768, -1, -1});
        auto shape = std::make_shared<ov::opset1::ShapeOf>(parameter);
        auto strided_slice =
            std::make_shared<ov::opset1::StridedSlice>(shape,
                                                       ov::opset1::Constant::create(element::i64, {1}, {0}),
                                                       ov::opset1::Constant::create(element::i64, {1}, {2}),
                                                       ov::opset1::Constant::create(element::i64, {1}, {1}),
                                                       std::vector<int64_t>{0},
                                                       std::vector<int64_t>{0});
        auto concat = std::make_shared<ov::opset1::Concat>(
            ov::OutputVector{strided_slice, ov::opset1::Constant::create(element::i64, {1}, {-1})},
            0);
        auto reshape = std::make_shared<ov::opset1::Reshape>(parameter, concat, false);

        model = make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{parameter});

        precisions_map map = {
            {ov::element::i64, ov::element::i32},
            {ov::element::boolean, ov::element::u8},
        };
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertPrecision>(
            map,
            type_to_fuse_map{{ov::opset1::Convert::get_type_info_static(), fuse_type_to_convert_cpu}});
        OV_ASSERT_NO_THROW(manager.run_passes(model));
        EXPECT_EQ(model->get_result()->get_output_partial_shape(0), ov::PartialShape({1, 768, -1}));
    }
}

TEST_F(TypeRelaxedTests, PartialValuePropagation2) {
    std::shared_ptr<ov::Model> model;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(element::f32, ov::PartialShape{-1, -1});
        auto axis = ov::opset1::Constant::create(element::i64, {1}, {1});
        auto broadcast_input =
            std::make_shared<ov::opset1::Unsqueeze>(std::make_shared<ov::opset1::Unsqueeze>(parameter, axis), axis);

        auto shape = std::make_shared<ov::opset1::ShapeOf>(parameter);
        auto gather_batch = std::make_shared<ov::opset1::Gather>(shape,
                                                                 ov::opset1::Constant::create(element::i64, {1}, {0}),
                                                                 ov::opset1::Constant::create(element::i64, {}, {0}));
        auto gather_sequence_twice =
            std::make_shared<ov::opset1::Gather>(shape,
                                                 ov::opset1::Constant::create(element::i64, {2}, {1, 1}),
                                                 ov::opset1::Constant::create(element::i64, {}, {0}));
        auto concat = std::make_shared<ov::opset1::Concat>(
            ov::OutputVector{gather_batch, ov::opset1::Constant::create(element::i64, {1}, {1}), gather_sequence_twice},
            0);
        auto reshape =
            std::make_shared<ov::opset1::Reshape>(concat, ov::opset1::Constant::create(element::i64, {1}, {-1}), false);
        auto equal = std::make_shared<ov::opset1::Equal>(reshape, ov::opset1::Constant::create(element::i64, {}, {-1}));

        auto select =
            std::make_shared<ov::opset1::Select>(equal, ov::opset1::Constant::create(element::i64, {1}, {1}), reshape);

        auto broadcast = std::make_shared<ov::opset1::Broadcast>(broadcast_input, select);
        model = make_shared<ov::Model>(ov::OutputVector{broadcast}, ov::ParameterVector{parameter});

        precisions_map map = {
            {ov::element::i64, ov::element::i32},
            {ov::element::boolean, ov::element::u8},
        };
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertPrecision>(
            map,
            type_to_fuse_map{{ov::opset1::Convert::get_type_info_static(), fuse_type_to_convert_cpu}});
        OV_ASSERT_NO_THROW(manager.run_passes(model));
        EXPECT_EQ(model->get_result()->get_output_partial_shape(0), ov::PartialShape({-1, 1, -1, -1}));
    }
}
