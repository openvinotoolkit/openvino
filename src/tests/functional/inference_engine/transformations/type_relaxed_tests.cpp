// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/test_common.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include <ngraph/pass/manager.hpp>

namespace element = ngraph::element;
using std::make_shared;
using TypeVector = element::TypeVector;

using TypeRelaxedTests = CommonTestUtils::TestsCommon;

TEST_F(TypeRelaxedTests, noOverrideCopyCtor) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        element::Type type(element::Type_t::f32);
        auto param = make_shared<ngraph::opset1::Parameter>(type, shape);
        auto op = ngraph::opset1::Relu(param);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Relu>>(op);
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = make_shared<ngraph::Function>(results, params);

        ASSERT_EQ(element::f32, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::f32, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, overrideOutputCopyCtor) {
    auto input_type = element::f32;
    auto overriden_type = element::i32;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param = make_shared<ngraph::opset1::Parameter>(input_type, shape);
        auto op = ngraph::opset1::Relu(param);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Relu>>(
                op, TypeVector{}, TypeVector{overriden_type});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});

        ASSERT_EQ(input_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, overrideInputCopyCtor) {
    auto input_type = element::f32;
    auto overriden_type = element::i32;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param = make_shared<ngraph::opset1::Parameter>(input_type, shape);
        auto op = ngraph::opset1::Relu(param);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Relu>>(
                op, TypeVector{overriden_type}, TypeVector{});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});

        ASSERT_EQ(input_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(3, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, mixedInputsAutoOutput) {
    auto input_type1 = element::u8;
    auto input_type2 = element::i8;
    auto overriden_type = element::i16;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(input_type1, shape);
        auto param2 = make_shared<ngraph::opset1::Parameter>(input_type2, shape);
        auto op = ngraph::opset1::Add(
                ngraph::op::TemporaryReplaceOutputType(param1->output(0), overriden_type).get(),
                ngraph::op::TemporaryReplaceOutputType(param2->output(0), overriden_type).get());
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
                op, TypeVector{overriden_type, overriden_type}, TypeVector{});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ASSERT_EQ(input_type1, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(input_type2, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, mixedInputsAutoOutputForwardCtor) {
    auto input_type1 = element::u8;
    auto input_type2 = element::i8;
    auto overriden_type = element::i16;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(input_type1, shape);
        auto param2 = make_shared<ngraph::opset1::Parameter>(input_type2, shape);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
                TypeVector{overriden_type, overriden_type}, TypeVector{},
                ngraph::op::TemporaryReplaceOutputType(param1, overriden_type).get(),
                ngraph::op::TemporaryReplaceOutputType(param2, overriden_type).get());
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ASSERT_EQ(input_type1, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(input_type2, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, notSupportedTypeOverride) {
    auto overriden_type = element::u8;
    auto orig_type = element::boolean;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(overriden_type, shape);
        auto param2 = make_shared<ngraph::opset1::Parameter>(overriden_type, shape);
        auto op = ngraph::opset1::LogicalAnd(
                ngraph::op::TemporaryReplaceOutputType(param1, orig_type).get(),
                ngraph::op::TemporaryReplaceOutputType(param2, orig_type).get());
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::LogicalAnd>>(
                op, TypeVector{orig_type, orig_type}, TypeVector{overriden_type});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, notSupportedTypeOverridePartially) {
    auto some_type = element::u8;
    auto overriden_type = element::f32;
    auto orig_type = element::i64;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(some_type, shape);
        auto param2 = make_shared<ngraph::opset1::Parameter>(overriden_type, ngraph::PartialShape{1});
        auto op = ngraph::opset1::Reshape(
                param1,
                ngraph::op::TemporaryReplaceOutputType(param2, orig_type).get(),
                false);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Reshape>>(
                op, TypeVector{element::undefined, orig_type}, TypeVector{});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ASSERT_EQ(some_type, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(overriden_type, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(some_type, relaxed_op->get_output_element_type(0));
    }

    ASSERT_EQ(4, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, multiOutputTypeOverride) {
    auto overriden_type = element::f16;
    auto orig_type = element::f32;
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(orig_type, shape);
        auto op = ngraph::opset1::Split(param1, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1}), 3);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Split>>(
                op, TypeVector{}, TypeVector{overriden_type, overriden_type, overriden_type});
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});

        for (size_t i = 0; i < 3; ++i) {
            ASSERT_EQ(overriden_type, relaxed_op->get_output_element_type(i));
            ASSERT_EQ(ngraph::Shape({1, 1, 22, 22}), relaxed_op->get_output_shape(i));
        }
    }
}

TEST_F(TypeRelaxedTests, setGetTypes) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        auto param1 = make_shared<ngraph::opset1::Parameter>(element::u8, shape);
        auto param2 = make_shared<ngraph::opset1::Parameter>(element::u8, shape);
        // create TypeRelaxed without any type adjustment, the same behaviour as for opset1::Add
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(param1, param2);
        auto result = make_shared<ngraph::opset1::Result>(relaxed_op);

        ngraph = make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

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
        ngraph->validate_nodes_and_infer_types();

        // recheck basic statements about input/output types; they should be the same as we haven't changed anything
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::u8, relaxed_op->get_output_element_type(0));

        // now we are modifying input types and see if the output type reflects this change
        relaxed_op->set_origin_input_type(element::i8, 0);
        relaxed_op->set_origin_input_type(element::i8, 1);
        ngraph->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::i8, relaxed_op->get_output_element_type(0));

        // override output type
        relaxed_op->set_overridden_output_type(element::f32, 0);
        ngraph->validate_nodes_and_infer_types();
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
        ngraph->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::f32, relaxed_op->get_output_element_type(0));
        ASSERT_EQ(2, relaxed_op->get_input_size());
        ASSERT_EQ(1, relaxed_op->get_output_size());

        // lets try to reset types to undefined again and make sure that all original types are restored
        relaxed_op->set_origin_input_type(element::undefined, 0);
        relaxed_op->set_origin_input_type(element::undefined, 1);
        relaxed_op->set_overridden_output_type(element::undefined, 0);
        ngraph->validate_nodes_and_infer_types();
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(0));
        ASSERT_EQ(element::u8, relaxed_op->get_input_element_type(1));
        ASSERT_EQ(element::u8, relaxed_op->get_output_element_type(0));

        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(0));
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(1));
        ASSERT_EQ(element::undefined, relaxed_op->get_origin_input_type(0));
    }

    ASSERT_EQ(4, ngraph->get_ops().size());
}

TEST_F(TypeRelaxedTests, OneOutputMultipleInputPorts) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = make_shared<ngraph::opset1::Parameter>(element::boolean, ngraph::Shape{1, 3, 22, 22});
        auto op = ngraph::opset1::Select(param1, param1, param1);
        auto relaxed_op = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Select>>(
                op, TypeVector{}, TypeVector{element::i64});

        f = make_shared<ngraph::Function>(ngraph::OutputVector{relaxed_op}, ngraph::ParameterVector{param1});

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
    std::shared_ptr<ngraph::Function> f;
    {
        auto const1 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto const2 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto equal = ngraph::opset1::Equal(const1, const2);
        auto relaxed_equal = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Equal>>(equal, TypeVector{}, TypeVector{ element::u8 });

        f = make_shared<ngraph::Function>(ngraph::OutputVector{ relaxed_equal }, ngraph::ParameterVector{});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ngraph::is_type<ngraph::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck1) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto const1 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto const2 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto equal = ngraph::opset1::Equal(const1, const2);
        auto relaxed_equal = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Equal>>(equal, TypeVector{}, TypeVector{ element::boolean });

        f = make_shared<ngraph::Function>(ngraph::OutputVector{ relaxed_equal }, ngraph::ParameterVector{});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ngraph::is_type<ngraph::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck2) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto const1 = ngraph::opset1::Constant::create(element::u8, ngraph::Shape{}, { 2 });
        auto const2 = ngraph::opset1::Constant::create(element::i8, ngraph::Shape{}, { 2 });

        auto original_input_types = TypeVector{ element::i32, element::i32 };
        auto relaxed_equal = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Equal>>(
            ngraph::element::TypeVector{ element::i32, element::i32 },
            ngraph::element::TypeVector{ element::u8 },
            ngraph::op::TemporaryReplaceOutputType(const1, element::i32).get(),
            ngraph::op::TemporaryReplaceOutputType(const2, element::i32).get());

        f = make_shared<ngraph::Function>(ngraph::OutputVector{ relaxed_equal }, ngraph::ParameterVector{});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ngraph::is_type<ngraph::opset1::Constant>(layer_before_result));
    }
}

TEST_F(TypeRelaxedTests, ConstantFoldingCheck3) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto const1 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto const2 = ngraph::opset1::Constant::create(element::i32, ngraph::Shape{}, { 2 });
        auto equal = ngraph::opset1::Equal(const1, const2);

        auto original_input_types = TypeVector{ element::f32, element::f32 };
        auto relaxed_equal = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Equal>>(equal, original_input_types, TypeVector{ element::u8 });

        f = make_shared<ngraph::Function>(ngraph::OutputVector{ relaxed_equal }, ngraph::ParameterVector{});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
        auto layer_before_result = f->get_result()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ngraph::is_type<ngraph::opset1::Constant>(layer_before_result));
    }
}
