// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/onnx.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

// ~~~~~~~~TERMINATION CONDITION/TRIP COUNT COMBINATIONS TESTS:~~~~~~~~

// input (trip_count, "") // Note this is analogous to a for loop
//     int trip_count = ...
//     for (int i=0; i < trip_count; ++i) {
//       cond = ...; // ignored
//     }
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_add.prototxt"));

    // Shape inference tests
    const auto& parameters = function->get_parameters();
    EXPECT_EQ(parameters.size(), 1);
    EXPECT_EQ(parameters.at(0)->get_element_type(), ngraph::element::f32);
    EXPECT_TRUE(parameters.at(0)->get_partial_shape().is_static());
    EXPECT_EQ(parameters.at(0)->get_partial_shape().to_shape(), (Shape{1, 2}));

    const auto& results = function->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(0).is_static());
    EXPECT_EQ(function->get_output_shape(0), (Shape{1, 2}));
    EXPECT_EQ(function->get_output_element_type(1), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(1).is_static());
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 1, 2}));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

// input ("", cond) // Note this is analogous to a while loop
//     bool cond = ...;
//     for (int i=0; cond; ++i) {
//       cond = ...;
//     }
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_no_identity_termination_cond.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(
        Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_max_int)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_add_trip_count_max_int.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(
        Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond_static_shapes)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/loop_2d_add_no_identity_termination_cond_static_shapes.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.run();
}

// cond = false
// input ("", cond) // Note this is analogous to a while loop
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond_false)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_no_identity_termination_cond_false.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({3.f, 4.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 4.f});
    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 4.f});
    test_case.run();
}

//  input ("", 1) // Note this is analogous to a do-while loop
//      bool cond = true
//      for (int i=0; cond; ++i) {
//        cond = ...;
//      }
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_const_no_identity_termination_cond)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_const_no_identity_termination_cond.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {4.f, 4.f});
    test_case.add_expected_output<float>(Shape{4, 1, 2}, {1, 1, 2, 2, 3, 3, 4, 4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME},
            onnx_controlflow_loop_2d_const_no_identity_termination_cond_static_shapes)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/loop_2d_add_const_no_identity_termination_cond_static_shapes.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {4.f, 4.f});
    test_case.run();
}

//  input (trip_count, cond)
//      int trip_count = ...;
//      bool cond = ...;
//      for (int i=0; i < trip_count && cond; ++i) {
//        cond = ...;
//      }
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_both_cond_and_trip_count_as_inputs)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_cond_and_trip_count_as_inputs.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // trip count
    test_case.add_input<int64_t>({10});

    // termination condition
    test_case.add_input<bool>({true});

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(
        Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME},
            onnx_controlflow_loop_2d_both_cond_and_trip_count_as_inputs_static_shapes)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/loop_2d_add_cond_and_trip_count_as_inputs_static_shapes.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // trip count
    test_case.add_input<int64_t>({10});

    // termination condition
    test_case.add_input<bool>({true});

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.run();
}

// ~~~~~~~~SCOPES VISIBILITY TESTS:~~~~~~~~

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_initializer_from_parent_scope)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_initializer_from_parent_scope.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {2.f, 2.f, 4.f, 4.f, 6.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_node_from_parent_scope)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_node_from_parent_scope.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {12.f, 12.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {4.f, 4.f, 8.f, 8.f, 12.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME},
            onnx_controlflow_loop_add_node_from_parent_scope_used_in_parent_and_in_body)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/loop_add_node_from_parent_scope_used_in_parent_and_in_body.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});
    // parent_input
    test_case.add_input<float>({3.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {18.f, 18.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {6.f, 6.f, 12.f, 12.f, 18.f, 18.f});
    test_case.add_expected_output<float>(Shape{1}, {9.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_value_access_to_body_scope_exception)
{
    try
    {
        const auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/loop/loop_2d_add_incorrect_access_body_scope.prototxt"));
        FAIL() << "Incorrect access to body scope not detected";
    }
    catch (const ngraph_error& e)
    {
        // patent graph should have no access to subgraph (body Loop) scope
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("from_body_scope node not found in graph cache"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_value_the_same_node_from_parent_and_subgraph)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_add_the_same_name.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_input_from_parent_graph)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_input_from_parent_graph.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});
    // b input
    test_case.add_input<float>({1.f, 1.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_the_proper_opset_in_subgraph)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_mul_opset1.prototxt"));

    const auto parent_ops = function->get_ops();
    const auto loop_node_it =
        std::find_if(parent_ops.begin(), parent_ops.end(), [](const std::shared_ptr<Node>& op) {
            return std::string{op->get_type_name()} == "Loop";
        });
    const auto body_ops =
        ngraph::as_type_ptr<default_opset::Loop>(*loop_node_it)->get_function()->get_ops();
    const auto body_mul_node_it =
        std::find_if(body_ops.begin(), body_ops.end(), [](const std::shared_ptr<Node>& op) {
            return std::string{op->get_type_name()} == "Multiply";
        });
    const auto body_mul_node = ngraph::as_type_ptr<default_opset::Multiply>(*body_mul_node_it);
    EXPECT_TRUE(body_mul_node);
}

// ~~~~~~~~STATIC/DYNAMIC/CONSTANT INPUTS TESTS:~~~~~~~~

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_scalars)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_scalars_add.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f});

    test_case.add_expected_output<float>(Shape{}, {3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 2.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_const_cond)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_add_const_cond.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_dynamic)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_2d_add_trip_count_dynamic.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // trip count
    test_case.add_input<int64_t>({3});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

// ~~~~~~~~SUBGRAPH TYPES INFERENCE:~~~~~~~~
NGRAPH_TEST(${BACKEND_NAME},
            onnx_controlflow_loop_2d_infer_types)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/onnx_controlflow_loop_2d_infer_types.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // trip count
    test_case.add_input<int64_t>({10});

    // termination condition
    test_case.add_input<bool>({true});

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME},
            onnx_controlflow_loop_add_node_from_parent_scope_infer_types)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/loop/loop_add_node_from_parent_scope_infer_types.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // a_init
    test_case.add_input<float>({0.f, 0.f});
    // parent_input
    test_case.add_input<float>({3.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {18.f, 18.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {6.f, 6.f, 12.f, 12.f, 18.f, 18.f});
    test_case.add_expected_output<float>(Shape{1}, {9.f});
    test_case.run();
}


// ~~~~~~~~ADDITIONAL TESTS:~~~~~~~~

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_concat_values)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_concat_values.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // trip_count
    test_case.add_input<int64_t>({5});
    // init condition
    test_case.add_input<bool>({true});
    // seq_init
    test_case.add_input<float>({0});

    // trip_count is concatenated during Loop iterations
    test_case.add_expected_output<int64_t>(Shape{6}, {0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int64_t>(
        Shape{2 + 3 + 4 + 5 + 6}, {0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5});
    test_case.run();
}

// infinitive loop shape inference
// input ("", ""):
//       for (int i=0; ; ++i) {
//         cond = ... // Note this value is ignored, but is required in the body
//       }
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_and_cond_skipped_shape_inference)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_2d_add_trip_count_and_cond_skipped.prototxt"));

    const auto& results = function->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(0).is_static());
    EXPECT_EQ(function->get_output_shape(0), (Shape{1, 2}));
    EXPECT_EQ(function->get_output_element_type(1), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(1).rank().is_static());
    EXPECT_EQ(function->get_output_partial_shape(1).rank(), 3);
    EXPECT_EQ(function->get_output_partial_shape(1), (PartialShape{Dimension::dynamic(), 1, 2}));
}

// infinitive loop execution
NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_infinite)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_infinite.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // trip_count
    test_case.add_input<int64_t>({std::numeric_limits<int64_t>::max()});
    // init condition
    test_case.add_input<bool>({true});
    // fake
    test_case.add_input<float>({0.f});
    // outer_scope
    test_case.add_input<float>({3.f});

    // final value not changed
    test_case.add_expected_output<float>(Shape{1}, {0.f});
    // outer_scope passed as scan output
    test_case.add_expected_output<float>(Shape{1}, {3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_no_variadic_inputs_and_outputs)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop/loop_no_variadic_inputs_and_outputs.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // trip_count
    test_case.add_input<int64_t>({1});
    // init condition
    test_case.add_input<bool>({true});

    // loop_scan_out
    test_case.add_expected_output<float>(Shape{1}, {1.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_power)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop/loop_pow.prototxt"));

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    // trip_count
    test_case.add_input<int64_t>({5});
    // pow init
    test_case.add_input<int64_t>({5});

    // pow_final
    test_case.add_expected_output<int64_t>(Shape{1}, {16});
    // pow_scans
    test_case.add_expected_output<int64_t>(Shape{5}, {0, 1, 4, 9, 16});
    test_case.run();
}
