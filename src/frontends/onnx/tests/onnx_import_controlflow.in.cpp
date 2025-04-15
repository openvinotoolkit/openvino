// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// ~~~~~~~~TERMINATION CONDITION/TRIP COUNT COMBINATIONS TESTS:~~~~~~~~

// input (trip_count, "") // Note this is analogous to a for loop
//     int trip_count = ...
//     for (int i=0; i < trip_count; ++i) {
//       cond = ...; // ignored
//     }
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add) {
    const auto model = convert_model("controlflow/loop_2d_add.onnx");

    // Shape inference tests
    const auto& parameters = model->get_parameters();
    EXPECT_EQ(parameters.size(), 1);
    EXPECT_EQ(parameters.at(0)->get_element_type(), ov::element::f32);
    EXPECT_TRUE(parameters.at(0)->get_partial_shape().is_static());
    EXPECT_EQ(parameters.at(0)->get_partial_shape().to_shape(), (Shape{1, 2}));

    const auto& results = model->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f32);
    EXPECT_TRUE(model->get_output_partial_shape(0).is_static());
    EXPECT_EQ(model->get_output_shape(0), (Shape{1, 2}));
    EXPECT_EQ(model->get_output_element_type(1), ov::element::f32);
    EXPECT_TRUE(model->get_output_partial_shape(1).is_static());
    EXPECT_EQ(model->get_output_shape(1), (Shape{3, 1, 2}));

    auto test_case = ov::test::TestCase(model, s_device);
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
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond) {
    const auto model = convert_model("controlflow/loop_2d_add_no_identity_termination_cond.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_max_int) {
    const auto model = convert_model("controlflow/loop_2d_add_trip_count_max_int.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond_static_shapes) {
    const auto model = convert_model("controlflow/loop_2d_add_no_identity_termination_cond_static_shapes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // termination condition
    test_case.add_input<bool>({true});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.run();
}

// cond = false
// input ("", cond) // Note this is analogous to a while loop
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_no_identity_termination_cond_false) {
    const auto model = convert_model("controlflow/loop_2d_add_no_identity_termination_cond_false.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_const_no_identity_termination_cond) {
    const auto model = convert_model("controlflow/loop_2d_add_const_no_identity_termination_cond.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {4.f, 4.f});
    test_case.add_expected_output<float>(Shape{4, 1, 2}, {1, 1, 2, 2, 3, 3, 4, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_const_no_identity_termination_cond_static_shapes) {
    const auto model = convert_model("controlflow/loop_2d_add_const_no_identity_termination_cond_static_shapes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_both_cond_and_trip_count_as_inputs) {
    const auto model = convert_model("controlflow/loop_2d_add_cond_and_trip_count_as_inputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // trip count
    test_case.add_input<int64_t>({10});

    // termination condition
    test_case.add_input<bool>({true});

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(Shape{6, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_both_cond_and_trip_count_as_inputs_static_shapes) {
    const auto model = convert_model("controlflow/loop_2d_add_cond_and_trip_count_as_inputs_static_shapes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_initializer_from_parent_scope) {
    const auto model = convert_model("controlflow/loop_2d_add_initializer_from_parent_scope.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {2.f, 2.f, 4.f, 4.f, 6.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_node_from_parent_scope) {
    const auto model = convert_model("controlflow/loop_2d_add_node_from_parent_scope.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {12.f, 12.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {4.f, 4.f, 8.f, 8.f, 12.f, 12.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_node_from_parent_scope_used_in_parent_and_in_body) {
    const auto model = convert_model("controlflow/loop_add_node_from_parent_scope_used_in_parent_and_in_body.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});
    // parent_input
    test_case.add_input<float>({3.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {18.f, 18.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {6.f, 6.f, 12.f, 12.f, 18.f, 18.f});
    test_case.add_expected_output<float>(Shape{1}, {9.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_value_access_to_body_scope_exception) {
    try {
        const auto model = convert_model("controlflow/loop_2d_add_incorrect_access_body_scope.onnx");
        FAIL() << "Incorrect access to body scope not detected";
    } catch (const ov::Exception& e) {
        // patent graph should have no access to subgraph (body Loop) scope
        EXPECT_HAS_SUBSTRING(e.what(), std::string("from_body_scope node not found in graph cache"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_value_the_same_node_from_parent_and_subgraph) {
    const auto model = convert_model("controlflow/loop_2d_add_the_same_name.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_input_from_parent_graph) {
    const auto model = convert_model("controlflow/loop_2d_add_input_from_parent_graph.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});
    // b input
    test_case.add_input<float>({1.f, 1.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_the_proper_opset_in_subgraph) {
    const auto model = convert_model("controlflow/loop_2d_mul_opset1.onnx");

    const auto parent_ops = model->get_ops();
    const auto loop_node_it = std::find_if(parent_ops.begin(), parent_ops.end(), [](const std::shared_ptr<Node>& op) {
        return std::string{op->get_type_name()} == "Loop";
    });
    const auto body_ops = ov::as_type_ptr<op::v5::Loop>(*loop_node_it)->get_function()->get_ops();
    const auto body_mul_node_it = std::find_if(body_ops.begin(), body_ops.end(), [](const std::shared_ptr<Node>& op) {
        return std::string{op->get_type_name()} == "Multiply";
    });
    const auto body_mul_node = ov::as_type_ptr<op::v1::Multiply>(*body_mul_node_it);
    EXPECT_TRUE(body_mul_node);
}

// ~~~~~~~~STATIC/DYNAMIC/CONSTANT INPUTS TESTS:~~~~~~~~

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_scalars) {
    const auto model = convert_model("controlflow/loop_scalars_add.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f});

    test_case.add_expected_output<float>(Shape{}, {3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 2.f, 3.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_const_cond) {
    const auto model = convert_model("controlflow/loop_2d_add_const_cond.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_dynamic) {
    const auto model = convert_model("controlflow/loop_2d_add_trip_count_dynamic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // trip count
    test_case.add_input<int64_t>({3});
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

// ~~~~~~~~SUBGRAPH TYPES INFERENCE:~~~~~~~~
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_infer_types) {
    const auto model = convert_model("controlflow/onnx_controlflow_loop_2d_infer_types.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // trip count
    test_case.add_input<int64_t>({10});

    // termination condition
    test_case.add_input<bool>({true});

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {6.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_node_from_parent_scope_infer_types) {
    const auto model = convert_model("controlflow/loop_add_node_from_parent_scope_infer_types.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_concat_values) {
    const auto model = convert_model("controlflow/loop_concat_values.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // trip_count
    test_case.add_input<int64_t>({5});
    // init condition
    test_case.add_input<bool>({true});
    // seq_init
    test_case.add_input<float>({0});

    // trip_count is concatenated during Loop iterations
    test_case.add_expected_output<int64_t>(Shape{6}, {0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int64_t>(Shape{2 + 3 + 4 + 5 + 6},
                                           {0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5});
    test_case.run();
}

// infinitive loop shape inference
// input ("", ""):
//       for (int i=0; ; ++i) {
//         cond = ... // Note this value is ignored, but is required in the body
//       }
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_trip_count_and_cond_skipped_shape_inference) {
    const auto model = convert_model("controlflow/loop_2d_add_trip_count_and_cond_skipped.onnx");

    const auto& results = model->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f32);
    EXPECT_TRUE(model->get_output_partial_shape(0).is_static());
    EXPECT_EQ(model->get_output_shape(0), (Shape{1, 2}));
    EXPECT_EQ(model->get_output_element_type(1), ov::element::f32);
    EXPECT_TRUE(model->get_output_partial_shape(1).rank().is_static());
    EXPECT_EQ(model->get_output_partial_shape(1).rank(), 3);
    EXPECT_EQ(model->get_output_partial_shape(1), (PartialShape{Dimension::dynamic(), 1, 2}));
}

// infinitive loop execution
OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_infinite) {
    const auto model = convert_model("controlflow/loop_infinite.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_no_variadic_inputs_and_outputs) {
    const auto model = convert_model("controlflow/loop_no_variadic_inputs_and_outputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // trip_count
    test_case.add_input<int64_t>({1});
    // init condition
    test_case.add_input<bool>({true});

    // loop_scan_out
    test_case.add_expected_output<float>(Shape{1}, {1.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_controlflow_loop_power) {
    const auto model = convert_model("controlflow/loop_pow.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_branches_with_same_inputs) {
    /*
       if (condition) {
         add(x, y)
       } else {
         mul(x, y)
       }
    */
    const auto model = convert_model("controlflow/if_branches_with_same_inputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);

    // condition
    test_case.add_input<bool>({true});
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);

    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i + j;
    });
    test_case.add_expected_output<float>(expected);
    test_case.run();

    std::transform(x.begin(), x.end(), y.begin(), expected.begin(), [](float i, float j) -> float {
        return i * j;
    });
    test_case.add_input<bool>({false});
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_branches_with_different_inputs) {
    /*
       if (condition) {
         add(x, y)
       } else {
         abs(y)
       }
    */
    const auto model = convert_model("controlflow/if_branches_with_different_inputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);

    // condition
    test_case.add_input<bool>({true});
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);

    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i + j;
    });
    test_case.add_expected_output<float>(expected);
    test_case.run();

    std::transform(y.begin(), y.end(), expected.begin(), [](float i) -> float {
        return std::fabs(i);
    });
    test_case.add_input<bool>({false});
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_branches_without_inputs) {
    /*
       if (condition) {
         return const tensor {0, 1, 2, 3, 4, 5, 6, 7}
       } else {
         return const tensor {0, 5, 10, 15, 20, 25, 20}
       }
    */
    const auto model = convert_model("controlflow/if_branches_without_inputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // condition
    test_case.add_input<bool>({true});

    test_case.add_expected_output<float>({0, 1, 2, 3, 4, 5, 6, 7});
    test_case.run();

    test_case.add_input<bool>({false});
    test_case.add_expected_output<float>({0, 5, 10, 15, 20, 25, 20, 15});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_inside_if) {
    /*
       if (condition) {
         if (any(x > y) {
           mul(x, y)
         } else {
           add(x, y)
         }
       } else {
         sub(x, y)
       }
    */
    const auto model = convert_model("controlflow/if_inside_if.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // case when condition == true and any(x > y)
    // expected value == x * y
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);
    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i * j;
    });
    test_case.add_input<bool>({true});  // condition
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();

    // case when condition == true and all(x < y)
    // expected value == x + y
    std::iota(x.begin(), x.end(), -static_cast<float>(x.size()));
    std::iota(y.begin(), y.end(), 1.f);
    std::transform(x.begin(), x.end(), y.begin(), expected.begin(), [](float i, float j) -> float {
        return i + j;
    });
    test_case.add_input<bool>({true});  // condition
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();

    // case when condition == false
    // expected value == x - y
    std::transform(x.begin(), x.end(), y.begin(), expected.begin(), [](float i, float j) -> float {
        return i - j;
    });
    test_case.add_input<bool>({false});
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_branches_with_multiple_outputs) {
    /*
       if (condition) {
         split(x, axis=0)
       } else {
         part1, part2, part3 = split(x, axis=1)
         transpose(part1), transpose(part2), transpose(part3)
       }
    */
    const auto model = convert_model("controlflow/if_branches_with_multiple_outputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // case when condition == true so split is along axis 0
    std::vector<float> x(36);
    std::iota(x.begin(), x.end(), 0.f);
    std::vector<float> expected1(12);
    std::iota(expected1.begin(), expected1.end(), 0.f);
    std::vector<float> expected2(12);
    std::iota(expected2.begin(), expected2.end(), 12.f);
    std::vector<float> expected3(12);
    std::iota(expected3.begin(), expected3.end(), 24.f);
    test_case.add_input<bool>({true});  // condition
    test_case.add_input<float>(x);
    test_case.add_expected_output<float>(expected1);
    test_case.add_expected_output<float>(expected2);
    test_case.add_expected_output<float>(expected3);
    test_case.run();

    // case when condition == false so split is along axis 1
    test_case.add_input<bool>({false});  // condition
    test_case.add_input<float>(x);
    test_case.add_expected_output<float>({0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31});
    test_case.add_expected_output<float>({2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33});
    test_case.add_expected_output<float>({4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_inside_loop) {
    /*
        for (i = 0; i < 3; i++) {
            if (i == 0)
                a = a + b
            else
                a = a * b
        }
    */
    const auto model = convert_model("controlflow/if_inside_loop.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {64.f, 64.f});
    test_case.add_expected_output<float>(Shape{3, 1, 2}, {4.f, 4.f, 16.f, 16.f, 64.f, 64.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_with_only_indentity_in_else_branch) {
    /*
       unsq = unsqueeze(input)
       padded = pad(unsq)
       avgpool = avgpool(padded, kernel=[3, 1, 1])
       if_output = if (avgpool.shape[1] == 1) {
         squeeze(avgpool)
       } else {
         identity(avgpool)
       }
       output = add(input, if_output)
    */
    const auto model = convert_model("controlflow/if_with_only_indentity_in_else_branch.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    std::vector<float> x(shape_size(Shape{1, 5, 2, 2}));
    std::iota(x.begin(), x.end(), 0.f);
    std::vector<float> expected{1.333333f, 3.f,  4.666666f, 6.333333f, 8.f,        10.f,     12.f,
                                14.f,      16.f, 18.f,      20.f,      22.f,       24.f,     26.f,
                                28.f,      30.f, 25.33333f, 27.f,      28.666667f, 30.33333f};
    test_case.add_input<float>(x);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_inside_if_inside_loop) {
    /*
        for (i = 0; i < 5; i++) {
            if (i > 2) {
                if (i > 3)
                    out *= float(i * 2)
                else
                    out *= float(i * 3)
            } else {
                out += out
            }
        }
    */

    const auto model = convert_model("controlflow/if_inside_if_inside_loop.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // out_init
    test_case.add_input<float>({1.f});

    test_case.add_expected_output<float>(Shape{1}, {576});
    test_case.add_expected_output<float>(Shape{5, 1}, {2, 4, 8, 72, 576});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_dynamic_inputs) {
    /*
       if (condition) {
         add(x, y)
       } else {
         mul(x, y)
       }
    */
    const auto model = convert_model("controlflow/if_dynamic_inputs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);
    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i + j;
    });

    test_case.add_input<bool>(Shape{}, {true});  // condition
    test_case.add_input<float>(Shape{4, 10}, x);
    test_case.add_input<float>(Shape{4, 10}, y);
    test_case.add_expected_output<float>(Shape{4, 10}, expected);
    test_case.run();

    std::transform(x.begin(), x.end(), y.begin(), expected.begin(), [](float i, float j) -> float {
        return i * j;
    });
    test_case.add_input<bool>(Shape{}, {false});
    test_case.add_input<float>(Shape{4, 10}, x);
    test_case.add_input<float>(Shape{4, 10}, y);
    test_case.add_expected_output<float>(Shape{4, 10}, expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_negative_missing_branches) {
    try {
        const auto model = convert_model("controlflow/if_missing_then_branch.onnx");
        FAIL() << "Model import succeed, but it shouldn't";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Missing 'then_branch' attribute"));
    } catch (...) {
        FAIL() << "Model import failed for unexpected reason";
    }

    try {
        const auto model = convert_model("controlflow/if_missing_else_branch.onnx");
        FAIL() << "Model import succeed, but it shouldn't";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Missing 'else_branch' attribute"));
    } catch (...) {
        FAIL() << "Model import failed for unexpected reason";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_if_negative_mismatch_between_branches_output) {
    try {
        const auto model = convert_model("controlflow/if_negative_mismatch_between_branches_output.onnx");
        FAIL() << "Model import succeed, but it shouldn't";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("'then' and 'else' branches have to have the same number of outputs"));
    } catch (...) {
        FAIL() << "Model import failed for unexpected reason";
    }
}
