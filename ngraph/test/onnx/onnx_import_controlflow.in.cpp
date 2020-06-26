//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/default_opset.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/type/element_type.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_check_model)
{
    // The model contains a loop which has statically set iterations count equal 3.
    // In the loop body there is just simple add operation.
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add.prototxt"));

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
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_scalars_check_model)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_scalars_add.prototxt"));

    const auto& parameters = function->get_parameters();
    EXPECT_EQ(parameters.size(), 1);
    EXPECT_EQ(parameters.at(0)->get_element_type(), ngraph::element::f32);
    EXPECT_TRUE(parameters.at(0)->get_partial_shape().is_static());
    EXPECT_EQ(parameters.at(0)->get_partial_shape().to_shape(), (Shape{}));

    const auto& results = function->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(0).is_static());
    EXPECT_EQ(function->get_output_shape(0), (Shape{}));
    EXPECT_EQ(function->get_output_element_type(1), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(1).is_static());
    EXPECT_EQ(function->get_output_shape(1), (Shape{3}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_initializer_from_parent_scope)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/loop_2d_add_initializer_from_parent_scope.prototxt"));

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
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_input_from_parent_scope)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add_input_from_parent_scope.prototxt"));

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
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_node_from_parent_scope)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add_node_from_parent_scope.prototxt"));

    const auto& parameters = function->get_parameters();
    EXPECT_EQ(parameters.size(), 1);
    EXPECT_EQ(parameters.at(0)->get_element_type(), ngraph::element::f32);
    EXPECT_TRUE(parameters.at(0)->get_partial_shape().is_static());
    EXPECT_EQ(parameters.at(0)->get_partial_shape().to_shape(), (Shape{1, 2}));

    const auto& results = function->get_results();
    EXPECT_EQ(results.size(), 3);
    EXPECT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(0).is_static());
    EXPECT_EQ(function->get_output_shape(0), (Shape{1, 2}));
    EXPECT_EQ(function->get_output_element_type(1), ngraph::element::f32);
    EXPECT_TRUE(function->get_output_partial_shape(1).is_static());
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_add_value_access_to_body_scope_exception)
{
    try
    {
        const auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/loop_2d_add_incorrect_access_body_scope.prototxt"));
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
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add_the_same_name.prototxt"));

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
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_exception_if_no_identity_cond)
{
    try
    {
        const auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add_no_identity_cond.prototxt"));
        FAIL() << "Not supported termination loop condition exception not thrown";
    }
    catch (const ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Given termination loop condition input is not supported by Loop operator"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_const_cond)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add_const_cond.prototxt"));

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
    EXPECT_EQ(function->get_output_shape(1), (Shape{3, 2}));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_controlflow_loop_2d_add_execution)
{
    // The model contains a loop which has statically set iterations count equal 3.
    // In the loop body there is just simple add operation.
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/loop_2d_add.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}");

    // a_init
    test_case.add_input<float>({0.f, 0.f});

    test_case.add_expected_output<float>(Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}
