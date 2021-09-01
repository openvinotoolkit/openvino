// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "openvino/core/pre_post_process/pre_post_process.hpp"
#include "ngraph/ops.hpp"
#include <ngraph/pass/visualize_tree.hpp>

using namespace ov;
using namespace ov::preprocess;

static std::shared_ptr<Function> create_simple_function(element::Type type,
                                                                const PartialShape& shape) {
    auto data1 = std::make_shared<ngraph::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    auto res = std::make_shared<ngraph::op::v0::Result>(data1);
    res->set_friendly_name("Result");
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1});
}

TEST(pre_post_process, simple_scale) {
    auto f = create_simple_function(ngraph::element::f32, ngraph::Shape{1, 3, 4, 5});
    ngraph::pass::VisualizeTree("before.png").run_on_function(f);     // Visualize the nGraph function to an image
    f = PrePostProcessor(f)
            .in()
            .preprocess()
            .scale(2.0f)
            .scale(4.0f)
            .network()
            .set_layout(PartialLayout("NCHW"))
            .build();

    f->validate_nodes_and_infer_types();
    ngraph::pass::VisualizeTree("after.png").run_on_function(f);     // Visualize the nGraph function to an image
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
}

TEST(pre_post_process, convert_element_type_and_scale) {
    auto f = create_simple_function(element::i8, ngraph::Shape{1, 3, 4, 5});
    ASSERT_EQ(f->get_output_element_type(0), element::i8);
    ngraph::pass::VisualizeTree("before2.png").run_on_function(f);     // Visualize the nGraph function to an image
    f = PrePostProcessor(f)
            .in()
            .tensor()
            .set_element_type(element::i16)
            .preprocess()
            .convert_element_type(element::f32)
            .scale(0.2f)
            .network()
            .build();
    f->validate_nodes_and_infer_types();

    ngraph::pass::VisualizeTree("after2.png").run_on_function(f);     // Visualize the nGraph function to an image
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, convert_element_type_from_unknown) {
    auto f = create_simple_function(element::i8, ngraph::Shape{1, 3, 4, 5});
    ASSERT_EQ(f->get_output_element_type(0), element::i8);
    ASSERT_ANY_THROW(
        f = PrePostProcessor(f)
                .in()
                .tensor()
                .preprocess()
                .convert_element_type(element::f32)
                .network()
                .build();
    );
}

TEST(pre_post_process, element_type_and_scale) {
    auto f = create_simple_function(element::i8, ngraph::Shape{1, 3, 4, 5});
    ASSERT_EQ(f->get_output_element_type(0), element::i8);
    ngraph::pass::VisualizeTree("before2.png").run_on_function(f);     // Visualize the nGraph function to an image
    f = PrePostProcessor(f)
            .in()
            .tensor()
            .set_element_type(element::f32)
            .preprocess()
            .scale(2.0f)
            .scale(4.0f)
            .network()
            .set_layout(PartialLayout("NCHW"))
            .build();
    f->validate_nodes_and_infer_types();

    ngraph::pass::VisualizeTree("after2.png").run_on_function(f);     // Visualize the nGraph function to an image
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, scale_vector_network_layout) {
    auto f = create_simple_function(element::f32, ngraph::PartialShape{Dimension::dynamic(), 3, 4, 5});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ngraph::pass::VisualizeTree("before2.png").run_on_function(f);     // Visualize the nGraph function to an image
    f = PrePostProcessor(f)
            .in()
            .preprocess()
            .scale({0.1f, 0.2f, 0.3f})
            .network()
            .set_layout(PartialLayout("NCHW"))
            .build();
    f->validate_nodes_and_infer_types();

    ngraph::pass::VisualizeTree("after3.png").run_on_function(f);     // Visualize the nGraph function to an image
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, scale_vector_tensor_layout) {
    auto f = create_simple_function(element::f32, ngraph::PartialShape{Dimension::dynamic(), 4, 4, 3});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ngraph::pass::VisualizeTree("before2.png").run_on_function(f);     // Visualize the nGraph function to an image
    f = PrePostProcessor(f)
            .in()
            .tensor()
            .set_layout(PartialLayout("NHWC"))
            .preprocess()
            .scale({0.1f, 0.2f, 0.3f})
            .build();
    f->validate_nodes_and_infer_types();

    ngraph::pass::VisualizeTree("after4.png").run_on_function(f);     // Visualize the nGraph function to an image
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
}