// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ngraph_transformations/propagate_optimal_bs.hpp"

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "mixed_affinity_functions.hpp"

class PropagateOptimalBSTest: public TransformationTestsF{
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::intel_cpu::PropagateOptimalBS>();
    }
};

TEST_F(PropagateOptimalBSTest, ConvWithBias) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithBiasFunction builder({input_shape});

    BSMarkup actual_markup{{"convolution", 1}};
    BSMarkup reference_markup{{"convolution", 1}, {"bias", 1}};
    model = builder.getOriginal(actual_markup);
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, ConvWithBias2) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithBiasFunction builder({input_shape});

    BSMarkup actual_markup{{"convolution", 1}, {"bias", 2}};
    BSMarkup reference_markup{{"convolution", 1}, {"bias", 2}};
    model = builder.getOriginal(actual_markup);
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, ConvWithTranspose) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithTransposeFunction builder({input_shape});

    BSMarkup actual_markup{{"convolution", 1}};
    model = builder.getOriginal(actual_markup);
    BSMarkup reference_markup{{"convolution", 1}, {"transpose", 1}};
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, ConvWithReshapeDynamicShapes) {
    ov::PartialShape input_shape{4, 3, -1, -1};
    ConvWithReshapeFunction builder({input_shape});

    BSMarkup markup{{"convolution", 1}};
    model = builder.getOriginal(markup);
    model_ref = builder.getOriginal(markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, TwoConvAndAddEqualShapes) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape, input_shape});

    BSMarkup actual_markup{{"convolution_1", 1}, {"convolution_2", 1}};
    model = builder.getOriginal(actual_markup);
    BSMarkup reference_markup{
        {"convolution_1", 1},
        {"bias_1", 1},
        {"convolution_2", 1},
        {"bias_2", 1},
        {"add", 1},
    };
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, TwoConvAndAddDifferentBatches) {
    ov::PartialShape input_shape_1{4, 3, 16, 16};
    ov::PartialShape input_shape_2{1, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape_1, input_shape_2});

    BSMarkup actual_markup{{"convolution_1", 2}, {"convolution_2", 1}};
    model = builder.getOriginal(actual_markup);
    BSMarkup reference_markup{
        {"convolution_1", 2},
        {"bias_1", 2},
        {"convolution_2", 1},
        {"bias_2", 1},
        {"add", 2},
    };
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, TwoConvAndAddDifferentBatches2) {
    ov::PartialShape input_shape_1{1, 3, 16, 16};
    ov::PartialShape input_shape_2{4, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape_1, input_shape_2});

    BSMarkup actual_markup{{"convolution_1", 1}, {"convolution_2", 2}};
    model = builder.getOriginal(actual_markup);
    BSMarkup reference_markup{
        {"convolution_1", 1},
        {"bias_1", 1},
        {"convolution_2", 2},
        {"bias_2", 2},
        {"add", 2},
    };
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}

TEST_F(PropagateOptimalBSTest, TwoConvAndAddDifferentBatches3) {
    ov::PartialShape input_shape_1{4, 3, 16, 16};
    ov::PartialShape input_shape_2{4, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape_1, input_shape_2});

    BSMarkup actual_markup{{"convolution_2", 1}};
    model = builder.getOriginal(actual_markup);
    BSMarkup reference_markup{
        {"convolution_2", 1},
        {"bias_2", 1},
        {"add", 1},
    };
    model_ref = builder.getOriginal(reference_markup);
    ov::pass::InitNodeInfo().run_on_model(model_ref);
}