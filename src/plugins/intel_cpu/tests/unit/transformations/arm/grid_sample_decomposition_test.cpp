// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov::intel_cpu;

class GridSampleDecompositionTest : public ov::test::TestsCommon {
protected:
    std::shared_ptr<ov::Model> createGridSample(const ov::Shape& data_shape,
                                                const ov::Shape& grid_shape,
                                                const ov::element::Type& data_type,
                                                const ov::element::Type& grid_type,
                                                bool align_corners,
                                                ov::op::v9::GridSample::InterpolationMode interp_mode,
                                                ov::op::v9::GridSample::PaddingMode padding_mode) {
        auto data = std::make_shared<ov::op::v0::Parameter>(data_type, data_shape);
        auto grid = std::make_shared<ov::op::v0::Parameter>(grid_type, grid_shape);
        
        ov::op::v9::GridSample::Attributes attrs;
        attrs.align_corners = align_corners;
        attrs.mode = interp_mode;
        attrs.padding_mode = padding_mode;
        
        auto grid_sample = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
        auto result = std::make_shared<ov::op::v0::Result>(grid_sample);
        
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, grid});
    }

    std::shared_ptr<ov::Model> createDynamicGridSample(const ov::PartialShape& data_shape,
                                                       const ov::PartialShape& grid_shape,
                                                       const ov::element::Type& data_type,
                                                       const ov::element::Type& grid_type,
                                                       bool align_corners,
                                                       ov::op::v9::GridSample::InterpolationMode interp_mode,
                                                       ov::op::v9::GridSample::PaddingMode padding_mode) {
        auto data = std::make_shared<ov::op::v0::Parameter>(data_type, data_shape);
        auto grid = std::make_shared<ov::op::v0::Parameter>(grid_type, grid_shape);
        
        ov::op::v9::GridSample::Attributes attrs;
        attrs.align_corners = align_corners;
        attrs.mode = interp_mode;
        attrs.padding_mode = padding_mode;
        
        auto grid_sample = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
        auto result = std::make_shared<ov::op::v0::Result>(grid_sample);
        
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, grid});
    }

    void checkDecomposition(std::shared_ptr<ov::Model> model, bool should_decompose) {
        ov::pass::Manager manager;
        manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
        manager.run_passes(model);
        
        bool has_grid_sample = false;
        bool has_gather_nd = false;
        bool has_floor = false;
        bool has_clamp = false;
        
        for (auto& op : model->get_ops()) {
            if (ov::is_type<ov::op::v9::GridSample>(op)) {
                has_grid_sample = true;
            }
            if (ov::is_type<ov::op::v8::GatherND>(op)) {
                has_gather_nd = true;
            }
            if (ov::is_type<ov::op::v0::Floor>(op)) {
                has_floor = true;
            }
            if (ov::is_type<ov::op::v0::Clamp>(op)) {
                has_clamp = true;
            }
        }
        
        if (should_decompose) {
            EXPECT_FALSE(has_grid_sample) << "GridSample should be decomposed";
            EXPECT_TRUE(has_gather_nd) << "Should have GatherND operation";
            EXPECT_TRUE(has_floor) << "Should have Floor operation";
            EXPECT_TRUE(has_clamp) << "Should have Clamp operation";
        } else {
            EXPECT_TRUE(has_grid_sample) << "GridSample should not be decomposed";
            EXPECT_FALSE(has_gather_nd) << "Should not have GatherND operation";
        }
    }
};

TEST_F(GridSampleDecompositionTest, BilinearBorderBasic) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2}, 
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, true);
}

TEST_F(GridSampleDecompositionTest, BilinearBorderAlignCorners) {
    auto model = createGridSample({2, 3, 8, 8}, {2, 5, 5, 2},
                                 ov::element::f32, ov::element::f32,
                                 true,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, true);
}

TEST_F(GridSampleDecompositionTest, DifferentDataTypes) {
    // f16 data
    auto model_f16 = createGridSample({1, 1, 4, 4}, {1, 2, 2, 2},
                                     ov::element::f16, ov::element::f32,
                                     false,
                                     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                     ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model_f16, true);
    
    // i32 data (should still decompose)
    auto model_i32 = createGridSample({1, 1, 4, 4}, {1, 2, 2, 2},
                                     ov::element::i32, ov::element::f32,
                                     false,
                                     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                     ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model_i32, true);
}

TEST_F(GridSampleDecompositionTest, LargeDimensions) {
    auto model = createGridSample({4, 16, 64, 64}, {4, 32, 32, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, true);
}

// Test cases where decomposition should NOT be applied
TEST_F(GridSampleDecompositionTest, NotAppliedForNearest) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::NEAREST,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, false);
}

TEST_F(GridSampleDecompositionTest, NotAppliedForBicubic) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BICUBIC,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, false);
}

TEST_F(GridSampleDecompositionTest, NotAppliedForZerosPadding) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::ZEROS);
    checkDecomposition(model, false);
}

TEST_F(GridSampleDecompositionTest, NotAppliedForReflectionPadding) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::REFLECTION);
    checkDecomposition(model, false);
}

TEST_F(GridSampleDecompositionTest, NotAppliedForDynamicShapes) {
    auto model = createDynamicGridSample({ov::Dimension::dynamic(), 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                        {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 2},
                                        ov::element::f32, ov::element::f32,
                                        false,
                                        ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                        ov::op::v9::GridSample::PaddingMode::BORDER);
    checkDecomposition(model, false);
}

TEST_F(GridSampleDecompositionTest, RuntimeInfoPreservation) {
    auto model = createGridSample({1, 2, 4, 4}, {1, 3, 3, 2},
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    
    // Get the original GridSample node and add runtime info
    std::shared_ptr<ov::op::v9::GridSample> grid_sample_node;
    for (auto& op : model->get_ops()) {
        if (auto gs = ov::as_type_ptr<ov::op::v9::GridSample>(op)) {
            grid_sample_node = gs;
            break;
        }
    }
    ASSERT_NE(grid_sample_node, nullptr);
    
    // Mark with decompression attribute
    ov::mark_as_decompression(grid_sample_node);
    auto original_rt_info = grid_sample_node->get_rt_info();
    
    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
    manager.run_passes(model);
    
    // Check that runtime info is preserved on some of the decomposed operations
    bool rt_info_preserved = false;
    for (auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Result>(op) || ov::is_type<ov::op::v0::Parameter>(op)) {
            continue;
        }
        auto& rt_info = op->get_rt_info();
        if (rt_info.count(ov::DisableConstantFolding::get_type_info_static())) {
            rt_info_preserved = true;
            break;
        }
    }
    EXPECT_TRUE(rt_info_preserved) << "Runtime info should be preserved on decomposed operations";
}

TEST_F(GridSampleDecompositionTest, MultipleGridSamples) {
    // Create model with two GridSample operations
    auto data1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4, 4});
    auto grid1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 3, 2});
    auto data2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 6, 6});
    auto grid2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 2});
    
    ov::op::v9::GridSample::Attributes attrs1;
    attrs1.align_corners = false;
    attrs1.mode = ov::op::v9::GridSample::InterpolationMode::BILINEAR;
    attrs1.padding_mode = ov::op::v9::GridSample::PaddingMode::BORDER;
    
    ov::op::v9::GridSample::Attributes attrs2;
    attrs2.align_corners = true;
    attrs2.mode = ov::op::v9::GridSample::InterpolationMode::BILINEAR;
    attrs2.padding_mode = ov::op::v9::GridSample::PaddingMode::BORDER;
    
    auto grid_sample1 = std::make_shared<ov::op::v9::GridSample>(data1, grid1, attrs1);
    auto grid_sample2 = std::make_shared<ov::op::v9::GridSample>(data2, grid2, attrs2);
    
    auto result1 = std::make_shared<ov::op::v0::Result>(grid_sample1);
    auto result2 = std::make_shared<ov::op::v0::Result>(grid_sample2);
    
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, 
                                            ov::ParameterVector{data1, grid1, data2, grid2});
    
    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
    manager.run_passes(model);
    
    // Both GridSample operations should be decomposed
    for (auto& op : model->get_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::v9::GridSample>(op)) 
            << "All GridSample operations should be decomposed";
    }
    
    // Should have multiple GatherND operations (at least 8 - 4 for each GridSample)
    int gather_nd_count = 0;
    for (auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v8::GatherND>(op)) {
            gather_nd_count++;
        }
    }
    EXPECT_GE(gather_nd_count, 8) << "Should have at least 8 GatherND operations";
}

TEST_F(GridSampleDecompositionTest, CorrectOutputShape) {
    const ov::Shape data_shape{2, 3, 8, 10};
    const ov::Shape grid_shape{2, 5, 7, 2};
    
    auto model = createGridSample(data_shape, grid_shape,
                                 ov::element::f32, ov::element::f32,
                                 false,
                                 ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                 ov::op::v9::GridSample::PaddingMode::BORDER);
    
    // Get original output shape
    auto original_output_shape = model->get_output_partial_shape(0);
    
    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
    manager.run_passes(model);
    
    // Check output shape is preserved
    auto decomposed_output_shape = model->get_output_partial_shape(0);
    EXPECT_EQ(original_output_shape, decomposed_output_shape) 
        << "Output shape should be preserved after decomposition";
    
    // Expected shape should be [N, C, H_out, W_out] = [2, 3, 5, 7]
    ov::Shape expected_shape{data_shape[0], data_shape[1], grid_shape[1], grid_shape[2]};
    EXPECT_EQ(decomposed_output_shape.get_shape(), expected_shape)
        << "Output shape should match expected dimensions";
}

// Parametrized tests
namespace {

struct GridSampleTestParams {
    ov::Shape data_shape;
    ov::Shape grid_shape;
    ov::element::Type data_type;
    ov::element::Type grid_type;
    bool align_corners;
    ov::op::v9::GridSample::InterpolationMode interp_mode;
    ov::op::v9::GridSample::PaddingMode padding_mode;
    bool should_decompose;
    std::string test_name;
};

class GridSampleDecompositionParamTest : public ov::test::TestsCommon,
                                        public testing::WithParamInterface<GridSampleTestParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        
        auto data = std::make_shared<ov::op::v0::Parameter>(params.data_type, params.data_shape);
        auto grid = std::make_shared<ov::op::v0::Parameter>(params.grid_type, params.grid_shape);
        
        ov::op::v9::GridSample::Attributes attrs;
        attrs.align_corners = params.align_corners;
        attrs.mode = params.interp_mode;
        attrs.padding_mode = params.padding_mode;
        
        auto grid_sample = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
        auto result = std::make_shared<ov::op::v0::Result>(grid_sample);
        
        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, grid});
    }
    
    std::shared_ptr<ov::Model> model;
};

TEST_P(GridSampleDecompositionParamTest, Decomposition) {
    const auto& params = GetParam();
    
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
    manager.run_passes(model);
    
    bool has_grid_sample = false;
    bool has_gather_nd = false;
    bool has_floor = false;
    bool has_clamp = false;
    
    for (auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v9::GridSample>(op)) {
            has_grid_sample = true;
        }
        if (ov::is_type<ov::op::v8::GatherND>(op)) {
            has_gather_nd = true;
        }
        if (ov::is_type<ov::op::v0::Floor>(op)) {
            has_floor = true;
        }
        if (ov::is_type<ov::op::v0::Clamp>(op)) {
            has_clamp = true;
        }
    }
    
    if (params.should_decompose) {
        EXPECT_FALSE(has_grid_sample) << "GridSample should be decomposed for " << params.test_name;
        EXPECT_TRUE(has_gather_nd) << "Should have GatherND operation for " << params.test_name;
        EXPECT_TRUE(has_floor) << "Should have Floor operation for " << params.test_name;
        EXPECT_TRUE(has_clamp) << "Should have Clamp operation for " << params.test_name;
    } else {
        EXPECT_TRUE(has_grid_sample) << "GridSample should not be decomposed for " << params.test_name;
        EXPECT_FALSE(has_gather_nd) << "Should not have GatherND operation for " << params.test_name;
    }
}

const std::vector<GridSampleTestParams> testParams = {
    // Basic bilinear + border cases (should decompose)
    {{1, 1, 4, 4}, {1, 2, 2, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "basic_bilinear_border"},
    
    {{2, 3, 8, 8}, {2, 4, 4, 2}, ov::element::f32, ov::element::f32, true,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "bilinear_border_align_corners"},
    
    // Different data types (should decompose)
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f16, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "f16_data"},
    
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::i32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "i32_data"},
    
    // Different grid types (should decompose)
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f32, ov::element::f16, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "f16_grid"},
    
    // Large dimensions (should decompose)
    {{4, 16, 64, 64}, {4, 32, 32, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "large_dimensions"},
    
    // Edge cases
    {{1, 1, 1, 1}, {1, 1, 1, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "single_pixel"},
    
    {{10, 3, 224, 224}, {10, 112, 112, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::BORDER,
     true, "typical_cv_size"},
    
    // Cases that should NOT decompose
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::NEAREST, ov::op::v9::GridSample::PaddingMode::BORDER,
     false, "nearest_interpolation"},
    
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BICUBIC, ov::op::v9::GridSample::PaddingMode::BORDER,
     false, "bicubic_interpolation"},
    
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::ZEROS,
     false, "zeros_padding"},
    
    {{1, 2, 4, 4}, {1, 3, 3, 2}, ov::element::f32, ov::element::f32, false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR, ov::op::v9::GridSample::PaddingMode::REFLECTION,
     false, "reflection_padding"},
};

INSTANTIATE_TEST_SUITE_P(GridSampleDecomposition, 
                        GridSampleDecompositionParamTest,
                        ::testing::ValuesIn(testParams),
                        [](const testing::TestParamInfo<GridSampleTestParams>& info) {
                            return info.param.test_name;
                        });

} // namespace