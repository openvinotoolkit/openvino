// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/op/deformable_psroi_pooling.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_00)
{
    const float spatial_scale = 0.0625;
    const int64_t group_size = 2;
    const size_t channels_in = 16;
    size_t output_dim = channels_in / (group_size * group_size); // 4

    size_t rois_count = 2;

    auto data_shape = Shape{1, channels_in, 2, 2};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};
    // ASSERT_EQ(def_psroi_pool->get_output_shape(0), (output_shape));

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    // std::fill(data_values.begin(), data_values.end(), 0.1);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        2,
        4,
        6,

        0,
        0,
        3,
        10,
        4,
    };
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.0);

    std::vector<float> expected_output_values{// First ROI
                                            0, 4, 
                                            8, 12, 
                                            
                                            16, 20, 
                                            24, 28, 
                                            
                                            32, 36, 
                                            40, 44, 
                                            
                                            48, 52, 
                                            56, 60,

                                            // Second ROI
                                            0, 4, 
                                            8, 12, 

                                            16, 20, 
                                            24, 28,

                                            32, 36, 
                                            40, 44, 
                                            
                                            48, 52, 
                                            56, 60};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_0p2)
{
    const float spatial_scale = 0.0625;
    const int64_t group_size = 2;
    const size_t channels_in = 16;
    size_t output_dim = channels_in / (group_size * group_size); // 4

    size_t rois_count = 2;

    auto data_shape = Shape{1, channels_in, 2, 2};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        2,
        4,
        6,

        0,
        0,
        3,
        10,
        4,
    };
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.2);

    std::vector<float> expected_output_values{// First ROI
                                            0, 4, 
                                            8, 12, 
                                            
                                            16, 20, 
                                            24, 28, 
                                            
                                            32, 36, 
                                            40, 44, 
                                            
                                            48, 52, 
                                            56, 60,

                                            // Second ROI
                                            0, 4, 
                                            8, 12, 

                                            16, 20, 
                                            24, 28,

                                            32, 36, 
                                            40, 44, 
                                            
                                            48, 52, 
                                            56, 60};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_0p5)
{
    const float spatial_scale = 0.0625;
    const int64_t group_size = 2;
    const size_t channels_in = 16;
    size_t output_dim = channels_in / (group_size * group_size); // 4

    size_t rois_count = 2;

    auto data_shape = Shape{1, channels_in, 2, 2};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        2,
        4,
        6,

        0,
        5,
        3,
        10,
        4,
    };
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.5);

    std::vector<float> expected_output_values{
                                            // First ROI
                                            0, 4, 
                                            8, 12, 
                                            
                                            16, 20, 
                                            24, 28, 
                                            
                                            32, 36, 
                                            40, 44, 
                                            
                                            48, 52, 
                                            56, 60,

                                            // Second ROI
                                            0, 4.1875, 
                                            8, 12.1875, 

                                            16, 20.1875, 
                                            24, 28.1875,

                                            32, 36.1875, 
                                            40, 44.1875, 
                                            
                                            48, 52.1875, 
                                            56, 60.1875};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_roi_oversize)
{
    const float spatial_scale = 0.0625;
    const int64_t group_size = 2;
    const size_t channels_in = 16;
    size_t output_dim = channels_in / (group_size * group_size); // 4

    size_t rois_count = 2;

    auto data_shape = Shape{1, channels_in, 2, 2};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        10,
        10,
        20,
        20,

        0,
        100,
        100,
        200,
        200,
    };
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.0);

    std::vector<float> expected_output_values{
                    0.375, 4.71875, 9.0625, 13.40625,
                    16.375, 20.71875, 25.0625, 29.40625, 
                    32.375, 36.71875, 41.0625, 45.40625, 
                    48.375, 52.71875, 57.0625, 61.40625, 
                    0, 0, 0, 0,
                    0, 0, 0, 0, 
                    0, 0, 0, 0,
                    0, 0, 0, 0};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_no_offset_input)
{
    const float spatial_scale = 1;
    const int64_t group_size = 2;
    const size_t spatial_bins_x = 1;
    const size_t spatial_bins_y = 1;
    const float trans_std = 1.0;
    const int64_t part_size = group_size;

    const size_t batch_in = 1;
    const size_t channels_in = 8;
    const size_t width_in = 3;
    const size_t height_in = 3;

    size_t output_dim = channels_in / (group_size * group_size); // 2

    const auto rois_dim = 1;

    auto data_shape = Shape{batch_in, channels_in, height_in, width_in};
    auto rois_shape = Shape{rois_dim, 5};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);

   auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins_x, spatial_bins_y, trans_std, part_size);

    Shape output_shape{rois_dim, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);

    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        1,
        2,
        2,
    };

    std::vector<float> expected_output_values{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_zero)
{
    const float spatial_scale = 1;
    const int64_t group_size = 2;
    const size_t spatial_bins_x = 1;
    const size_t spatial_bins_y = 1;
    const float trans_std = 1.0;
    const int64_t part_size = group_size;

    const size_t batch_in = 1;
    const size_t channels_in = 8;
    const size_t width_in = 3;
    const size_t height_in = 3;

    size_t output_dim = channels_in / (group_size * group_size); // 2

    const auto rois_dim = 1;

    auto data_shape = Shape{batch_in, channels_in, height_in, width_in};
    auto rois_shape = Shape{rois_dim, 5};
    auto offsets_shape = Shape{rois_dim, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

   auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins_x, spatial_bins_y, trans_std, part_size);

    Shape output_shape{rois_dim, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        1,
        2,
        2,
    };

    
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.0);

    std::vector<float> expected_output_values{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_01)
{
    const float spatial_scale = 1;
    const int64_t group_size = 2;
    const size_t spatial_bins_x = 1;
    const size_t spatial_bins_y = 1;
    const float trans_std = 1.0;
    const int64_t part_size = group_size;

    const size_t batch_in = 1;
    const size_t channels_in = 8;
    const size_t width_in = 3;
    const size_t height_in = 3;

    size_t output_dim = channels_in / (group_size * group_size); // 2

    const auto rois_dim = 1;

    auto data_shape = Shape{batch_in, channels_in, height_in, width_in};
    auto rois_shape = Shape{rois_dim, 5};
    auto offsets_shape = Shape{rois_dim, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

   auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size, 
        "bilinear_deformable", spatial_bins_x, spatial_bins_y, trans_std, part_size);

    Shape output_shape{rois_dim, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        1,
        2,
        2,
    };

    
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.1);

    std::vector<float> expected_output_values{2.8, 12.8, 23.8, 33.8, 38.8, 48.8, 59.8, 69.8};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_offset_05)
{
    const float spatial_scale = 1;
    const int64_t group_size = 2;
    const size_t spatial_bins_x = 1;
    const size_t spatial_bins_y = 1;
    const float trans_std = 1.0;
    const int64_t part_size = group_size;

    const size_t batch_in = 1;
    const size_t channels_in = 8;
    const size_t width_in = 3;
    const size_t height_in = 3;

    size_t output_dim = channels_in / (group_size * group_size); // 2

    const auto rois_dim = 1;

    auto data_shape = Shape{batch_in, channels_in, height_in, width_in};
    auto rois_shape = Shape{rois_dim, 5};
    auto offsets_shape = Shape{rois_dim, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

   auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins_x, spatial_bins_y, trans_std, part_size);

    Shape output_shape{rois_dim, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::iota(data_values.begin(), data_values.end(), 0);
    
    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        1,
        2,
        2,
    };

    
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.5);

    std::vector<float> expected_output_values{6., 15.5, 25.5, 35., 42., 51.5, 61.5, 71.};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_single_value)
{
    const float spatial_scale = 0.0625;
    const int64_t group_size = 2;
    const size_t channels_in = 16;
    size_t output_dim = channels_in / (group_size * group_size); // 4

    size_t rois_count = 1;

    auto data_shape = Shape{1, channels_in, 2, 2};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};

    std::vector<float> data_values(shape_size(data_shape));
    std::fill(data_values.begin(), data_values.end(), 0.1);

    std::vector<float> rois_data{
        // input_batch_id, x1, y1, x2, y2
        0,
        10,
        10,
        10,
        10,
    };

    
    std::vector<float> offsets_values(shape_size(offsets_shape));
    std::fill(offsets_values.begin(), offsets_values.end(), 0.1);

    std::vector<float> expected_output_values{0.1, 0.1, 
                                            0.1, 0.1, 
                                            
                                            0.1, 0.1, 
                                            0.1, 0.1, 
                                            
                                            0.1, 0.1, 
                                            0.1, 0.1, 
                                            
                                            0.1, 0.1, 
                                            0.1, 0.1};

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(data_values);
    test.add_input<float>(rois_data);
    test.add_input<float>(offsets_values);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_psroi_pooling_single_value_big_shape)
{
const int64_t output_dim = 112;
    const float spatial_scale = 0.0625;
    const int64_t group_size = 3;

    size_t rois_count = 2;

    auto data_shape = Shape{1, 1024, 63, 38};
    auto rois_shape = Shape{rois_count, 5};
    auto offsets_shape = Shape{rois_count, 2, group_size, group_size};

    auto data_param = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto rois_param = std::make_shared<op::Parameter>(element::f32, rois_shape);
    auto offsets_param = std::make_shared<op::Parameter>(element::f32, offsets_shape);

    auto def_psroi_pool = std::make_shared<op::v1::DeformablePSROIPooling>(
        data_param, rois_param, offsets_param, output_dim, spatial_scale, group_size);

    Shape output_shape{rois_count, output_dim, group_size, group_size};

    std::vector<float> input_data(shape_size(data_shape));
    std::fill(input_data.begin(), input_data.end(), 0.1);
    
    std::vector<float> input_rois{
        // input_batch_id, x1, y1, x2, y2
        0,
        1,
        2,
        4,
        6,

        0,
        0,
        3,
        10,
        4,
    };

    std::vector<float> input_offsets(shape_size(offsets_shape));
    std::fill(input_offsets.begin(), input_offsets.end(), 0.0);

    std::vector<float> expected_output_values(shape_size(output_shape));
    std::fill(expected_output_values.begin(), expected_output_values.end(), 0.1);

    auto f =
        std::make_shared<Function>(def_psroi_pool,
                                   ParameterVector{data_param, rois_param, offsets_param});

    auto test = test::TestCase<TestEngine>(f);
    test.add_input<float>(input_data);
    test.add_input<float>(input_rois);
    test.add_input<float>(input_offsets);

    test.add_expected_output<float>(output_shape, expected_output_values);
    test.run();
}
