//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/op/psroi_pooling.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, psroi_pooling_average)
{
    size_t num_channels = 8;
    size_t group_size = 2;
    size_t output_dim = num_channels / (group_size * group_size);
    size_t num_boxes = 3;
    Shape image_shape{2, num_channels, 20, 20};
    Shape coords_shape{num_boxes, 5};
    auto image = std::make_shared<op::Parameter>(element::Type_t::f32, image_shape);
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, coords_shape);
    auto f =
        std::make_shared<Function>(std::make_shared<op::v0::PSROIPooling>(
                                       image, coords, output_dim, group_size, 1, 1, 1, "average"),
                                   ParameterVector{image, coords});
    Shape output_shape{num_boxes, output_dim, group_size, group_size};

    std::vector<float> image_input(shape_size(image_shape));
    float val = 0;
    std::generate(
        image_input.begin(), image_input.end(), [val]() mutable -> float { return val += 0.1; });
    std::vector<float> coords_input{
        // batch_id, x1, y1, x2, y2
        0,
        1,
        2,
        4,
        6,
        1,
        0,
        3,
        10,
        4,
        0,
        10,
        7,
        11,
        13,
    };
    std::vector<float> output{
        6.2499962, 46.44986,  90.249184, 130.44876, 166.25095, 206.45341, 250.25606, 290.45853,
        326.36069, 366.86316, 408.36572, 448.86816, 486.37045, 526.86841, 568.35828, 608.84839,
        18.100033, 58.199684, 104.09898, 144.1996,  178.10167, 218.20412, 264.1069,  304.20935,

    };

    auto tc = test::TestCase<TestEngine>(f);
    tc.add_input<float>(image_input);
    tc.add_input<float>(coords_input);
    tc.add_expected_output<float>(output_shape, output);
    tc.run();
}

NGRAPH_TEST(${BACKEND_NAME}, psroi_pooling_average_spatial_scale)
{
    size_t num_channels = 8;
    size_t group_size = 2;
    size_t output_dim = num_channels / (group_size * group_size);
    size_t num_boxes = 4;
    float spatial_scale = 0.2;
    Shape image_shape{2, num_channels, 20, 20};
    Shape coords_shape{num_boxes, 5};
    auto image = std::make_shared<op::Parameter>(element::Type_t::f32, image_shape);
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, coords_shape);
    auto f = std::make_shared<Function>(
        std::make_shared<op::v0::PSROIPooling>(
            image, coords, output_dim, group_size, spatial_scale, 1, 1, "average"),
        ParameterVector{image, coords});
    Shape output_shape{num_boxes, output_dim, group_size, group_size};

    std::vector<float> image_input(shape_size(image_shape));
    float val = 0;
    std::generate(
        image_input.begin(), image_input.end(), [val]() mutable -> float { return val += 0.1; });
    std::vector<float> coords_input{
        // batch_id, x1, y1, x2, y2
        0, 5, 10, 20, 30, 0, 0, 15, 50, 20, 1, 50, 35, 55, 65, 1, 0, 60, 5, 70,
    };
    std::vector<float> output{
        6.24999619, 46.399868,  90.2491837, 130.398758, 166.250946, 206.403397, 250.256058,
        290.408508, 6.34999657, 46.8498573, 87.3492432, 127.848656, 166.350952, 206.853409,
        247.355896, 287.858368, 338.11142,  378.163879, 424.116669, 464.169128, 498.121185,
        538.165649, 584.104431, 624.144653, 345.111847, 385.164307, 427.116852, 467.169312,
        505.121613, 545.16394,  587.103699, 627.143921,
    };

    auto tc = test::TestCase<TestEngine>(f);
    tc.add_input<float>(image_input);
    tc.add_input<float>(coords_input);
    tc.add_expected_output<float>(output_shape, output);
    tc.run();
}

NGRAPH_TEST(${BACKEND_NAME}, psroi_pooling_bilinear)
{
    size_t num_channels = 12;
    size_t group_size = 3;
    size_t spatial_bins_x = 2;
    size_t spatial_bins_y = 3;
    size_t output_dim = num_channels / (spatial_bins_x * spatial_bins_y);
    size_t num_boxes = 5;
    Shape image_shape{2, num_channels, 20, 20};
    Shape coords_shape{num_boxes, 5};
    auto image = std::make_shared<op::Parameter>(element::Type_t::f32, image_shape);
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, coords_shape);
    auto f = std::make_shared<Function>(
        std::make_shared<op::v0::PSROIPooling>(
            image, coords, output_dim, group_size, 1, spatial_bins_x, spatial_bins_y, "bilinear"),
        ParameterVector{image, coords});
    Shape output_shape{num_boxes, output_dim, group_size, group_size};

    std::vector<float> image_input(shape_size(image_shape));
    float val = 0;
    std::generate(
        image_input.begin(), image_input.end(), [val]() mutable -> float { return val += 0.1; });
    std::vector<float> coords_input{
        0,   0.1, 0.2, 0.7,  0.4, 1,    0.4,  0.1, 0.9, 0.3, 0,   0.5, 0.7,
        0.7, 0.9, 1,   0.15, 0.3, 0.65, 0.35, 0,   0.0, 0.2, 0.7, 0.8,
    };
    std::vector<float> output{
        210.71394, 210.99896, 211.28398, 211.98065, 212.26567, 212.55066, 213.24738, 213.53239,
        213.8174,  250.71545, 251.00047, 251.28548, 251.98218, 252.2672,  252.5522,  253.2489,
        253.53392, 253.81892, 687.40869, 687.64606, 687.88354, 688.67511, 688.91254, 689.14996,
        689.94147, 690.17896, 690.41644, 727.40021, 727.6377,  727.87518, 728.66669, 728.90405,
        729.14154, 729.93292, 730.17041, 730.4079,  230.28471, 230.3797,  230.47472, 231.55144,
        231.64642, 231.74141, 232.81813, 232.91313, 233.00813, 270.28638, 270.38141, 270.47641,
        271.5531,  271.64813, 271.74313, 272.81985, 272.91486, 273.00986, 692.63281, 692.87018,
        693.1076,  692.94928, 693.18683, 693.42426, 693.26593, 693.50342, 693.74078, 732.62402,
        732.86139, 733.09888, 732.94049, 733.17804, 733.41547, 733.25714, 733.49463, 733.73199,
        215.63843, 215.97093, 216.30345, 219.43855, 219.77106, 220.10358, 223.23871, 223.57123,
        223.90375, 255.63994, 255.97246, 256.30496, 259.44009, 259.77261, 260.10513, 263.2403,
        263.57281, 263.9053,

    };

    auto tc = test::TestCase<TestEngine>(f);
    tc.add_input<float>(image_input);
    tc.add_input<float>(coords_input);
    tc.add_expected_output<float>(output_shape, output);
    tc.run();
}

NGRAPH_TEST(${BACKEND_NAME}, psroi_pooling_bilinear_spatial_scale)
{
    size_t num_channels = 12;
    size_t group_size = 4;
    size_t spatial_bins_x = 2;
    size_t spatial_bins_y = 3;
    size_t output_dim = num_channels / (spatial_bins_x * spatial_bins_y);
    size_t num_boxes = 6;
    float spatial_scale = 0.5;
    Shape image_shape{2, num_channels, 20, 20};
    Shape coords_shape{num_boxes, 5};
    auto image = std::make_shared<op::Parameter>(element::Type_t::f32, image_shape);
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, coords_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::v0::PSROIPooling>(image,
                                                                               coords,
                                                                               output_dim,
                                                                               group_size,
                                                                               spatial_scale,
                                                                               spatial_bins_x,
                                                                               spatial_bins_y,
                                                                               "bilinear"),
                                        ParameterVector{image, coords});
    Shape output_shape{num_boxes, output_dim, group_size, group_size};

    std::vector<float> image_input(shape_size(image_shape));
    float val = 0;
    std::generate(
        image_input.begin(), image_input.end(), [val]() mutable -> float { return val += 0.1; });
    std::vector<float> coords_input{
        0, 0.1, 0.2, 0.7, 0.4,  0, 0.5, 0.7, 1.2, 1.3, 0, 1.0,  1.3, 1.2,  1.8,
        1, 0.5, 1.1, 0.7, 1.44, 1, 0.2, 1.1, 0.5, 1.2, 1, 0.34, 1.3, 1.15, 1.35,
    };
    std::vector<float> output{
        205.40955, 205.50456, 205.59955, 205.69453, 205.83179, 205.9268,  206.0218,  206.11681,
        206.25403, 206.34901, 206.44403, 206.53905, 206.67627, 206.77126, 206.86627, 206.96129,
        245.41107, 245.50606, 245.60106, 245.69604, 245.8333,  245.9283,  246.02327, 246.1183,
        246.25554, 246.35052, 246.44556, 246.54054, 246.67778, 246.77277, 246.86775, 246.96278,
        217.84717, 217.95801, 218.06885, 218.17969, 219.11389, 219.22473, 219.33557, 219.44641,
        220.3806,  220.49144, 220.60228, 220.71312, 221.64732, 221.75816, 221.86897, 221.97981,
        257.84872, 257.95956, 258.0704,  258.18124, 259.11545, 259.22629, 259.33713, 259.44797,
        260.38217, 260.49301, 260.60385, 260.71469, 261.6489,  261.75974, 261.87057, 261.98141,
        228.9705,  229.00215, 229.03383, 229.06549, 230.02608, 230.05774, 230.08943, 230.12109,
        231.08168, 231.11334, 231.14502, 231.1767,  232.13728, 232.16895, 232.20062, 232.23228,
        268.97217, 269.00385, 269.03549, 269.06717, 270.02777, 270.05945, 270.09109, 270.12277,
        271.08337, 271.11502, 271.1467,  271.17838, 272.13901, 272.17065, 272.2023,  272.23398,
        703.65057, 703.68219, 703.71387, 703.74554, 704.36816, 704.39984, 704.43146, 704.4632,
        705.08575, 705.11749, 705.14911, 705.18085, 705.80347, 705.83514, 705.86676, 705.89844,
        743.64136, 743.67291, 743.70459, 743.73633, 744.35889, 744.39056, 744.42218, 744.45392,
        745.07648, 745.10815, 745.13983, 745.17157, 745.79413, 745.82574, 745.85742, 745.8891,
        701.86963, 701.91724, 701.9646,  702.01221, 702.08081, 702.12823, 702.17578, 702.22321,
        702.29181, 702.33936, 702.38678, 702.43433, 702.50293, 702.55035, 702.5979,  702.64545,
        741.86041, 741.90796, 741.95538, 742.00293, 742.07153, 742.11896, 742.1665,  742.21405,
        742.28253, 742.33008, 742.3775,  742.42505, 742.49365, 742.54108, 742.58862, 742.63617,
        705.60645, 705.73468, 705.86298, 705.99115, 705.71198, 705.84027, 705.96844, 706.09668,
        705.81757, 705.94574, 706.07397, 706.20215, 705.9231,  706.05127, 706.1795,  706.3078,
        745.59698, 745.72534, 745.85352, 745.98169, 745.70264, 745.83081, 745.95898, 746.08722,
        745.80811, 745.93628, 746.06451, 746.19269, 745.91364, 746.04181, 746.1701,  746.29834,
    };

    auto tc = test::TestCase<TestEngine>(f);
    tc.add_input<float>(image_input);
    tc.add_input<float>(coords_input);
    tc.add_expected_output<float>(output_shape, output);
    tc.run();
}
