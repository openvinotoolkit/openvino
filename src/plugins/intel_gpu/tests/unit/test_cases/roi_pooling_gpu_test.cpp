// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <test_utils/test_utils.h>

#include <intel_gpu/primitives/roi_pooling.hpp>
#include <utility>
#include <vector>

using namespace cldnn;
using namespace tests;

namespace {

template <class T>
std::vector<T> increasinglyFilledBlob(size_t size, bool skip_zero = false, size_t divider = 10) {
    std::vector<T> inputValues;
    T one = 1;
    T start = 0;
    if (skip_zero) {
        start = 1;
    }
    for (size_t i = start; i < size + start; i++) {
        inputValues.push_back(one * i / divider);
    }
    return inputValues;
}

template <class T>
std::vector<T> equallyFilledBlob(size_t size, T value) {
    std::vector<T> inputValues;
    for (size_t i = 0; i < size; i++) {
        inputValues.push_back(value);
    }
    return inputValues;
}

template <class T>
struct roi_pooling_test_inputs {
    roi_pooling_test_inputs() = default;

    roi_pooling_test_inputs(int input_height,
                            int input_width,
                            int channel_count,
                            int roi_count,
                            int output_height,
                            int output_width,
                            float spatial_scale,
                            std::vector<T>&& image_values,
                            std::vector<T>&& proposal_values,
                            std::vector<T>&& output_values,
                            // For position sensitive roi
                            int batch_count = 1,
                            int group_size = 0,
                            int output_dim = 0,
                            int spatial_bins_x = 1,
                            int spatial_bins_y = 1,
                            float trans_std = 0.f,
                            int part_size = 0,
                            bool no_trans = true,
                            float offset = 0,
                            std::string test_name = {})
        : input_height(input_height),
          input_width(input_width),
          channel_count(channel_count),
          roi_count(roi_count),
          output_height(output_height),
          output_width(output_width),
          spatial_scale(spatial_scale),
          image_values(std::move(image_values)),
          proposal_values(std::move(proposal_values)),
          output_values(std::move(output_values)),
          // For position sensitive roi
          batch_count(batch_count),
          group_size(group_size),
          output_dim(output_dim),
          spatial_bins_x(spatial_bins_x),
          spatial_bins_y(spatial_bins_y),
          trans_std(trans_std),
          part_size(part_size),
          no_trans(no_trans),
          offset(offset),
          test_name(std::move(test_name)) {}

    int input_height;
    int input_width;
    int channel_count;
    int roi_count;
    int output_height;
    int output_width;
    float spatial_scale;
    std::vector<T> image_values;
    std::vector<T> proposal_values;
    std::vector<T> output_values;
    // For position sensitive roi
    int batch_count;
    int group_size;
    int output_dim;
    int spatial_bins_x;
    int spatial_bins_y;
    float trans_std;
    int part_size;
    bool no_trans;
    float offset;
    std::string test_name;
};

template <class T>
using roi_pooling_test_params = std::tuple<roi_pooling_test_inputs<T>,
                                           pooling_mode,
                                           bool,  // position sensitive
                                           T,     // threshold
                                           format::type>;

template <class T>
struct roi_pooling_gpu_test : public testing::TestWithParam<roi_pooling_test_params<T>> {
public:
    void test(bool is_caching_test) {
        format::type fmt;
        pooling_mode mode;
        bool position_sensitive;
        T threshold;
        roi_pooling_test_inputs<T> p;
        std::tie(p, mode, position_sensitive, threshold, fmt) =
            testing::TestWithParam<roi_pooling_test_params<T>>::GetParam();

        auto& engine = get_test_engine();
        const auto data_type = ov::element::from<T>();
        const auto plane_format = format::bfyx;

        std::vector<std::pair<primitive_id, memory::ptr>> inputs;

        const layout image_layout(
            data_type,
            plane_format,
            tensor(plane_format, {p.batch_count, p.channel_count, p.input_height, p.input_width}));
        auto image = engine.allocate_memory(image_layout);
        set_values(image, p.image_values);
        inputs.emplace_back("image", image);

        const layout proposal_layout(data_type, plane_format, tensor(plane_format, {p.roi_count, 5, 1, 1}));
        auto proposal = engine.allocate_memory(proposal_layout);
        set_values(proposal, p.proposal_values);
        inputs.emplace_back("proposal", proposal);

        // Set third input
        if (mode == pooling_mode::deformable_bilinear && !p.no_trans) {
            const layout offsets_layout(data_type,
                                        plane_format,
                                        tensor(plane_format, {p.roi_count, 2, p.group_size, p.group_size}));
            auto offsets = engine.allocate_memory(offsets_layout);
            set_values(offsets, equallyFilledBlob(offsets_layout.get_linear_size(), p.offset));
            inputs.emplace_back("offsets", offsets);
        }

        std::vector<input_info> inputs_ids;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputs_ids),
                       [](const decltype(inputs)::value_type& pair) {
                           return input_info("reordered_" + pair.first);
                       });

        topology topology;
        for (auto& input : inputs) {
            topology.add(input_layout(input.first, input.second->get_layout()));
            topology.add(reorder("reordered_" + input.first, input_info(input.first), fmt, ov::element::from<T>()));
        }

        topology.add(roi_pooling("roi_pooling",
                                 inputs_ids,
                                 mode,
                                 position_sensitive,
                                 p.output_width,
                                 p.output_height,
                                 p.spatial_scale,
                                 p.trans_std,
                                 p.no_trans,
                                 p.part_size,
                                 p.group_size,
                                 p.output_dim,
                                 p.spatial_bins_x,
                                 p.spatial_bins_y));

        topology.add(reorder("reordered_roi_pooling", input_info("roi_pooling"), plane_format, ov::element::from<T>()));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        for (auto& input : inputs) {
            network->set_input_data(input.first, input.second);
        }
        const auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "reordered_roi_pooling");

        auto output = outputs.at("reordered_roi_pooling").get_memory();
        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), p.output_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_NEAR(p.output_values[i], output_ptr[i], threshold);
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<roi_pooling_test_params<T>>& info) {
        format::type fmt;
        pooling_mode mode;
        bool position_sensitive;
        roi_pooling_test_inputs<T> p;
        T threshold;
        std::tie(p, mode, position_sensitive, threshold, fmt) = info.param;

        auto mode_str = mode == pooling_mode::max                   ? "max"
                        : mode == pooling_mode::bilinear            ? "bilinear"
                        : mode == pooling_mode::deformable_bilinear ? "deformable_bilinear"
                                                                    : "average";

        std::ostringstream result;
        result << "IS=" << p.input_height << "," << p.input_width << "_";
        result << "OS=" << p.output_height << "," << p.output_width << "_";
        result << "Ch=" << p.channel_count << "_";
        result << "Rois=" << p.roi_count << "_";
        result << "Ss=" << p.spatial_scale << "_";
        result << "Mode=" << mode_str << "_";
        result << "PS=" << position_sensitive << "_";
        result << "Prec=" << ov::element::Type(ov::element::from<T>()) << "_";
        result << "Format=" << fmt_to_str(fmt);
        if (!p.test_name.empty()) {
            result << "_TN=" << p.test_name;
        }
        return result.str();
    }
};

using roi_pooling_gpu_test_float = roi_pooling_gpu_test<float>;

TEST_P(roi_pooling_gpu_test_float, test) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(roi_pooling_gpu_test_float, test_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

const std::vector<roi_pooling_test_inputs<float>> roi_pooling_max_inputs = {
    {
        6,                                                                         // input_height
        6,                                                                         // input_width
        3,                                                                         // channel_count
        3,                                                                         // roi_count
        1,                                                                         // output_height
        1,                                                                         // output_width
        1.f,                                                                       // spatial_scale
        increasinglyFilledBlob<float>(3 * 6 * 6),                                  // image_values
        std::vector<float>{0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3},           // proposal_values
        std::vector<float>{2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f},  // output_values
    },
    {
        6,                                                                                           // input_height
        6,                                                                                           // input_width
        1,                                                                                           // channel_count
        3,                                                                                           // roi_count
        2,                                                                                           // output_height
        2,                                                                                           // output_width
        1.f,                                                                                         // spatial_scale
        increasinglyFilledBlob<float>(1 * 6 * 6),                                                    // image_values
        std::vector<float>{0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5},                             // proposal_values
        std::vector<float>{1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f},  // output_values
    },
};

const std::vector<roi_pooling_test_inputs<float>> roi_pooling_bilinear_inputs = {
    {
        6,                                                                 // input_height
        6,                                                                 // input_width
        3,                                                                 // channel_count
        2,                                                                 // roi_count
        1,                                                                 // output_height
        1,                                                                 // output_width
        1.f,                                                               // spatial_scale
        increasinglyFilledBlob<float>(3 * 6 * 6),                          // image_values
        std::vector<float>{0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6},  // proposal_values
        std::vector<float>{1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f},         // output_values
    },
    {
        8,                                         // input_height
        8,                                         // input_width
        1,                                         // channel_count
        3,                                         // roi_count
        2,                                         // output_height
        2,                                         // output_width
        1.f,                                       // spatial_scale
        increasinglyFilledBlob<float>(1 * 8 * 8),  // image_values
        std::vector<float>{0.f,
                           0.15f,
                           0.2f,
                           0.75f,
                           0.8f,
                           0.f,
                           0.15f,
                           0.2f,
                           0.75f,
                           0.8f,
                           0.f,
                           0.15f,
                           0.2f,
                           0.75f,
                           0.8f},  // proposal_values
        std::vector<float>{1.225f,
                           1.645f,
                           4.585f,
                           5.005f,
                           1.225f,
                           1.645f,
                           4.585f,
                           5.005f,
                           1.225f,
                           1.645f,
                           4.585f,
                           5.005f},  // output_values
    },
    {
        50,                                                             // input_height
        50,                                                             // input_width
        1,                                                              // channel_count
        1,                                                              // roi_count
        4,                                                              // output_height
        4,                                                              // output_width
        1.f,                                                            // spatial_scale
        equallyFilledBlob<float>(1 * 50 * 50, 1),                       // image_values
        std::vector<float>{0.f, 0.f, 0.248046786f, 0.471333951f, 1.f},  // proposal_values
        std::vector<float>(16, 1.f),                                    // output_values
    },
};

const std::vector<roi_pooling_test_inputs<float>> ps_roi_pooling_average_inputs = {
    {
        20,                                                                   // input_height
        20,                                                                   // input_width
        8,                                                                    // channel_count
        3,                                                                    // roi_count
        2,                                                                    // output_height
        2,                                                                    // output_width
        1.f,                                                                  // spatial_scale
        increasinglyFilledBlob<float>(2 * 8 * 20 * 20, true),                 // image_values
        std::vector<float>{0, 1, 2, 4, 6, 1, 0, 3, 10, 4, 0, 10, 7, 11, 13},  // proposal_values
        std::vector<float>{6.2499962, 46.44986,  90.249184, 130.44876, 166.25095, 206.45341,
                           250.25606, 290.45853, 326.36069, 366.86316, 408.36572, 448.86816,
                           486.37045, 526.86841, 568.35828, 608.84839, 18.100033, 58.199684,
                           104.09898, 144.1996,  178.10167, 218.20412, 264.1069,  304.20935},  // output_values
        2,                                                                                     // batch_count
        2,                                                                                     // group_size
        8 / (2 * 2),                                                                           // output_dim
        1,                                                                                     // spatial_bins_x
        1,                                                                                     // spatial_bins_y
    },
    {
        20,                                                                                          // input_height
        20,                                                                                          // input_width
        8,                                                                                           // channel_count
        4,                                                                                           // roi_count
        2,                                                                                           // output_height
        2,                                                                                           // output_width
        0.2f,                                                                                        // spatial_scale
        increasinglyFilledBlob<float>(2 * 8 * 20 * 20, true),                                        // image_values
        std::vector<float>{0, 5, 10, 20, 30, 0, 0, 15, 50, 20, 1, 50, 35, 55, 65, 1, 0, 60, 5, 70},  // proposal_values
        std::vector<float>{6.24999619, 46.399868,  90.2491837, 130.398758, 166.250946, 206.403397, 250.256058,
                           290.408508, 6.34999657, 46.8498573, 87.3492432, 127.848656, 166.350952, 206.853409,
                           247.355896, 287.858368, 338.11142,  378.163879, 424.116669, 464.169128, 498.121185,
                           538.165649, 584.104431, 624.144653, 345.111847, 385.164307, 427.116852, 467.169312,
                           505.121613, 545.16394,  587.103699, 627.143921},  // output_values
        2,                                                                   // batch_count
        2,                                                                   // group_size
        8 / (2 * 2),                                                         // output_dim
        1,                                                                   // spatial_bins_x
        1,                                                                   // spatial_bins_y
    },
};

const std::vector<roi_pooling_test_inputs<float>> ps_roi_pooling_bilinear_inputs = {
    {
        20,                                                     // input_height
        20,                                                     // input_width
        12,                                                     // channel_count
        5,                                                      // roi_count
        3,                                                      // output_height
        3,                                                      // output_width
        1.f,                                                    // spatial_scale
        increasinglyFilledBlob<float>(2 * 12 * 20 * 20, true),  // image_values
        std::vector<float>{0,   0.1, 0.2, 0.7,  0.4, 1,    0.4,  0.1, 0.9, 0.3, 0,   0.5, 0.7,
                           0.7, 0.9, 1,   0.15, 0.3, 0.65, 0.35, 0,   0.0, 0.2, 0.7, 0.8},  // proposal_values
        std::vector<float>{210.71394, 210.99896, 211.28398, 211.98065, 212.26567, 212.55066, 213.24738, 213.53239,
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
                           263.57281, 263.9053},  // output_values
        2,                                        // batch_count
        3,                                        // group_size
        12 / (2 * 3),                             // output_dim
        2,                                        // spatial_bins_x
        3,                                        // spatial_bins_y
    },
    {
        20,                                                     // input_height
        20,                                                     // input_width
        12,                                                     // channel_count
        6,                                                      // roi_count
        4,                                                      // output_height
        4,                                                      // output_width
        0.5f,                                                   // spatial_scale
        increasinglyFilledBlob<float>(2 * 12 * 20 * 20, true),  // image_values
        std::vector<float>{0, 0.1, 0.2, 0.7, 0.4,  0, 0.5, 0.7, 1.2, 1.3, 0, 1.0,  1.3, 1.2,  1.8,
                           1, 0.5, 1.1, 0.7, 1.44, 1, 0.2, 1.1, 0.5, 1.2, 1, 0.34, 1.3, 1.15, 1.35},  // proposal_values
        std::vector<float>{
            205.40955, 205.50456, 205.59955, 205.69453, 205.83179, 205.9268,  206.0218,  206.11681, 206.25403,
            206.34901, 206.44403, 206.53905, 206.67627, 206.77126, 206.86627, 206.96129, 245.41107, 245.50606,
            245.60106, 245.69604, 245.8333,  245.9283,  246.02327, 246.1183,  246.25554, 246.35052, 246.44556,
            246.54054, 246.67778, 246.77277, 246.86775, 246.96278, 217.84717, 217.95801, 218.06885, 218.17969,
            219.11389, 219.22473, 219.33557, 219.44641, 220.3806,  220.49144, 220.60228, 220.71312, 221.64732,
            221.75816, 221.86897, 221.97981, 257.84872, 257.95956, 258.0704,  258.18124, 259.11545, 259.22629,
            259.33713, 259.44797, 260.38217, 260.49301, 260.60385, 260.71469, 261.6489,  261.75974, 261.87057,
            261.98141, 228.9705,  229.00215, 229.03383, 229.06549, 230.02608, 230.05774, 230.08943, 230.12109,
            231.08168, 231.11334, 231.14502, 231.1767,  232.13728, 232.16895, 232.20062, 232.23228, 268.97217,
            269.00385, 269.03549, 269.06717, 270.02777, 270.05945, 270.09109, 270.12277, 271.08337, 271.11502,
            271.1467,  271.17838, 272.13901, 272.17065, 272.2023,  272.23398, 703.65057, 703.68219, 703.71387,
            703.74554, 704.36816, 704.39984, 704.43146, 704.4632,  705.08575, 705.11749, 705.14911, 705.18085,
            705.80347, 705.83514, 705.86676, 705.89844, 743.64136, 743.67291, 743.70459, 743.73633, 744.35889,
            744.39056, 744.42218, 744.45392, 745.07648, 745.10815, 745.13983, 745.17157, 745.79413, 745.82574,
            745.85742, 745.8891,  701.86963, 701.91724, 701.9646,  702.01221, 702.08081, 702.12823, 702.17578,
            702.22321, 702.29181, 702.33936, 702.38678, 702.43433, 702.50293, 702.55035, 702.5979,  702.64545,
            741.86041, 741.90796, 741.95538, 742.00293, 742.07153, 742.11896, 742.1665,  742.21405, 742.28253,
            742.33008, 742.3775,  742.42505, 742.49365, 742.54108, 742.58862, 742.63617, 705.60645, 705.73468,
            705.86298, 705.99115, 705.71198, 705.84027, 705.96844, 706.09668, 705.81757, 705.94574, 706.07397,
            706.20215, 705.9231,  706.05127, 706.1795,  706.3078,  745.59698, 745.72534, 745.85352, 745.98169,
            745.70264, 745.83081, 745.95898, 746.08722, 745.80811, 745.93628, 746.06451, 746.19269, 745.91364,
            746.04181, 746.1701,  746.29834},  // output_values
        2,                                     // batch_count
        4,                                     // group_size
        12 / (2 * 3),                          // output_dim
        2,                                     // spatial_bins_x
        3,                                     // spatial_bins_y
    },
};

const std::vector<roi_pooling_test_inputs<float>> deformable_ps_roi_pooling_inputs = {
    {
        2,                                                        // input_height
        2,                                                        // input_width
        16,                                                       // channel_count
        2,                                                        // roi_count
        2,                                                        // output_height
        2,                                                        // output_width
        0.0625f,                                                  // spatial_scale
        increasinglyFilledBlob<float>(1 * 16 * 2 * 2, false, 1),  // image_values
        std::vector<float>{0, 1, 2, 4, 6, 0, 0, 3, 10, 4},        // proposal_values
        std::vector<float>{                                       // First ROI
                           0,
                           4,
                           8,
                           12,
                           16,
                           20,
                           24,
                           28,
                           32,
                           36,
                           40,
                           44,
                           48,
                           52,
                           56,
                           60,
                           // Second ROI
                           0,
                           4,
                           8,
                           12,
                           16,
                           20,
                           24,
                           28,
                           32,
                           36,
                           40,
                           44,
                           48,
                           52,
                           56,
                           60},             // output_values
        1,                                  // batch_count
        2,                                  // group_size
        16 / (2 * 2) - (16 / (2 * 2)) % 2,  // output_dim
        1,                                  // spatial_bins_x
        1,                                  // spatial_bins_y
        1.f,                                // trans_std
        1,                                  // part_size
        false,                              // no_trans
        0.0f,                               // offset
        "offset_00",                        // test_name
    },
    {
        2,                                                        // input_height
        2,                                                        // input_width
        16,                                                       // channel_count
        2,                                                        // roi_count
        2,                                                        // output_height
        2,                                                        // output_width
        0.0625f,                                                  // spatial_scale
        increasinglyFilledBlob<float>(1 * 16 * 2 * 2, false, 1),  // image_values
        std::vector<float>{0, 1, 2, 4, 6, 0, 0, 3, 10, 4},        // proposal_values
        std::vector<float>{                                       // First ROI
                           0,
                           4,
                           8,
                           12,
                           16,
                           20,
                           24,
                           28,
                           32,
                           36,
                           40,
                           44,
                           48,
                           52,
                           56,
                           60,
                           // Second ROI
                           0,
                           4,
                           8,
                           12,
                           16,
                           20,
                           24,
                           28,
                           32,
                           36,
                           40,
                           44,
                           48,
                           52,
                           56,
                           60},             // output_values
        1,                                  // batch_count
        2,                                  // group_size
        16 / (2 * 2) - (16 / (2 * 2)) % 2,  // output_dim
        1,                                  // spatial_bins_x
        1,                                  // spatial_bins_y
        1.f,                                // trans_std
        1,                                  // part_size
        false,                              // no_trans
        0.2f,                               // offset
        "offset_0p2",                       // test_name
    },
    {
        2,                                                        // input_height
        2,                                                        // input_width
        16,                                                       // channel_count
        2,                                                        // roi_count
        2,                                                        // output_height
        2,                                                        // output_width
        0.0625f,                                                  // spatial_scale
        increasinglyFilledBlob<float>(1 * 16 * 2 * 2, false, 1),  // image_values
        std::vector<float>{0, 1, 2, 4, 6, 0, 0, 3, 10, 4},        // proposal_values
        std::vector<float>{                                       // First ROI
                           0,
                           4,
                           8,
                           12,
                           16,
                           20,
                           24,
                           28,
                           32,
                           36,
                           40,
                           44,
                           48,
                           52,
                           56,
                           60,
                           // Second ROI
                           0,
                           4.1875,
                           8,
                           12.1875,
                           16,
                           20.1875,
                           24,
                           28.1875,
                           32,
                           36.1875,
                           40,
                           44.1875,
                           48,
                           52.1875,
                           56,
                           60.1875},        // output_values
        1,                                  // batch_count
        2,                                  // group_size
        16 / (2 * 2) - (16 / (2 * 2)) % 2,  // output_dim
        1,                                  // spatial_bins_x
        1,                                  // spatial_bins_y
        1.f,                                // trans_std
        1,                                  // part_size
        false,                              // no_trans
        0.5f,                               // offset
        "offset_0p5",                       // test_name
    },
    {
        2,                                                             // input_height
        2,                                                             // input_width
        16,                                                            // channel_count
        2,                                                             // roi_count
        2,                                                             // output_height
        2,                                                             // output_width
        0.0625f,                                                       // spatial_scale
        increasinglyFilledBlob<float>(1 * 16 * 2 * 2, false, 1),       // image_values
        std::vector<float>{0, 10, 10, 20, 20, 0, 100, 100, 200, 200},  // proposal_values
        std::vector<float>{                                            // First ROI
                           0.375,
                           4.71875,
                           9.0625,
                           13.40625,
                           16.375,
                           20.71875,
                           25.0625,
                           29.40625,
                           32.375,
                           36.71875,
                           41.0625,
                           45.40625,
                           48.375,
                           52.71875,
                           57.0625,
                           61.40625,
                           // Second ROI
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0},              // output_values
        1,                                  // batch_count
        2,                                  // group_size
        16 / (2 * 2) - (16 / (2 * 2)) % 2,  // output_dim
        1,                                  // spatial_bins_x
        1,                                  // spatial_bins_y
        1.f,                                // trans_std
        1,                                  // part_size
        false,                              // no_trans
        0.f,                                // offset
        "roi_oversize",                     // test_name
    },
    {
        3,                                                                  // input_height
        3,                                                                  // input_width
        8,                                                                  // channel_count
        1,                                                                  // roi_count
        2,                                                                  // output_height
        2,                                                                  // output_width
        1.f,                                                                // spatial_scale
        increasinglyFilledBlob<float>(1 * 8 * 3 * 3, false, 1),             // image_values
        std::vector<float>{0, 1, 1, 2, 2},                                  // proposal_values
        std::vector<float>{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0},  // output_values
        1,                                                                  // batch_count
        2,                                                                  // group_size
        8 / (2 * 2) - (8 / (2 * 2)) % 2,                                    // output_dim
        1,                                                                  // spatial_bins_x
        1,                                                                  // spatial_bins_y
        1.f,                                                                // trans_std
        2,                                                                  // part_size
        true,                                                               // no_trans
        0.f,                                                                // offset
        "no_offset_input",                                                  // test_name
    },
    {
        3,                                                                  // input_height
        3,                                                                  // input_width
        8,                                                                  // channel_count
        1,                                                                  // roi_count
        2,                                                                  // output_height
        2,                                                                  // output_width
        1.f,                                                                // spatial_scale
        increasinglyFilledBlob<float>(1 * 8 * 3 * 3, false, 1),             // image_values
        std::vector<float>{0, 1, 1, 2, 2},                                  // proposal_values
        std::vector<float>{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0},  // output_values
        1,                                                                  // batch_count
        2,                                                                  // group_size
        8 / (2 * 2) - (8 / (2 * 2)) % 2,                                    // output_dim
        1,                                                                  // spatial_bins_x
        1,                                                                  // spatial_bins_y
        1.f,                                                                // trans_std
        2,                                                                  // part_size
        false,                                                              // no_trans
        0.f,                                                                // offset
        "offset_zero",                                                      // test_name
    },
    {
        3,                                                                  // input_height
        3,                                                                  // input_width
        8,                                                                  // channel_count
        1,                                                                  // roi_count
        2,                                                                  // output_height
        2,                                                                  // output_width
        1.f,                                                                // spatial_scale
        increasinglyFilledBlob<float>(1 * 8 * 3 * 3, false, 1),             // image_values
        std::vector<float>{0, 1, 1, 2, 2},                                  // proposal_values
        std::vector<float>{2.8, 12.8, 23.8, 33.8, 38.8, 48.8, 59.8, 69.8},  // output_values
        1,                                                                  // batch_count
        2,                                                                  // group_size
        8 / (2 * 2) - (8 / (2 * 2)) % 2,                                    // output_dim
        1,                                                                  // spatial_bins_x
        1,                                                                  // spatial_bins_y
        1.f,                                                                // trans_std
        2,                                                                  // part_size
        false,                                                              // no_trans
        0.1f,                                                               // offset
        "offset_01",                                                        // test_name
    },
    {
        3,                                                              // input_height
        3,                                                              // input_width
        8,                                                              // channel_count
        1,                                                              // roi_count
        2,                                                              // output_height
        2,                                                              // output_width
        1.f,                                                            // spatial_scale
        increasinglyFilledBlob<float>(1 * 8 * 3 * 3, false, 1),         // image_values
        std::vector<float>{0, 1, 1, 2, 2},                              // proposal_values
        std::vector<float>{6., 15.5, 25.5, 35., 42., 51.5, 61.5, 71.},  // output_values
        1,                                                              // batch_count
        2,                                                              // group_size
        8 / (2 * 2) - (8 / (2 * 2)) % 2,                                // output_dim
        1,                                                              // spatial_bins_x
        1,                                                              // spatial_bins_y
        1.f,                                                            // trans_std
        2,                                                              // part_size
        false,                                                          // no_trans
        0.5f,                                                           // offset
        "offset_05",                                                    // test_name
    },
    {
        2,                                              // input_height
        2,                                              // input_width
        16,                                             // channel_count
        1,                                              // roi_count
        2,                                              // output_height
        2,                                              // output_width
        0.0625f,                                        // spatial_scale
        equallyFilledBlob<float>(1 * 16 * 2 * 2, 0.1),  // image_values
        std::vector<float>{0, 10, 10, 10, 10},          // proposal_values
        std::vector<
            float>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},  // output_values
        1,                                                                                           // batch_count
        2,                                                                                           // group_size
        16 / (2 * 2) - (16 / (2 * 2)) % 2,                                                           // output_dim
        1,                                                                                           // spatial_bins_x
        1,                                                                                           // spatial_bins_y
        1.f,                                                                                         // trans_std
        1,                                                                                           // part_size
        false,                                                                                       // no_trans
        0.1f,                                                                                        // offset
        "single_value",                                                                              // test_name
    },
    {
        63,                                                               // input_height
        38,                                                               // input_width
        1024,                                                             // channel_count
        2,                                                                // roi_count
        3,                                                                // output_height
        3,                                                                // output_width
        0.0625f,                                                          // spatial_scale
        equallyFilledBlob<float>(1 * 1024 * 63 * 38, 0.1),                // image_values
        std::vector<float>{0, 1, 2, 4, 6, 0, 0, 3, 10, 4},                // proposal_values
        std::vector<float>(2                                              // roi_count
                               * (1024 / (3 * 3) - (1024 / (3 * 3)) % 2)  // output_dim
                               * 3                                        // group_size
                               * 3,                                       // group_size
                           0.1),                                          // output_values
        1,                                                                // batch_count
        3,                                                                // group_size
        1024 / (3 * 3) - (1024 / (3 * 3)) % 2,                            // output_dim
        1,                                                                // spatial_bins_x
        1,                                                                // spatial_bins_y
        1.f,                                                              // trans_std
        1,                                                                // part_size
        false,                                                            // no_trans
        0.f,                                                              // offset
        "single_value_big_shape",                                         // test_name
    },
};

const std::vector<format::type> layout_formats = {format::bfyx,
                                                  format::b_fs_yx_fsv16,
                                                  format::b_fs_yx_fsv32,
                                                  format::bs_fs_yx_bsv16_fsv16,
                                                  format::bs_fs_yx_bsv32_fsv32,
                                                  format::bs_fs_yx_bsv32_fsv16};

INSTANTIATE_TEST_SUITE_P(smoke_roi_pooling_max,
                         roi_pooling_gpu_test_float,
                         testing::Combine(testing::ValuesIn(roi_pooling_max_inputs),
                                          testing::Values(pooling_mode::max),
                                          testing::Values(false),
                                          testing::Values(1e-45f),
                                          testing::ValuesIn(layout_formats)),
                         roi_pooling_gpu_test_float::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_roi_pooling_bilinear,
                         roi_pooling_gpu_test_float,
                         testing::Combine(testing::ValuesIn(roi_pooling_bilinear_inputs),
                                          testing::Values(pooling_mode::bilinear),
                                          testing::Values(false),
                                          testing::Values(1e-6f),
                                          testing::ValuesIn(layout_formats)),
                         roi_pooling_gpu_test_float::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_ps_roi_pooling_average,
                         roi_pooling_gpu_test_float,
                         testing::Combine(testing::ValuesIn(ps_roi_pooling_average_inputs),
                                          testing::Values(pooling_mode::average),
                                          testing::Values(true),
                                          testing::Values(1e-1f),
                                          testing::ValuesIn(layout_formats)),
                         roi_pooling_gpu_test_float::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_ps_roi_pooling_bilinear,
                         roi_pooling_gpu_test_float,
                         testing::Combine(testing::ValuesIn(ps_roi_pooling_bilinear_inputs),
                                          testing::Values(pooling_mode::bilinear),
                                          testing::Values(true),
                                          testing::Values(1e-1f),
                                          testing::ValuesIn(layout_formats)),
                         roi_pooling_gpu_test_float::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_deformable_ps_roi_pooling,
                         roi_pooling_gpu_test_float,
                         testing::Combine(testing::ValuesIn(deformable_ps_roi_pooling_inputs),
                                          testing::Values(pooling_mode::deformable_bilinear),
                                          testing::Values(true),
                                          testing::Values(1e-5f),
                                          testing::ValuesIn(layout_formats)),
                         roi_pooling_gpu_test_float::PrintToStringParamName);

}  // namespace
