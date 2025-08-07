// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/prior_box.hpp>
#include <random>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {
struct prior_box_attributes {
    std::vector<float> min_sizes;            // Desired min_size of prior boxes
    std::vector<float> max_sizes;            // Desired max_size of prior boxes
    std::vector<float> aspect_ratios;        // Aspect ratios of prior boxes
    std::vector<float> densities;            // This is the square root of the number of boxes of each type
    std::vector<float> fixed_ratios;         // This is an aspect ratio of a box
    std::vector<float> fixed_sizes;          // This is an initial box size (in pixels)
    bool clip;                               // Clip output to [0,1]
    bool flip;                               // Flip aspect ratios
    float step;                              // Distance between prior box centers
    float offset;                            // Box offset relative to top center of image
    std::vector<float> variances;            // Values to adjust prior boxes with
    bool scale_all_sizes;                    // Scale all sizes
    bool min_max_aspect_ratios_order;        // Order of output prior box
};

template <class InputType, class OutputType>
using prior_box_param = std::tuple<format,                       // Input and output format
                                   std::vector<InputType>,       // output_size
                                   std::vector<InputType>,       // image_size
                                   prior_box_attributes,         // attributes
                                   std::vector<OutputType>>;     // expected values

template <class InputType, class OutputType>
class PriorBoxGPUTest : public ::testing::TestWithParam<prior_box_param<InputType, OutputType>> {
public:
    void execute(bool is_caching_test) {
        const auto input_data_type = ov::element::from<InputType>();
        const auto output_data_type = ov::element::from<OutputType>();
        const auto plain_format = format::bfyx;

        auto &engine = get_test_engine();
        const auto& [target_format, output_size, image_size, attrs, expected_values] = this->GetParam();

        auto layout_output_size_input = layout{input_data_type, plain_format, tensor{2}};
        auto layout_image_size_input = layout{input_data_type, plain_format, tensor{2}};
        const auto output_size_input = engine.allocate_memory(layout_output_size_input);
        const auto image_size_input = engine.allocate_memory(layout_image_size_input);

        const cldnn::tensor output_size_tensor{cldnn::spatial(output_size[0], output_size[1])};
        const cldnn::tensor img_size_tensor{cldnn::spatial(image_size[0], image_size[1])};

        topology topo;
        topo.add(input_layout("output_size", layout_output_size_input));
        topo.add(reorder("reordered_output_size", input_info("output_size"), target_format, input_data_type));
        topo.add(input_layout("image_size", layout_image_size_input));
        topo.add(reorder("reordered_image_size", input_info("image_size"), target_format, input_data_type));

        set_values<InputType>(output_size_input, output_size);
        set_values<InputType>(image_size_input, image_size);

        std::vector<input_info> inputs{ input_info("reordered_output_size"), input_info("reordered_image_size")};
        const auto prior_box = cldnn::prior_box("blocked_prior_box",
                                                inputs,
                                                output_size_tensor,
                                                img_size_tensor,
                                                attrs.min_sizes,
                                                attrs.max_sizes,
                                                attrs.aspect_ratios,
                                                attrs.flip,
                                                attrs.clip,
                                                attrs.variances,
                                                attrs.step,
                                                attrs.offset,
                                                attrs.scale_all_sizes,
                                                attrs.fixed_ratios,
                                                attrs.fixed_sizes,
                                                attrs.densities,
                                                true,
                                                attrs.min_max_aspect_ratios_order);
        topo.add(prior_box);
        topo.add(reorder("prior_box", input_info("blocked_prior_box"), plain_format, output_data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr network = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("output_size", output_size_input);
        network->set_input_data("image_size", image_size_input);

        const auto outputs = network->execute();
        const auto output = outputs.at("prior_box").get_memory();

        cldnn::mem_lock<OutputType> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], expected_values[i], 2e-3)
                << "target_format=" << fmt_to_str(target_format) << ", i=" << i;
        }
    }
};

using prior_box_test_i32_f32 = PriorBoxGPUTest<int32_t, float>;
TEST_P(prior_box_test_i32_f32, prior_box_test_i32_f32) {
    this->execute(false);
}

INSTANTIATE_TEST_SUITE_P(
        prior_box_test_all_formats,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::ValuesIn(
        std::vector<format>{format::bfyx,
                            format::b_fs_yx_fsv16,
                            format::b_fs_yx_fsv32,
                            format::bs_fs_yx_bsv16_fsv16,
                            format::bs_fs_yx_bsv32_fsv16,
                            format::bs_fs_yx_bsv32_fsv32}),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 0.0f, 0.0f, {}, true, false}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.127526, 0.16835, 0.372474, 0.33165, 0.0918861, 0.0918861, 0.408114, 0.408114,
            0.65, 0.15, 0.85, 0.35, 0.627526, 0.16835, 0.872474, 0.33165, 0.591886,  0.0918861, 0.908114, 0.408114,
            0.15, 0.65, 0.35, 0.85, 0.127526, 0.66835, 0.372474, 0.83165, 0.0918861, 0.591886,  0.408114, 0.908114,
            0.65, 0.65, 0.85, 0.85, 0.627526, 0.66835, 0.872474, 0.83165, 0.591886,  0.591886,  0.908114, 0.908114,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1
        })));

INSTANTIATE_TEST_SUITE_P(
        prior_box_test_clip_flip,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, true, true, 0.0f, 0.0f, {}, true, false}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.127526, 0.16835, 0.372474, 0.33165, 0.16835, 0.127526, 0.33165, 0.372474,
            0.0918861, 0.0918861, 0.408114, 0.408114,
            0.65, 0.15, 0.85, 0.35, 0.627526, 0.16835, 0.872474, 0.33165, 0.66835, 0.127526, 0.83165, 0.372474,
            0.591886, 0.0918861, 0.908114, 0.408114,
            0.15, 0.65, 0.35, 0.85, 0.127526, 0.66835, 0.372474, 0.83165, 0.16835, 0.627526, 0.33165, 0.872474,
            0.0918861, 0.591886, 0.408114, 0.908114,
            0.65, 0.65, 0.85, 0.85, 0.627526, 0.66835, 0.872474, 0.83165, 0.66835, 0.627526, 0.83165, 0.872474,
            0.591886, 0.591886, 0.908114, 0.908114,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        })));

INSTANTIATE_TEST_SUITE_P(
        prior_box_test_minmax_aspect_ratio,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, true, true, 0.0f, 0.0f, {}, true, true}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.0918861, 0.0918861, 0.408114, 0.408114, 0.127526, 0.16835, 0.372474, 0.33165,
            0.16835, 0.127526, 0.33165, 0.372474,
            0.65, 0.15, 0.85, 0.35, 0.591886, 0.0918861, 0.908114, 0.408114, 0.627526, 0.16835, 0.872474, 0.33165,
            0.66835, 0.127526, 0.83165, 0.372474,
            0.15, 0.65, 0.35, 0.85, 0.0918861, 0.591886, 0.408114, 0.908114, 0.127526, 0.66835, 0.372474, 0.83165,
            0.16835, 0.627526, 0.33165, 0.872474,
            0.65, 0.65, 0.85, 0.85, 0.591886, 0.591886, 0.908114, 0.908114, 0.627526, 0.66835, 0.872474, 0.83165,
            0.66835, 0.627526, 0.83165, 0.872474,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        })));

INSTANTIATE_TEST_SUITE_P(
        prior_box_test_four_variances,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 0.0f, 0.0f, {0.1, 0.2, 0.3, 0.4}, true, true}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.0918861, 0.0918861, 0.408114, 0.408114, 0.127526, 0.16835, 0.372474, 0.33165,
            0.65, 0.15, 0.85, 0.35,
            0.591886, 0.0918861, 0.908114, 0.408114, 0.627526, 0.16835, 0.872474, 0.33165, 0.15, 0.65, 0.35, 0.85,
            0.0918861, 0.591886, 0.408114, 0.908114,
            0.127526, 0.66835, 0.372474, 0.83165, 0.65, 0.65, 0.85, 0.85, 0.591886, 0.591886, 0.908114, 0.908114,
            0.627526, 0.66835, 0.872474, 0.83165,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4
        })));

INSTANTIATE_TEST_SUITE_P(
        DISABLED_prior_box_test_dont_scale,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 0.0f, 0.0f, {}, false, true}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.0918861, 0.0918861, 0.408114, 0.408114, 0.127526, 0.16835, 0.372474, 0.33165,
            0.65, 0.15, 0.85, 0.35, 0.591886, 0.0918861, 0.908114, 0.408114, 0.627526, 0.16835, 0.872474, 0.33165,
            0.15, 0.65, 0.35, 0.85, 0.0918861, 0.591886, 0.408114, 0.908114, 0.127526, 0.66835, 0.372474, 0.83165,
            0.65, 0.65, 0.85, 0.85, 0.591886, 0.591886, 0.908114, 0.908114, 0.627526, 0.66835, 0.872474, 0.83165,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        })));

INSTANTIATE_TEST_SUITE_P(
        DISABLED_prior_box_test_fixed_density,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {0.2, 0.5}, {2.0, 3.0}, {0.1, 0.5}, true, true, 0.0f, 0.0f, {}, true, false}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.127526, 0.16835, 0.372474, 0.33165, 0.16835, 0.127526, 0.33165, 0.372474, 0.0918861,
            0.0918861, 0.408114, 0.408114, 0.65, 0.15, 0.85, 0.35, 0.627526, 0.16835, 0.872474, 0.33165, 0.66835, 0.127526,
            0.83165, 0.372474, 0.591886, 0.0918861, 0.908114, 0.408114,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        })));

INSTANTIATE_TEST_SUITE_P(
        DISABLED_prior_box_test_step_offset,
        prior_box_test_i32_f32,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 4.0f, 1.0f, {}, false, false}),
        testing::Values(std::vector<float>{
            0.3, 0.3, 0.5, 0.5, 0.277526, 0.31835, 0.522474, 0.48165, 0.241886, 0.241886, 0.558114, 0.558114,
            0.7, 0.3, 0.9, 0.5, 0.677526, 0.31835, 0.922475, 0.48165, 0.641886, 0.241886, 0.958114, 0.558114,
            0.3, 0.7, 0.5, 0.9, 0.277526, 0.71835, 0.522474, 0.88165, 0.241886, 0.641886, 0.558114, 0.958114,
            0.7, 0.7, 0.9, 0.9, 0.677526, 0.71835, 0.922475, 0.88165, 0.641886, 0.641886, 0.958114, 0.958114,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        })));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(prior_box_test_i32_f32, prior_box_test_i32_f32_cached) {
    this->execute(true);
}
#else
using prior_box_test_i32_f32_cached = PriorBoxGPUTest<int32_t, float>;
TEST_P(prior_box_test_i32_f32_cached, prior_box_test_i32_f32) {
    this->execute(true);
}

INSTANTIATE_TEST_SUITE_P(
        prior_box_test_four_variances,
        prior_box_test_i32_f32_cached,
        testing::Combine(
        testing::Values(format::bfyx),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 0.0f, 0.0f, {0.1, 0.2, 0.3, 0.4}, true, true}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.0918861, 0.0918861, 0.408114, 0.408114, 0.127526, 0.16835, 0.372474, 0.33165,
            0.65, 0.15, 0.85, 0.35,
            0.591886, 0.0918861, 0.908114, 0.408114, 0.627526, 0.16835, 0.872474, 0.33165, 0.15, 0.65, 0.35, 0.85,
            0.0918861, 0.591886, 0.408114, 0.908114,
            0.127526, 0.66835, 0.372474, 0.83165, 0.65, 0.65, 0.85, 0.85, 0.591886, 0.591886, 0.908114, 0.908114,
            0.627526, 0.66835, 0.872474, 0.83165,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4
        })));
#endif
}  // namespace
