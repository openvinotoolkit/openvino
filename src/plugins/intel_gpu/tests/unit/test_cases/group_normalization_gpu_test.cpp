// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/group_normalization.hpp>
#include <intel_gpu/primitives/detection_output.hpp>
#include "openvino/reference/group_normalization.hpp"
#include "compilation_context.hpp"


using namespace cldnn;
using namespace ::tests;

namespace {

typedef std::tuple<
std::vector<std::int32_t>,  // Input shape
std::size_t,                // Number of groups
double,                     // Epsilon
format                      // First input layout
>
GroupNormalizationParams;

class GroupNormalizationGPUTest : public ::testing::TestWithParam<GroupNormalizationParams> {
public:
    GroupNormalizationGPUTest() = default;

    void SetUp() override {
        std::vector<std::int32_t> input_shape;
        const auto& params = GetParam();
        std::tie(input_shape, num_groups_, epsilon_, format_) = params;
        std::copy(std::begin(input_shape), std::end(input_shape), std::back_inserter(data_shape_));
        tests::random_generator rg{"GroupNormalizationGPUTest"};
        data_ = rg.generate_random_1d<float>(ov::shape_size(input_shape), -1, 1);
        scale_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        bias_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        const auto planar_format = format::dimension(format_) == 4 ? format::bfyx : format::bfzyx;

        topology tp;
        auto &engine = get_test_engine();
        data_layout_ = layout{data_types::f32, planar_format, tensor{input_shape}};
        scale_bias_layout_ = layout{data_types::f32, planar_format, tensor{1,
            static_cast<std::int32_t>(scale_.size()), 1, 1}};

        primitive_id reordered_data_primitive = data_primitive_ + "_reordered";
        tp.add(input_layout{data_primitive_, data_layout_});
        tp.add(input_layout{scale_primitive_, scale_bias_layout_});
        tp.add(input_layout{bias_primitive_, scale_bias_layout_});
        tp.add(reorder{reordered_data_primitive, data_primitive_, format_, data_types::f32});

        // auto g = group_normalization{
        //     "group_normalization_output",
        //     input_info{reordered_data_primitive},
        //     input_info{scale_primitive_},
        //     input_info{bias_primitive_},
        //     static_cast<std::int64_t>(num_groups_),
        //     epsilon_
        // };
        // tp.add(g);
        // tp.add(reorder{"reordered_output", input_info("group_normalization_output"), planar_format, data_types::f32});


        for (int i = 0 ; i < 3; i++) {
            auto dis_num = (i + 1);

            std::ostringstream name_str;
            name_str << "group_norm" << dis_num;
            const std::string name = name_str.str();

            std::ostringstream output_str;
            output_str << "reordered_output" << dis_num;
            const std::string output_name = output_str.str();

            auto g = group_normalization{
                name,
                input_info{reordered_data_primitive},
                input_info{scale_primitive_},
                input_info{bias_primitive_},
                static_cast<std::int64_t>(num_groups_),
                epsilon_
            };
            tp.add(g);
            tp.add(reorder{output_name, input_info(name), planar_format, data_types::f32});
        }

        auto config = get_test_default_config(engine);
        //config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        network_ = std::make_shared<cldnn::network>(engine, tp, config);
    }

    //setup for detection_output
    // void SetUp() override {
    //     std::vector<std::int32_t> input_shape;
    //     const auto& params = GetParam();
    //     std::tie(input_shape, num_groups_, epsilon_, format_) = params;
    //     //std::copy(std::begin(input_shape), std::end(input_shape), std::back_inserter(data_shape_));
    //     tests::random_generator rg{"GroupNormalizationGPUTest"};

    //     //int32_t b = input_shape[0]
    //     int32_t num_of_images = input_shape[1];
    //     int32_t num_priors = input_shape[2];
    //     int32_t num_loc_classes = input_shape[3];

    //     std::vector<std::int32_t> input_location = {num_of_images, num_priors * num_loc_classes * 4, 1, 1};
    //     std::vector<std::int32_t> input_confidence = {num_of_images, num_priors * num_loc_classes, 1, 1};
    //     std::vector<std::int32_t> input_prior_box = {1, 2, 1, num_priors * 4};

    //     data_ = rg.generate_random_1d<float>(ov::shape_size(input_location), -1, 1);
    //     scale_ = rg.generate_random_1d<float>(ov::shape_size(input_confidence), -1, 1);
    //     bias_ = rg.generate_random_1d<float>(ov::shape_size(input_prior_box), -1, 1);
    //     const auto planar_format = format::dimension(format_) == 4 ? format::bfyx : format::bfzyx;

    //     topology tp;
    //     auto &engine = get_test_engine();
    //     data_layout_ = layout{data_types::f32, planar_format, tensor{input_location}};
    //     scale_bias_layout_ = layout{data_types::f32, planar_format, tensor{input_confidence}};
    //     input_prior_box_layout_ = layout{data_types::f32, planar_format, tensor{input_prior_box}};

    //     primitive_id reordered_data_primitive = data_primitive_ + "_reordered";
    //     tp.add(input_layout{data_primitive_, data_layout_});
    //     tp.add(input_layout{scale_primitive_, scale_bias_layout_});
    //     tp.add(input_layout{bias_primitive_, input_prior_box_layout_});
    //     tp.add(reorder{reordered_data_primitive, data_primitive_, format_, data_types::f32});

    //     for (int i = 0 ; i < 1; i++) {
    //         auto dis_num = (i + 1);
    //         std::ostringstream name_str;
    //         //name_str << "group_norm" << dis_num;
    //         name_str << "detection_output" << dis_num;
    //         const std::string name = name_str.str();

    //         std::ostringstream output_str;
    //         output_str << "reordered_output" << dis_num;
    //         const std::string output_name = output_str.str();

    //         // auto g = group_normalization{
    //         //     name,
    //         //     input_info{reordered_data_primitive},
    //         //     input_info{scale_primitive_},
    //         //     input_info{bias_primitive_},
    //         //     static_cast<std::int64_t>(num_groups_),
    //         //     epsilon_
    //         // };

    //         auto g = detection_output{
    //             name,
    //             { input_info(reordered_data_primitive), input_info(scale_primitive_), input_info(bias_primitive_) },
    //             2,
    //             150
    //         };


    //         tp.add(g);
    //         tp.add(reorder{output_name, input_info(name), planar_format, data_types::f32});
    //     }

    //     auto config = get_test_default_config(engine);
    //     //config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    //     network_ = std::make_shared<cldnn::network>(engine, tp, config);
    // }

    void Test() {
        auto &engine = get_test_engine();
        auto data_gpu_mem = engine.allocate_memory(data_layout_);
        auto scale_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        auto bias_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        //auto bias_gpu_mem = engine.allocate_memory(input_prior_box_layout_);
        set_values(data_gpu_mem, data_);
        set_values(scale_gpu_mem, scale_);
        set_values(bias_gpu_mem, bias_);
        network_->set_input_data(data_primitive_, data_gpu_mem);
        network_->set_input_data(scale_primitive_, scale_gpu_mem);
        network_->set_input_data(bias_primitive_, bias_gpu_mem);
        auto outputs = network_->execute();

        auto output = outputs.at("reordered_output1").get_memory();
        cldnn::mem_lock<float> output_gpu_mem(output, get_test_stream());

        std::vector<float> reference_output(data_.size());
        ov::reference::group_normalization(data_.data(), scale_.data(), bias_.data(), reference_output.data(),
                                           ov::Shape{data_shape_}, num_groups_, epsilon_);

        ASSERT_EQ(output_gpu_mem.size(), reference_output.size());
        for (std::size_t i = 0; i < reference_output.size(); i++) {
            ASSERT_NEAR(output_gpu_mem[i], reference_output[i], 0.0001);
        }
    }

private:
    std::vector<float> data_{};
    std::vector<float> scale_{};
    std::vector<float> bias_{};
    std::size_t num_groups_{};
    double epsilon_{};
    format format_{format::any};
    network::ptr network_{};
    layout data_layout_{};
    layout scale_bias_layout_{};
    //layout input_prior_box_layout_{};
    std::vector<std::size_t> data_shape_;
    static const primitive_id data_primitive_;
    static const primitive_id scale_primitive_;
    static const primitive_id bias_primitive_;
};

const primitive_id GroupNormalizationGPUTest::data_primitive_{"data"};
const primitive_id GroupNormalizationGPUTest::scale_primitive_{"scale"};
const primitive_id GroupNormalizationGPUTest::bias_primitive_{"bias"};

TEST_P(GroupNormalizationGPUTest, blocked_layouts_support) {
    Test();
}

const std::vector<cldnn::format> f_blocked_4d_formats {
    format::b_fs_yx_fsv2,
    format::b_fs_yx_fsv4,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
};

const std::vector<cldnn::format> f_blocked_5d_formats {
    // format::b_fs_zyx_fsv2,
    // format::b_fs_zyx_fsv4,
    // format::b_fs_zyx_fsv16,
    // format::b_fs_zyx_fsv32,

    format::b_fs_zyx_fsv16,
    //format::bfzyx,
};

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<int32_t>{3, 64, 32, 64}),
        ::testing::Values(4),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_4d_formats)));

// INSTANTIATE_TEST_SUITE_P(
//     GroupNormalizationGPUTest_blocked_layouts_support_5d, GroupNormalizationGPUTest,
//     ::testing::Combine(
//         ::testing::Values(std::vector<int32_t>{3, 64, 28, 32, 12}),
//         ::testing::Values(4),
//         ::testing::Values(0.0025),
//         ::testing::ValuesIn(f_blocked_5d_formats)));


INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_5d_16_1024_256_128, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<int32_t>{1, 16, 1024, 256, 128}), // fail with loop 1
        ::testing::Values(8),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_5d_formats)));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_5d_16_128_128_128, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<int32_t>{1, 16, 128, 128, 128}),  // pass with loop 1, fail with loop 3
        //::testing::Values(std::vector<int32_t>{1, 16, 256, 256, 256}), // fail with loop 1, fail with wait()
        //::testing::Values(std::vector<int32_t>{1, 16, 1024, 256, 128}), // fail with loop 1, fail with wait() or finish()
        ::testing::Values(8),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_5d_formats)));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_5d_16_8_16_16, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<int32_t>{1, 16, 8, 16, 16}),  // pass with loop 3
        ::testing::Values(8),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_5d_formats)));

// INSTANTIATE_TEST_SUITE_P(
//     GroupNormalizationGPUTest_blocked_layouts_support_5d, GroupNormalizationGPUTest,
//     ::testing::Combine(
//         //::testing::Values(std::vector<int32_t>{1, 16, 16, 16}), //passed with loop 1
//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 32}), //passed with loop 1
//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 32}), //passed with loop 3
//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 64}), //passed with loop 1
//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 64}), //passed with loop 3

//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 64}), //hang with loop 6
//         //::testing::Values(std::vector<int32_t>{1, 32, 32, 64}), //hang with loop 20
//         //::testing::Values(std::vector<int32_t>{1, 32, 64, 64}), //failed with loop 1
//         //::testing::Values(std::vector<int32_t>{1, 64, 64, 64}), //failed with loop 1
//         ::testing::Values(std::vector<int32_t>{1, 128, 128, 128}), //failed with loop 1
//         //::testing::Values(std::vector<int32_t>{1, 128, 128, 128}), //failed with loop 3
        
//         ::testing::Values(8),
//         ::testing::Values(0.0025),
//         ::testing::Values(format::bfyx)));

} // anonymous namespace
