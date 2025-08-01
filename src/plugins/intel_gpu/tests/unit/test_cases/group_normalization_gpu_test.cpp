// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "random_generator.hpp"
#include "program_wrapper.h"
#include "pass_manager.h"

#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/reference/group_normalization.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"


using namespace cldnn;
using namespace ::tests;

namespace {

typedef std::tuple<
    std::vector<ov::Dimension::value_type>, // Input shape
    std::size_t,                            // Number of groups
    double,                                 // Epsilon
    format,                                 // First input layout
    format,                                 // Output layout
    padding                                 // Output padding
>
GroupNormalizationParams;

class GroupNormalizationGPUTest : public ::testing::TestWithParam<GroupNormalizationParams> {
public:
    GroupNormalizationGPUTest() = default;

    void SetUp() override {
        std::vector<ov::Dimension::value_type> input_shape;
        const auto& params = GetParam();
        std::tie(input_shape, num_groups_, epsilon_, in_format_, out_format_, output_pad_) = params;
        std::copy(std::begin(input_shape), std::end(input_shape), std::back_inserter(data_shape_));
        tests::random_generator rg{"GroupNormalizationGPUTest"};
        data_ = rg.generate_random_1d<float>(ov::shape_size(input_shape), -1, 1);
        scale_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        bias_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        const auto planar_format = format::dimension(in_format_) == 4 ? format::bfyx : format::bfzyx;
        bool is_dynamic_test = (in_format_ != out_format_);

        topology tp;
        auto &engine = get_test_engine();
        data_layout_ = layout{data_types::f32, planar_format, tensor{input_shape}};
        scale_bias_layout_ = layout{data_types::f32, planar_format, tensor{1,
            static_cast<std::int32_t>(scale_.size()), 1, 1}};

        primitive_id reordered_data_primitive = data_primitive_ + "_reordered";
        if (is_dynamic_test) {
            ov::PartialShape pshape = {};
            for (size_t i = 0; i < in_format_.dimension(); i++) {
                pshape.push_back(ov::Dimension::dynamic());
            }
            layout in_layout{pshape, data_types::f32, planar_format};
            tp.add(input_layout{data_primitive_, in_layout});
        } else {
            tp.add(input_layout{data_primitive_, data_layout_});
        }
        tp.add(input_layout{scale_primitive_, scale_bias_layout_});
        tp.add(input_layout{bias_primitive_, scale_bias_layout_});
        tp.add(reorder{reordered_data_primitive, data_primitive_, in_format_, data_types::f32});

        auto g = group_normalization{
            "group_normalization_output",
            input_info{reordered_data_primitive},
            input_info{scale_primitive_},
            input_info{bias_primitive_},
            static_cast<std::int64_t>(num_groups_),
            epsilon_
        };
        g.output_paddings = {output_pad_};
        tp.add(g);
        tp.add(reorder{"group_normalization_output_reordered", input_info("group_normalization_output"), out_format_, data_types::f32});
        tp.add(eltwise{"dummy_max",
                       input_info("group_normalization_output_reordered"),
                       input_info("group_normalization_output_reordered"),
                       cldnn::eltwise_mode::max});
        tp.add(reorder{"output", input_info("dummy_max"), planar_format, data_types::f32});

        auto config = get_test_default_config(engine);
        if (is_dynamic_test) {
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        }

        network_ = std::make_shared<cldnn::network>(engine, tp, config);
    }

    void Test() {
        auto &engine = get_test_engine();
        auto data_gpu_mem = engine.allocate_memory(data_layout_);
        auto scale_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        auto bias_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        set_values(data_gpu_mem, data_);
        set_values(scale_gpu_mem, scale_);
        set_values(bias_gpu_mem, bias_);
        network_->set_input_data(data_primitive_, data_gpu_mem);
        network_->set_input_data(scale_primitive_, scale_gpu_mem);
        network_->set_input_data(bias_primitive_, bias_gpu_mem);
        auto outputs = network_->execute();
        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> output_gpu_mem(output, get_test_stream());

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
    format in_format_{format::any};
    format out_format_{format::any};
    padding output_pad_{padding()};
    network::ptr network_{};
    layout data_layout_{};
    layout scale_bias_layout_{};
    std::vector<std::size_t> data_shape_;
    static const primitive_id data_primitive_;
    static const primitive_id scale_primitive_;
    static const primitive_id bias_primitive_;
};

const primitive_id GroupNormalizationGPUTest::data_primitive_{"data"};
const primitive_id GroupNormalizationGPUTest::scale_primitive_{"scale"};
const primitive_id GroupNormalizationGPUTest::bias_primitive_{"bias"};

TEST_P(GroupNormalizationGPUTest, random) {
    Test();
}

const std::vector<cldnn::format> f_planar_4d_formats {
    format::bfyx,
};

const std::vector<cldnn::format> f_blocked_4d_formats {
    format::b_fs_yx_fsv16,
};

const std::vector<cldnn::format> f_4d_formats {
    format::bfyx,
    format::b_fs_yx_fsv16,
};

const std::vector<cldnn::format> f_planar_5d_formats {
    format::bfzyx,
};

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_planar_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 32, 64}, std::vector<ov::Dimension::value_type>{3, 124, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 1536, 151, 1}, std::vector<ov::Dimension::value_type>{1, 12, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_planar_4d_formats),
        ::testing::ValuesIn(f_4d_formats),
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 32, 64}, std::vector<ov::Dimension::value_type>{3, 124, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 1536, 151, 1}, std::vector<ov::Dimension::value_type>{1, 12, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 2, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_4d_formats),
        ::testing::ValuesIn(f_4d_formats),
        ::testing::ValuesIn({padding(), padding({0, 16, 0, 0})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_planar_layouts_support_5d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 28, 32, 12}, std::vector<ov::Dimension::value_type>{3, 124, 10, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 1536, 9, 151, 1}, std::vector<ov::Dimension::value_type>{1, 12, 8, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_planar_5d_formats),
        ::testing::ValuesIn(f_planar_5d_formats),
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})})));

} // anonymous namespace

TEST(group_normalization, input_bfyx_output_fsv16) {
    GTEST_SKIP();
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{1, 3, 3, 2}, data_types::f32, format::bfyx };
    auto scale_layout = layout{ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx };
    auto bias_layout = layout{ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx };

    auto input_mem = engine.allocate_memory(in_layout);
    auto scale_mem = engine.allocate_memory(scale_layout);
    auto bias_mem = engine.allocate_memory(bias_layout);

    set_values<float>(input_mem,
               { 0.125, 0.125, 0.875, -0.125, 0.125, 0.750,
                0.875, -0.375, -0.375, -1.000, -0.625, -1.000,
                -0.125, -0.750, -0.250, 0.625, -0.500, -0.875 });
    set_values(scale_mem, { 0.125f });
    set_values(bias_mem, { 0.75f });

    topology topology_g(
        input_layout("input", in_layout),
        input_layout("scale", scale_layout),
        input_layout("bias", bias_layout),
        group_normalization("group_normalization", input_info("input"), input_info("scale"), input_info("bias"), static_cast<std::int64_t>(1), 0.0025),
        permute("output", input_info("group_normalization"), {0, 1, 2, 3})
    );

    topology topology_t(
        input_layout("input", in_layout),
        input_layout("scale", scale_layout),
        input_layout("bias", bias_layout),
        reorder("reorder1", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        group_normalization("group_normalization", input_info("reorder1"), input_info("scale"), input_info("bias"), static_cast<std::int64_t>(1), 0.0025),
        reorder("reorder2", input_info("group_normalization"), format::b_fs_yx_fsv16, data_types::f32),
        permute("output", input_info("reorder2"), {0, 1, 2, 3})
    );

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc gn_impl = { format::bfyx, "", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"group_normalization", gn_impl}}));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network_g(engine, topology_g, config);
    network_g.set_input_data("input", input_mem);
    network_g.set_input_data("scale", scale_mem);
    network_g.set_input_data("bias", bias_mem);

    auto outputs_g = network_g.execute();
    auto output_g = outputs_g.at("output").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_mem_g(output_g, get_test_stream());

    // Disable mem reuse to avoid wrong reuse due to not calculating of memory dependencies in the below model creation flow
    config.set_property(ov::intel_gpu::enable_memory_pool(false));
    auto program = program::build_program(engine, topology_t, config, false, true);
    auto& reorder_node = program->get_node("reorder1");
    std::vector<layout> layouts = {in_layout};
    reorder_node.set_output_layouts(layouts, false);
    program_wrapper::build(*program);

    network network_t(program);
    network_t.set_input_data("input", input_mem);
    network_t.set_input_data("scale", scale_mem);
    network_t.set_input_data("bias", bias_mem);

    auto outputs_t = network_t.execute();
    auto output_t = outputs_g.at("output").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_mem_t(output_t, get_test_stream());

    ASSERT_EQ(output_mem_g.size(), output_mem_t.size());
    ASSERT_EQ(outputs_g.begin()->first, outputs_t.begin()->first);

    for (std::size_t i = 0; i < output_mem_t.size(); i++) {
        ASSERT_NEAR(output_mem_t[i], output_mem_g[i], 0.0001);
    }
}

TEST(group_normalization, basic_b_fs_yx_fsv16) {
    auto& engine = get_test_engine();

    const ov::Shape input_shape = {1, 128, 256, 256};
    const ov::Shape param_shape = {128, 1, 1, 1};
    auto in_layout = layout{input_shape, data_types::f32, format::bfyx};
    auto scale_layout = layout{param_shape, data_types::f32, format::bfyx};
    auto bias_layout = layout{param_shape, data_types::f32, format::bfyx};
    const float epsilon = 1e-5f;
    const int64_t num_groups = 1;

    auto input_mem = engine.allocate_memory(in_layout);
    auto scale_mem = engine.allocate_memory(scale_layout);
    auto bias_mem = engine.allocate_memory(bias_layout);

    tests::random_generator rg{"GroupNormalizationGPUTest"};

    std::vector<float> in_vals = rg.generate_random_1d<float>(ov::shape_size(input_shape), -1, 1);
    std::vector<float> scale_vals = rg.generate_random_1d<float>(input_shape[1], -1, 1);
    std::vector<float> bias_vals = rg.generate_random_1d<float>(input_shape[1], -1, 1);

    set_values(input_mem, in_vals);
    set_values(scale_mem, scale_vals);
    set_values(bias_mem, bias_vals);

    topology topology(
        input_layout("input_bfyx_f32", in_layout),
        input_layout("scale_bfyx_f32", scale_layout),
        input_layout("bias_bfyx_f32", bias_layout),
        reorder("input_fsv16_f32", input_info("input_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
        reorder("scale_fsv16_f32", input_info("scale_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
        reorder("bias_fsv16_f32", input_info("bias_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
        group_normalization("group_normalization_fsv16_f32",
                            input_info("input_fsv16_f32"),
                            input_info("scale_fsv16_f32"),
                            input_info("bias_fsv16_f32"),
                            num_groups,
                            epsilon),
        reorder("output_bfyx_f32", input_info("group_normalization_fsv16_f32"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc gn_impl = {format::b_fs_yx_fsv16, "group_normalization_fsv16", impl_types::ocl};
    config.set_property(
        ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"group_normalization_fsv16_f32", gn_impl}}));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input_bfyx_f32", input_mem);
    network.set_input_data("scale_bfyx_f32", scale_mem);
    network.set_input_data("bias_bfyx_f32", bias_mem);

    auto outputs = network.execute();
    auto output = outputs.at("output_bfyx_f32").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_mem_lock(output, get_test_stream());

    cldnn::mem_lock<float, mem_lock_type::read> input_mem_lock(input_mem, get_test_stream());
    cldnn::mem_lock<float, mem_lock_type::read> scale_mem_lock(scale_mem, get_test_stream());
    cldnn::mem_lock<float, mem_lock_type::read> bias_mem_lock(bias_mem, get_test_stream());

    std::vector<float> reference_output(output_mem_lock.size());
    ov::reference::group_normalization(input_mem_lock.data(),
                                       scale_mem_lock.data(),
                                       bias_mem_lock.data(),
                                       reference_output.data(),
                                       input_shape,
                                       num_groups,
                                       epsilon);

    for (std::size_t i = 0; i < reference_output.size(); i++) {
        ASSERT_NEAR(reference_output[i], output_mem_lock[i], 0.01);
    }
}