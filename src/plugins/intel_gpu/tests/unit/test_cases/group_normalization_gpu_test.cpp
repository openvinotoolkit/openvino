// Copyright (C) 2018-2026 Intel Corporation
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
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "openvino/reference/group_normalization.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"

#include <algorithm>
#include <array>
#include <sstream>


using namespace cldnn;
using namespace ::tests;

namespace {

constexpr auto byxf_impl_type = "ov::intel_gpu::ocl::GroupNormalizationByxfOptImpl";

typedef std::tuple<
    std::vector<ov::Dimension::value_type>, // Input shape
    std::size_t,                            // Number of groups
    double,                                 // Epsilon
    format,                                 // First input layout
    padding,                                // First input padding
    format,                                 // Output layout
    padding                                 // Output padding
>
GroupNormalizationParams;

class GroupNormalizationGPUTest : public ::testing::TestWithParam<GroupNormalizationParams> {
public:
    GroupNormalizationGPUTest() = default;

    void SetUp() override {
        const auto& params = GetParam();
        const auto& [input_shape, _num_groups_, _epsilon_, _in_format_, _in_pad_, _out_format_, _output_pad_] = params;
        num_groups_ = _num_groups_;
        epsilon_ = _epsilon_;
        in_format_ = _in_format_;
        in_pad_ = _in_pad_;
        out_format_ = _out_format_;
        output_pad_ = _output_pad_;
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
        tp.add(reorder{reordered_data_primitive, data_primitive_, layout{input_shape, data_types::f32, in_format_, in_pad_}});

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
        if (in_format_ == format::byxf) {
            ASSERT_EQ(network_->get_primitive("group_normalization_output")->get_impl()->get_type_info(), byxf_impl_type);
        }
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
    padding in_pad_{padding()};
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
        ::testing::ValuesIn({padding()}),
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
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})}),
        ::testing::ValuesIn(f_4d_formats),
        ::testing::ValuesIn({padding(), padding({0, 16, 0, 0})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_4d_fused, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 32, 64}, std::vector<ov::Dimension::value_type>{3, 128, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 48, 151, 1}, std::vector<ov::Dimension::value_type>{1, 16, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{4, 8}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_4d_formats),
        ::testing::Values(padding()),
        ::testing::ValuesIn(f_blocked_4d_formats),
        ::testing::Values(padding())));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_byxf_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 32, 64}, std::vector<ov::Dimension::value_type>{3, 124, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 1536, 151, 1}, std::vector<ov::Dimension::value_type>{1, 12, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::Values(format::byxf),
        ::testing::Values(padding()),
        ::testing::Values(format::byxf),
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_planar_layouts_support_5d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<ov::Dimension::value_type>{3, 64, 28, 32, 12}, std::vector<ov::Dimension::value_type>{3, 124, 10, 97, 61},
                             std::vector<ov::Dimension::value_type>{1, 1536, 9, 151, 1}, std::vector<ov::Dimension::value_type>{1, 12, 8, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_planar_5d_formats),
        ::testing::ValuesIn({padding()}),
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

TEST(group_normalization, basic_b_fs_yx_fsv16_fused) {
    auto& engine = get_test_engine();

    const ov::Shape input_shape = {1, 128, 256, 256};
    const ov::Shape param_shape = {128, 1, 1, 1};
    auto in_layout = layout{input_shape, data_types::f32, format::bfyx};
    auto scale_layout = layout{param_shape, data_types::f32, format::bfyx};
    auto bias_layout = layout{param_shape, data_types::f32, format::bfyx};
    const float epsilon = 1e-5f;
    const int64_t num_groups = 32;

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

    topology topology(input_layout("input_bfyx_f32", in_layout),
                      input_layout("scale_bfyx_f32", scale_layout),
                      input_layout("bias_bfyx_f32", bias_layout),
                      reorder("input_fsv16_f32", input_info("input_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
                      reorder("scale_fsv16_f32", input_info("scale_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
                      reorder("bias_fsv16_f32", input_info("bias_bfyx_f32"), format::b_fs_yx_fsv16, data_types::f32),
                      group_normalization("group_normalization_fsv16_fused_f32",
                                          input_info("input_fsv16_f32"),
                                          input_info("scale_fsv16_f32"),
                                          input_info("bias_fsv16_f32"),
                                          num_groups,
                                          epsilon),
                      reorder("output_bfyx_f32", input_info("group_normalization_fsv16_fused_f32"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc gn_impl = {format::b_fs_yx_fsv16, "group_normalization_fsv16", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"group_normalization_fsv16_fused_f32", gn_impl}}));
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

// Tests fused eltwise_sum with a separate 4D input into the group_normalization fused kernel.
// This covers the case where FUSED_OP0_LOAD indexes a separate tensor using (b),(f),(y),(x).
TEST(group_normalization, basic_b_fs_yx_fsv16_fused_eltwise_sum) {
    auto& engine = get_test_engine();

    const ov::Shape input_shape = {1, 32, 64, 64};
    const ov::Shape param_shape = {32, 1, 1, 1};
    auto in_layout = layout{input_shape, data_types::f16, format::bfyx};
    auto eltwise_layout = layout{input_shape, data_types::f16, format::bfyx};
    auto scale_layout = layout{param_shape, data_types::f16, format::bfyx};
    auto bias_layout = layout{param_shape, data_types::f16, format::bfyx};
    const float epsilon = 1e-5f;
    const int64_t num_groups = 8;

    auto input_mem = engine.allocate_memory(in_layout);
    auto eltwise_mem = engine.allocate_memory(eltwise_layout);
    auto scale_mem = engine.allocate_memory(scale_layout);
    auto bias_mem = engine.allocate_memory(bias_layout);

    tests::random_generator rg{"GroupNormalizationGPUTest_fused_eltwise"};

    auto in_vals = rg.generate_random_1d<ov::float16>(ov::shape_size(input_shape), -1, 1);
    auto eltwise_vals = rg.generate_random_1d<ov::float16>(ov::shape_size(input_shape), -1, 1);
    auto scale_vals = rg.generate_random_1d<ov::float16>(input_shape[1], -1, 1);
    auto bias_vals = rg.generate_random_1d<ov::float16>(input_shape[1], -1, 1);

    set_values(input_mem, in_vals);
    set_values(eltwise_mem, eltwise_vals);
    set_values(scale_mem, scale_vals);
    set_values(bias_mem, bias_vals);

    // Topology: group_norm -> eltwise_sum(with separate input) -> output
    // The eltwise_sum should be fused into the group_normalization kernel.
    topology topology(
        input_layout("input", in_layout),
        input_layout("eltwise_input", eltwise_layout),
        input_layout("scale", scale_layout),
        input_layout("bias", bias_layout),
        reorder("input_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("eltwise_input_fsv16", input_info("eltwise_input"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("scale_fsv16", input_info("scale"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("bias_fsv16", input_info("bias"), format::b_fs_yx_fsv16, data_types::f16),
        group_normalization("group_norm",
                            input_info("input_fsv16"),
                            input_info("scale_fsv16"),
                            input_info("bias_fsv16"),
                            num_groups,
                            epsilon),
        eltwise("eltwise_sum",
                {input_info("group_norm"), input_info("eltwise_input_fsv16")},
                eltwise_mode::sum),
        reorder("output", input_info("eltwise_sum"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc gn_impl = {format::b_fs_yx_fsv16, "group_normalization_fsv16", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"group_norm", gn_impl}}));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("eltwise_input", eltwise_mem);
    network.set_input_data("scale", scale_mem);
    network.set_input_data("bias", bias_mem);

    auto outputs = network.execute();

    // Verify eltwise_sum was fused into group_norm (not executed as separate primitive)
    auto executed_primitives = network.get_executed_primitives();
    ASSERT_TRUE(executed_primitives.count("eltwise_sum") == 0);

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_mem_lock(output, get_test_stream());

    // Compute reference: group_norm + eltwise_sum
    std::vector<float> in_f32(in_vals.begin(), in_vals.end());
    std::vector<float> scale_f32(scale_vals.begin(), scale_vals.end());
    std::vector<float> bias_f32(bias_vals.begin(), bias_vals.end());
    std::vector<float> ref_gn(ov::shape_size(input_shape));
    ov::reference::group_normalization(in_f32.data(), scale_f32.data(), bias_f32.data(),
                                       ref_gn.data(), input_shape, num_groups, epsilon);

    for (std::size_t i = 0; i < ref_gn.size(); i++) {
        float expected = ref_gn[i] + static_cast<float>(eltwise_vals[i]);
        ASSERT_NEAR(static_cast<float>(output_mem_lock[i]), expected, 0.05) << "at index " << i;
    }
}

// Tests fused eltwise_sum + ReLU activation into the group_normalization fused kernel.
// This matches the SegNext model pattern: GroupNorm -> Add -> ReLU.
TEST(group_normalization, basic_b_fs_yx_fsv16_fused_eltwise_sum_relu) {
    auto& engine = get_test_engine();

    const ov::Shape input_shape = {1, 32, 33, 17};  // Non-power-of-2 spatial dims to stress y/x derivation and CHUNK_SIZE tail handling
    const ov::Shape param_shape = {32, 1, 1, 1};
    auto in_layout = layout{input_shape, data_types::f16, format::bfyx};
    auto eltwise_layout = layout{input_shape, data_types::f16, format::bfyx};
    auto scale_layout = layout{param_shape, data_types::f16, format::bfyx};
    auto bias_layout = layout{param_shape, data_types::f16, format::bfyx};
    const float epsilon = 1e-5f;
    const int64_t num_groups = 8;

    auto input_mem = engine.allocate_memory(in_layout);
    auto eltwise_mem = engine.allocate_memory(eltwise_layout);
    auto scale_mem = engine.allocate_memory(scale_layout);
    auto bias_mem = engine.allocate_memory(bias_layout);

    tests::random_generator rg{"GroupNormalizationGPUTest_fused_eltwise_relu"};

    auto in_vals = rg.generate_random_1d<ov::float16>(ov::shape_size(input_shape), -1, 1);
    auto eltwise_vals = rg.generate_random_1d<ov::float16>(ov::shape_size(input_shape), -1, 1);
    auto scale_vals = rg.generate_random_1d<ov::float16>(input_shape[1], -1, 1);
    auto bias_vals = rg.generate_random_1d<ov::float16>(input_shape[1], -1, 1);

    set_values(input_mem, in_vals);
    set_values(eltwise_mem, eltwise_vals);
    set_values(scale_mem, scale_vals);
    set_values(bias_mem, bias_vals);

    topology topology(
        input_layout("input", in_layout),
        input_layout("eltwise_input", eltwise_layout),
        input_layout("scale", scale_layout),
        input_layout("bias", bias_layout),
        reorder("input_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("eltwise_input_fsv16", input_info("eltwise_input"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("scale_fsv16", input_info("scale"), format::b_fs_yx_fsv16, data_types::f16),
        reorder("bias_fsv16", input_info("bias"), format::b_fs_yx_fsv16, data_types::f16),
        group_normalization("group_norm",
                            input_info("input_fsv16"),
                            input_info("scale_fsv16"),
                            input_info("bias_fsv16"),
                            num_groups,
                            epsilon),
        eltwise("eltwise_sum",
                {input_info("group_norm"), input_info("eltwise_input_fsv16")},
                eltwise_mode::sum),
        activation("relu", input_info("eltwise_sum"), activation_func::relu),
        reorder("output", input_info("relu"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc gn_impl = {format::b_fs_yx_fsv16, "group_normalization_fsv16", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"group_norm", gn_impl}}));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("eltwise_input", eltwise_mem);
    network.set_input_data("scale", scale_mem);
    network.set_input_data("bias", bias_mem);

    auto outputs = network.execute();

    // Verify eltwise_sum and relu were fused into group_norm (not executed as separate primitives)
    auto executed_primitives = network.get_executed_primitives();
    ASSERT_TRUE(executed_primitives.count("eltwise_sum") == 0);
    ASSERT_TRUE(executed_primitives.count("relu") == 0);

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_mem_lock(output, get_test_stream());

    // Compute reference: group_norm + eltwise_sum + relu
    std::vector<float> in_f32(in_vals.begin(), in_vals.end());
    std::vector<float> scale_f32(scale_vals.begin(), scale_vals.end());
    std::vector<float> bias_f32(bias_vals.begin(), bias_vals.end());
    std::vector<float> ref_gn(ov::shape_size(input_shape));
    ov::reference::group_normalization(in_f32.data(), scale_f32.data(), bias_f32.data(),
                                       ref_gn.data(), input_shape, num_groups, epsilon);

    for (std::size_t i = 0; i < ref_gn.size(); i++) {
        float sum_val = ref_gn[i] + static_cast<float>(eltwise_vals[i]);
        float expected = std::max(0.0f, sum_val);
        ASSERT_NEAR(static_cast<float>(output_mem_lock[i]), expected, 0.05) << "at index " << i;
    }
}

namespace {

enum class byxf_fused { none, swish, sum_byxf, sum_bfyx, mul_scalar };

struct byxf_shape {
    ov::Shape shape;
    int64_t num_groups;
};

struct byxf_precision {
    data_types dt;
    bool is_dynamic;  // N, H, W dynamic; C static
};

struct byxf_case {
    std::vector<ov::Shape> shapes;  // executed in this order on one network
    int64_t num_groups;
    byxf_precision precision;
    byxf_fused fused = byxf_fused::none;
    padding in_pad = padding();
    float data_offset = 0.0f;
};

// Rounded to the network precision, so the host reference gets the values the GPU gets.
std::vector<float> random_values(tests::random_generator& rg, size_t count, data_types dt, float offset = 0.0f) {
    auto values = rg.generate_random_1d<float>(count, -1, 1);
    for (auto& v : values) {
        v += offset;
        if (dt == data_types::f16) {
            v = static_cast<float>(ov::float16(v));
        }
    }
    return values;
}

memory::ptr make_memory(engine& engine, const layout& l, const std::vector<float>& values) {
    auto mem = engine.allocate_memory(l);
    if (l.data_type == data_types::f16) {
        set_values(mem, std::vector<ov::float16>(values.begin(), values.end()));
    } else {
        set_values(mem, values);
    }
    return mem;
}

// input(bfyx, f32) -> reorder(byxf, dt, in_pad) -> GN [-> fused op] -> reorder(bfyx, f32), checked against
// ov::reference::group_normalization for every shape of the case.
void run_byxf_test(const byxf_case& c, bool is_caching_test = false) {
    constexpr double epsilon = 1e-5;
    constexpr float scale_down = 0.5f;
    auto& engine = get_test_engine();
    tests::random_generator rg{"GroupNormalizationByxfGPUTest"};
    const auto dt = c.precision.dt;
    const auto channels = static_cast<int64_t>(c.shapes.front()[1]);
    const auto in_pshape = c.precision.is_dynamic ? ov::PartialShape{-1, channels, -1, -1} : ov::PartialShape(c.shapes.front());
    const bool has_addend = c.fused == byxf_fused::sum_byxf || c.fused == byxf_fused::sum_bfyx;

    const layout param_layout{ov::PartialShape{1, channels, 1, 1}, dt, format::bfyx};
    const auto scale = random_values(rg, channels, dt);
    const auto bias = random_values(rg, channels, dt);

    topology topology(input_layout("input", layout{in_pshape, data_types::f32, format::bfyx}),
                      data("scale", make_memory(engine, param_layout, scale)),
                      data("bias", make_memory(engine, param_layout, bias)),
                      reorder("input_byxf", input_info("input"), layout{in_pshape, dt, format::byxf, c.in_pad}),
                      group_normalization("gn", input_info("input_byxf"), input_info("scale"), input_info("bias"), c.num_groups, epsilon));
    switch (c.fused) {
    case byxf_fused::none:
        break;
    case byxf_fused::swish:
        topology.add(activation("fused", input_info("gn"), activation_func::swish, {1.0f, 0.0f}));
        break;
    case byxf_fused::sum_byxf:
    case byxf_fused::sum_bfyx:
        topology.add(input_layout("addend", layout{in_pshape, data_types::f32, format::bfyx}));
        topology.add(reorder("addend_dt", input_info("addend"), c.fused == byxf_fused::sum_byxf ? format::byxf : format::bfyx, dt));
        topology.add(eltwise("fused", {input_info("gn"), input_info("addend_dt")}, eltwise_mode::sum));
        break;
    case byxf_fused::mul_scalar:
        topology.add(data("scale_down", make_memory(engine, layout{ov::PartialShape{1, 1, 1, 1}, dt, format::bfyx}, {scale_down})));
        topology.add(eltwise("fused", {input_info("gn"), input_info("scale_down")}, eltwise_mode::prod));
        break;
    }
    topology.add(reorder("output", input_info(c.fused == byxf_fused::none ? "gn" : "fused"), format::bfyx, data_types::f32));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    // impl_types::any keeps validate_impl in the selection; a forced impl type would take the first OCL impl unvalidated.
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"gn", {format::byxf, "", impl_types::any}}}));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    for (const auto& shape : c.shapes) {
        const auto count = ov::shape_size(shape);
        const auto input = random_values(rg, count, dt, c.data_offset);
        auto input_mem = engine.allocate_memory(layout{shape, data_types::f32, format::bfyx});
        set_values(input_mem, input);
        network->set_input_data("input", input_mem);
        std::vector<float> addend;
        if (has_addend) {
            addend = random_values(rg, count, dt);
            auto addend_mem = engine.allocate_memory(layout{shape, data_types::f32, format::bfyx});
            set_values(addend_mem, addend);
            network->set_input_data("addend", addend_mem);
        }
        auto outputs = network->execute();

        const auto gn_inst = network->get_primitive("gn");
        ASSERT_EQ(gn_inst->get_impl()->get_type_info(), byxf_impl_type);
        ASSERT_EQ(gn_inst->get_impl()->is_dynamic(), c.precision.is_dynamic);
        ASSERT_EQ(gn_inst->get_impl_params()->get_input_layout(0).data_padding, c.in_pad);
        if (c.fused != byxf_fused::none) {
            ASSERT_EQ(network->get_executed_primitives().count("fused"), 0);
        }

        std::vector<float> expected(count);
        ov::reference::group_normalization(input.data(), scale.data(), bias.data(), expected.data(), shape, c.num_groups, epsilon);
        for (size_t i = 0; i < count; ++i) {
            switch (c.fused) {
            case byxf_fused::none:
                break;
            case byxf_fused::swish:
                expected[i] /= 1.0f + std::exp(-expected[i]);
                break;
            case byxf_fused::sum_byxf:
            case byxf_fused::sum_bfyx:
                expected[i] += addend[i];
                break;
            case byxf_fused::mul_scalar:
                expected[i] *= scale_down;
                break;
            }
        }

        const float tolerance = dt == data_types::f32 ? 1e-4f : (has_addend ? 2e-2f : 1e-2f);
        ASSERT_EQ(outputs.at("output").get_layout().get_shape(), shape);
        // The output buffer may be larger than the current shape (preallocation for dynamic shapes).
        cldnn::mem_lock<float, mem_lock_type::read> output_lock(outputs.at("output").get_memory(), get_test_stream());
        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(output_lock[i], expected[i], tolerance) << "shape " << shape << " index " << i;
        }
    }
}

using byxf_params = std::tuple<byxf_shape, byxf_precision, byxf_fused, padding>;

class GroupNormalizationByxfGPUTest : public ::testing::TestWithParam<byxf_params> {};

TEST_P(GroupNormalizationByxfGPUTest, random) {
    const auto& [shape, precision, fused, in_pad] = GetParam();
    run_byxf_test({{shape.shape}, shape.num_groups, precision, fused, in_pad});
}

std::string byxf_test_name(const ::testing::TestParamInfo<byxf_params>& info) {
    static const std::array<const char*, 5> fused_names = {"none", "swish", "sum_byxf", "sum_bfyx", "mul_scalar"};
    const auto& [shape, precision, fused, in_pad] = info.param;
    std::ostringstream name;
    name << shape.shape[0] << "x" << shape.shape[1] << "x" << shape.shape[2] << "x" << shape.shape[3] << "_g" << shape.num_groups << "_"
         << ov::element::Type(precision.dt) << (precision.is_dynamic ? "_dynamic_" : "_static_") << fused_names[static_cast<size_t>(fused)];
    if (in_pad) {
        name << "_pad" << in_pad._lower_size[1] << "_" << in_pad._lower_size[2] << "_" << in_pad._lower_size[3];
    }
    return name.str();
}

const std::vector<byxf_precision> byxf_f16_dynamic_f32_static = {{data_types::f16, true}, {data_types::f32, false}};

// SD UNet/VAE channel counts (group size 4..80) and odd C that select VEC 8 (FV 3), 4, 2 and 1.
INSTANTIATE_TEST_SUITE_P(byxf_group_sizes,
                         GroupNormalizationByxfGPUTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<byxf_shape>{{{2, 128, 7, 9}, 32},
                                                                                        {{2, 256, 7, 9}, 32},
                                                                                        {{2, 320, 7, 9}, 32},
                                                                                        {{2, 512, 7, 9}, 32},
                                                                                        {{2, 640, 7, 9}, 32},
                                                                                        {{2, 960, 7, 9}, 32},
                                                                                        {{2, 1280, 7, 9}, 32},
                                                                                        {{2, 1920, 7, 9}, 32},
                                                                                        {{2, 2560, 7, 9}, 32},
                                                                                        {{2, 24, 7, 9}, 4},
                                                                                        {{2, 36, 7, 9}, 6},
                                                                                        {{2, 30, 7, 9}, 10},
                                                                                        {{2, 15, 7, 9}, 5}}),
                                            ::testing::ValuesIn(byxf_f16_dynamic_f32_static),
                                            ::testing::Values(byxf_fused::none),
                                            ::testing::Values(padding())),
                         byxf_test_name);

INSTANTIATE_TEST_SUITE_P(byxf_spatial_batch,
                         GroupNormalizationByxfGPUTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<byxf_shape>{{{1, 320, 1, 1}, 32},
                                                                                        {{2, 320, 7, 9}, 32},
                                                                                        {{16, 320, 7, 9}, 32},
                                                                                        {{3, 320, 64, 64}, 32},
                                                                                        {{1, 320, 128, 128}, 32}}),
                                            ::testing::ValuesIn(byxf_f16_dynamic_f32_static),
                                            ::testing::Values(byxf_fused::none),
                                            ::testing::Values(padding())),
                         byxf_test_name);

// Swish (SD GN + SiLU), eltwise sum with a byxf or a bfyx addend, and the SD2.1 VAE per-tensor ScaleDown multiply.
INSTANTIATE_TEST_SUITE_P(byxf_fused_ops,
                         GroupNormalizationByxfGPUTest,
                         ::testing::Combine(::testing::Values(byxf_shape{{2, 320, 16, 16}, 32}),
                                            ::testing::ValuesIn(std::vector<byxf_precision>{{data_types::f16, false}, {data_types::f16, true}}),
                                            ::testing::Values(byxf_fused::swish, byxf_fused::sum_byxf, byxf_fused::sum_bfyx, byxf_fused::mul_scalar),
                                            ::testing::Values(padding())),
                         byxf_test_name);

INSTANTIATE_TEST_SUITE_P(byxf_sd_shapes,
                         GroupNormalizationByxfGPUTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<byxf_shape>{{{2, 320, 64, 64}, 32},
                                                                                        {{2, 1280, 8, 8}, 32},
                                                                                        {{2, 2560, 16, 16}, 32},
                                                                                        {{1, 512, 64, 64}, 32},
                                                                                        {{1, 128, 256, 256}, 32}}),
                                            ::testing::Values(byxf_precision{data_types::f16, true}),
                                            ::testing::Values(byxf_fused::swish),
                                            ::testing::Values(padding())),
                         byxf_test_name);

INSTANTIATE_TEST_SUITE_P(byxf_padding,
                         GroupNormalizationByxfGPUTest,
                         ::testing::Combine(::testing::Values(byxf_shape{{2, 320, 7, 9}, 32}),
                                            ::testing::Values(byxf_precision{data_types::f16, true}),
                                            ::testing::Values(byxf_fused::none),
                                            ::testing::Values(padding({0, 16, 0, 0}, {0, 0, 0, 0}), padding({0, 0, 1, 1}))),
                         byxf_test_name);

}  // namespace

// Runtime shape changes on one network: dispatch update, internal buffer reallocation and reuse.
TEST(group_normalization, byxf_dynamic_shape_change) {
    run_byxf_test({{{2, 320, 16, 16}, {1, 320, 7, 9}, {16, 320, 8, 8}, {3, 320, 64, 64}, {2, 320, 16, 16}},
                   32,
                   {data_types::f16, true},
                   byxf_fused::swish});
}

// |mean| >> std: the shifted sums keep the variance accurate (plain E[x^2] - E[x]^2 is off by ~4e-3 here).
TEST(group_normalization, byxf_large_mean_offset) {
    run_byxf_test({{{1, 64, 128, 128}}, 32, {data_types::f32, false}, byxf_fused::none, padding(), 100.0f});
}

TEST(group_normalization, byxf_cached_static) {
    run_byxf_test({{{2, 320, 16, 16}}, 32, {data_types::f16, false}, byxf_fused::swish}, true);
}

TEST(group_normalization, byxf_cached_dynamic) {
    run_byxf_test({{{2, 320, 16, 16}, {1, 320, 7, 9}}, 32, {data_types::f16, true}, byxf_fused::swish}, true);
}
