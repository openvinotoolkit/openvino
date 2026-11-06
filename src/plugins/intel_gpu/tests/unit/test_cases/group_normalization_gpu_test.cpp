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
#include "openvino/reference/group_normalization.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "ocl_v2/group_normalization_ref.hpp"
#include "ocl_v2/primitive_ocl_base.hpp"

#include <algorithm>
#include <optional>
#include <tuple>


using namespace cldnn;
using namespace ::tests;

namespace {

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

// group_normalization_ref tests.
// The registry tries fsv16_opt and bfyx_opt before ref. Each case gets ref in one of two ways:
// - natural: the input is rejected by both opt impls (bfwzyx, or b_fs_yx_fsv16 with feature lower padding not a
//   multiple of 16), so ref is selected. Works for static and dynamic.
// - swap (static only): the impl selected by the network is replaced by the ref impl. This also allows an output
//   format that differs from the input format, which the graph does not produce for group_normalization.
namespace {

// sum_i32: the second eltwise input is i32, a type gn itself does not accept.
enum class gn_fused { none, relu, sum, sum_relu, sum_i32 };
// Input values. large_mean: uniform in [999, 1001]. constant: every value is 1000.1, so the variance is 0.
// Both make the f32 rounding error of the first-pass mean visible.
enum class gn_dist { uniform, large_mean, constant };

struct gn_ref_params {
    std::string name;
    std::vector<std::vector<int64_t>> shapes;  // Static: shapes[0]. Dynamic: executed in order, with C fixed.
    int64_t groups;
    format in_fmt;
    padding in_pad;
    format out_fmt;  // Differs from in_fmt only with swap.
    padding out_pad;
    data_types in_dt;
    data_types sb_dt;  // scale and bias
    data_types out_dt;
    bool dynamic;
    bool swap;
    bool rank1_sb;  // scale and bias of shape [C] instead of [1, C, 1, 1]
    gn_fused fused;
    gn_dist dist;
};

std::ostream& operator<<(std::ostream& os, const gn_ref_params& p) {
    return os << p.name;
}

// Values exactly representable in dt, so the f32 reference sees the same input as the kernel.
std::vector<float> gn_ref_values(tests::random_generator& rg, size_t count, data_types dt, float lo, float hi) {
    auto values = rg.generate_random_1d<float>(count, lo, hi);
    for (auto& v : values) {
        if (dt == data_types::f16) {
            v = static_cast<float>(ov::float16(v));
        }
    }
    return values;
}

// Converts any layout (blocked, padded, f16 or f32) to a plain f32 vector.
std::vector<float> gn_ref_read_plain_f32(memory::ptr mem) {
    auto& engine = get_test_engine();
    const auto& l = mem->get_layout();
    topology tp(input_layout("in", l), reorder("out", input_info("in"), format::get_default_format(l.get_rank()), data_types::f32));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network net(engine, tp, config);
    net.set_input_data("in", mem);
    auto out = net.execute().at("out").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> ptr(out, get_test_stream());
    return std::vector<float>(ptr.begin(), ptr.end());
}

// The gn node can be renamed when a user is fused into it, so look it up by type.
std::shared_ptr<primitive_inst> gn_ref_find_inst(network& net) {
    for (const auto& id : net.get_all_primitive_ids()) {
        if (id != "_optimized_" && net.get_primitive(id)->type() == group_normalization::type_id()) {
            return net.get_primitive(id);
        }
    }
    return nullptr;
}

// Replaces the impl of a static gn with the ref impl. out_layout, if set, becomes the gn output layout.
void gn_ref_swap_impl(network& net, primitive_inst& inst, const std::optional<layout>& out_layout) {
    ov::intel_gpu::ocl::GroupNormalizationRef ref_manager(shape_types::any);
    ASSERT_TRUE(ref_manager.validate(inst.get_node()));
    auto params = *inst.get_impl_params();
    if (out_layout) {
        params.output_layouts[0] = *out_layout;
    }
    auto ref_impl = ref_manager.create(inst.get_node(), params);
    ASSERT_NE(ref_impl, nullptr);
    ref_impl->set_kernels(net.get_program()->get_kernels_cache().compile(params, ref_impl->get_kernels_source()));
    inst.set_impl(std::move(ref_impl));
    // The internal buffers of the previous impl have a different size.
    inst.allocate_internal_buffers();
    if (out_layout) {
        inst.set_output_memory(net.get_engine().allocate_memory(*out_layout), false);
    }
    inst.set_arguments();
}

// The executed impl is ref, and every stage dispatches uniform work-groups within the device limit.
void gn_ref_check_dispatch(primitive_inst& inst, size_t batch, size_t groups) {
    ASSERT_NE(inst.get_impl(), nullptr);
    ASSERT_NE(inst.get_impl()->get_type_info().find("GroupNormalizationRefImpl"), std::string::npos) << inst.get_impl()->get_type_info();
    auto impl = dynamic_cast<ov::intel_gpu::ocl::PrimitiveImplOCL*>(inst.get_impl());
    ASSERT_NE(impl, nullptr);
    ASSERT_EQ(impl->_stages.size(), 2u);
    const size_t max_wg = inst.get_network().get_engine().get_device_info().max_work_group_size;
    for (size_t s = 0; s < impl->_stages.size(); s++) {
        const auto& wgs = impl->_stages[s]->kd.params.workGroups;
        ASSERT_EQ(wgs.global.size(), 3u);
        ASSERT_EQ(wgs.local.size(), 3u);
        size_t wg_size = 1;
        for (size_t i = 0; i < wgs.global.size(); i++) {
            ASSERT_NE(wgs.local[i], 0u) << "stage " << s << " axis " << i;
            ASSERT_EQ(wgs.global[i] % wgs.local[i], 0u) << "stage " << s << " axis " << i << ": gws=" << wgs.global[i] << " lws=" << wgs.local[i];
            wg_size *= wgs.local[i];
        }
        ASSERT_LE(wg_size, max_wg) << "stage " << s;
    }
    // calc_stats: one work-group per (batch, group).
    const auto& wgs = impl->_stages[0]->kd.params.workGroups;
    ASSERT_EQ(wgs.global[0], wgs.local[0] * groups);
    ASSERT_EQ(wgs.global[1], std::max<size_t>(batch, 1));
    ASSERT_EQ(wgs.local[1], 1u);
    ASSERT_EQ(wgs.local[2], 1u);
}

class group_normalization_ref_gpu_test : public ::testing::TestWithParam<gn_ref_params> {};

TEST_P(group_normalization_ref_gpu_test, accuracy) {
    const auto& p = GetParam();
    auto& engine = get_test_engine();
    const float epsilon = 0.0025f;
    const size_t rank = p.shapes[0].size();
    const int64_t channels = p.shapes[0][1];
    const auto plain_fmt = format::get_default_format(rank);
    const bool has_sum = p.fused == gn_fused::sum || p.fused == gn_fused::sum_relu || p.fused == gn_fused::sum_i32;
    const auto eltw_dt = p.fused == gn_fused::sum_i32 ? data_types::i32 : data_types::f32;
    const bool has_relu = p.fused == gn_fused::relu || p.fused == gn_fused::sum_relu;
    const bool change_out_fmt = p.out_fmt != p.in_fmt;
    ASSERT_TRUE(!p.dynamic || !p.swap) << "swap works only for static shapes";
    ASSERT_TRUE(!change_out_fmt || (p.swap && p.fused == gn_fused::none)) << "a different output format needs swap without fused ops";

    const auto data_pshape = p.dynamic ? ov::PartialShape::dynamic(rank) : ov::PartialShape(p.shapes[0]);
    const layout sb_layout{p.rank1_sb ? ov::PartialShape{channels} : ov::PartialShape{1, channels, 1, 1}, data_types::f32, format::bfyx};

    topology tp;
    tp.add(input_layout{"data", layout{data_pshape, data_types::f32, plain_fmt}});
    tp.add(input_layout{"scale", sb_layout});
    tp.add(input_layout{"bias", sb_layout});
    // gn receives the input in in_fmt / in_dt / in_pad.
    tp.add(reorder{"data_r", input_info("data"), layout{ov::PartialShape(p.shapes[0]), p.in_dt, p.in_fmt, p.in_pad}});
    input_info scale_in("scale");
    input_info bias_in("bias");
    if (p.sb_dt != data_types::f32) {
        tp.add(reorder{"scale_r", scale_in, format::bfyx, p.sb_dt});
        tp.add(reorder{"bias_r", bias_in, format::bfyx, p.sb_dt});
        scale_in = input_info("scale_r");
        bias_in = input_info("bias_r");
    }
    group_normalization gn{"gn", input_info("data_r"), scale_in, bias_in, p.groups, epsilon};
    if (p.out_dt != p.in_dt) {
        gn.output_data_types = {optional_data_type{p.out_dt}};
    }
    gn.output_paddings = {p.out_pad};
    tp.add(gn);
    primitive_id last = "gn";
    if (has_sum) {
        tp.add(input_layout{"eltw", layout{data_pshape, eltw_dt, plain_fmt}});
        tp.add(eltwise{"sum", {input_info(last), input_info("eltw")}, eltwise_mode::sum});
        last = "sum";
    }
    if (has_relu) {
        tp.add(activation{"relu", input_info(last), activation_func::relu});
        last = "relu";
    }
    if (p.fused != gn_fused::none) {
        // A network output is not fused into its producer, so the fused chain needs a user.
        tp.add(reorder{"output", input_info(last), plain_fmt, data_types::f32});
    }

    auto config = get_test_default_config(engine);
    // Post-op fusion and the dynamic flow need optimize_data. Ranks other than 4/5 and rank-1 scale need the new shape infer.
    config.set_property(ov::intel_gpu::optimize_data(p.dynamic || p.fused != gn_fused::none));
    if (p.dynamic || p.rank1_sb || rank < 4 || rank > 5) {
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    }
    network net(engine, tp, config);

    auto inst = gn_ref_find_inst(net);
    ASSERT_NE(inst, nullptr);
    const auto in_l = inst->get_input_layout(0);
    ASSERT_EQ(in_l.format, p.in_fmt);
    ASSERT_EQ(in_l.data_type, p.in_dt);
    ASSERT_EQ(in_l.data_padding, p.in_pad);
    ASSERT_EQ(inst->get_node().get_fused_primitives().size(), static_cast<size_t>(has_sum) + static_cast<size_t>(has_relu));
    if (p.swap) {
        std::optional<layout> out_layout;
        if (change_out_fmt) {
            out_layout = layout{ov::PartialShape(p.shapes[0]), p.out_dt, p.out_fmt, p.out_pad};
        }
        ASSERT_NO_FATAL_FAILURE(gn_ref_swap_impl(net, *inst, out_layout));
    }

    tests::random_generator rg{"group_normalization_ref_gpu_test_" + p.name};
    const auto scale = gn_ref_values(rg, channels, p.sb_dt, -1.0f, 1.0f);
    const auto bias = gn_ref_values(rg, channels, p.sb_dt, -0.5f, 0.5f);
    auto scale_mem = engine.allocate_memory(sb_layout);
    auto bias_mem = engine.allocate_memory(sb_layout);
    set_values(scale_mem, scale);
    set_values(bias_mem, bias);

    for (const auto& dims : p.shapes) {
        const ov::Shape shape(dims.begin(), dims.end());
        const size_t count = ov::shape_size(shape);
        ASSERT_EQ(dims[1], channels);
        std::vector<float> data;
        if (p.dist == gn_dist::large_mean) {
            data = gn_ref_values(rg, count, p.in_dt, 999.0f, 1001.0f);
        } else if (p.dist == gn_dist::constant) {
            data.assign(count, 1000.1f);
        } else {
            data = gn_ref_values(rg, count, p.in_dt, -1.0f, 1.0f);
        }
        const float eltw_range = eltw_dt == data_types::i32 ? 3.0f : 1.0f;
        auto eltw = rg.generate_random_1d<float>(has_sum ? count : 0, -eltw_range, eltw_range);
        if (eltw_dt == data_types::i32) {
            std::transform(eltw.begin(), eltw.end(), eltw.begin(), [](float v) {
                return std::round(v);
            });
        }

        const layout data_l{shape, data_types::f32, plain_fmt};
        auto data_mem = engine.allocate_memory(data_l);
        set_values(data_mem, data);
        net.set_input_data("data", data_mem);
        net.set_input_data("scale", scale_mem);
        net.set_input_data("bias", bias_mem);
        if (has_sum) {
            auto eltw_mem = engine.allocate_memory(layout{shape, eltw_dt, plain_fmt});
            if (eltw_dt == data_types::i32) {
                set_values(eltw_mem, std::vector<int32_t>(eltw.begin(), eltw.end()));
            } else {
                set_values(eltw_mem, eltw);
            }
            net.set_input_data("eltw", eltw_mem);
        }
        net.execute();
        net.get_stream().finish();
        // The runtime skips a primitive with an empty output and keeps its previous output memory and dispatch.
        // The empty run only has to execute, and the next run has to be correct.
        if (count == 0) {
            continue;
        }
        // Read the gn output itself, which includes the fused ops.
        auto out_mem = inst->output_memory_ptr();

        const auto& out_l = out_mem->get_layout();
        ASSERT_EQ(out_l.format, p.out_fmt);
        ASSERT_EQ(out_l.data_type, p.out_dt);
        ASSERT_EQ(out_l.data_padding, p.out_pad);
        ASSERT_EQ(out_l.get_shape(), shape);
        ASSERT_NO_FATAL_FAILURE(gn_ref_check_dispatch(*inst, dims[0], p.groups));

        std::vector<float> expected(count);
        ov::reference::group_normalization(data.data(), scale.data(), bias.data(), expected.data(), shape, p.groups, epsilon);
        for (size_t i = 0; i < count; i++) {
            if (has_sum) {
                expected[i] += eltw[i];
            }
            if (has_relu) {
                expected[i] = std::max(expected[i], 0.0f);
            }
        }
        const float tolerance = p.out_dt == data_types::f16 ? 0.02f : 2e-4f;

        const auto actual = gn_ref_read_plain_f32(out_mem);
        ASSERT_EQ(actual.size(), count);
        for (size_t i = 0; i < count; i++) {
            ASSERT_NEAR(actual[i], expected[i], tolerance) << "shape " << shape << " i=" << i;
        }
    }
}

const auto dt_f32 = data_types::f32;
const auto dt_f16 = data_types::f16;
// Feature lower padding is not a multiple of 16, so fsv16_opt rejects b_fs_yx_fsv16.
const padding fsv16_odd_pad({0, 1, 1, 2}, {0, 3, 2, 1});

// clang-format off
const std::vector<gn_ref_params> gn_ref_cases = {
    // name, shapes, G, in_fmt, in_pad, out_fmt, out_pad, in_dt, sb_dt, out_dt, dynamic, swap, rank1_sb, fused, dist
    // Static, swap: formats and inputs that an opt impl accepts.
    {"bfyx_f32", {{2, 32, 5, 7}}, 4, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_f16_g1_odd_spatial", {{2, 16, 9, 7}}, 1, format::bfyx, padding(), format::bfyx, padding(), dt_f16, dt_f16, dt_f16, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_g_eq_c_n_lt_lws", {{3, 24, 3, 5}}, 24, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f16, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_large_n", {{1, 16, 256, 256}}, 1, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_large_mean", {{2, 32, 16, 16}}, 2, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::large_mean},
    // 1M elements per group: the f32 sum of the first pass has a large rounding error.
    {"bfyx_large_n_large_mean", {{1, 16, 256, 256}}, 1, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::large_mean},
    {"bfyx_large_n_constant", {{1, 16, 256, 256}}, 1, format::bfyx, padding(), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::constant},
    {"bfyx_spatial_pad_1x32x1x1", {{1, 32, 1, 1}}, 8, format::bfyx, padding({0, 0, 1, 2}, {0, 0, 2, 1}), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_rank2_feature_pad", {{1, 40}}, 5, format::bfyx, padding({0, 2}, {0, 3}), format::bfyx, padding(), dt_f32, dt_f32, dt_f32, false, true, true, gn_fused::none, gn_dist::uniform},
    {"bfyx_fused_sum", {{2, 32, 5, 5}}, 8, format::bfyx, padding(), format::bfyx, padding(), dt_f16, dt_f16, dt_f16, false, true, false, gn_fused::sum, gn_dist::uniform},
    {"bfzyx_f32_out_pad", {{2, 16, 3, 5, 7}}, 4, format::bfzyx, padding(), format::bfzyx, padding({0, 0, 1, 0, 1}, {0, 0, 0, 2, 0}), dt_f32, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfzyx_f16_in_pad_fused_relu", {{1, 32, 4, 3, 5}}, 8, format::bfzyx, padding({0, 0, 1, 1, 0}, {0, 0, 0, 1, 2}), format::bfzyx, padding(), dt_f16, dt_f32, dt_f16, false, true, false, gn_fused::relu, gn_dist::uniform},
    {"fsv16_f16", {{3, 48, 6, 6}}, 6, format::b_fs_yx_fsv16, padding(), format::b_fs_yx_fsv16, padding(), dt_f16, dt_f16, dt_f16, false, true, false, gn_fused::none, gn_dist::uniform},
    {"fsv16_f32_f16_sb_g1", {{2, 32, 5, 3}}, 1, format::b_fs_yx_fsv16, padding(), format::b_fs_yx_fsv16, padding(), dt_f32, dt_f16, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_batch_pad_f16_to_f32", {{3, 24, 5, 5}}, 3, format::bfyx, padding({1, 0, 0, 0}, {2, 0, 0, 0}), format::bfyx, padding(), dt_f16, dt_f32, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"rank3_bfyx_f16", {{4, 40, 33}}, 8, format::bfyx, padding(), format::bfyx, padding(), dt_f16, dt_f16, dt_f16, false, true, true, gn_fused::none, gn_dist::uniform},
    // Static, swap: the output format differs from the input format.
    {"fsv16_to_bfyx_f16_to_f32", {{2, 32, 5, 5}}, 8, format::b_fs_yx_fsv16, padding(), format::bfyx, padding(), dt_f16, dt_f16, dt_f32, false, true, false, gn_fused::none, gn_dist::uniform},
    {"bfyx_to_fsv16_f32_to_f16_out_pad", {{2, 40, 3, 7}}, 5, format::bfyx, padding(), format::b_fs_yx_fsv16, padding({0, 0, 1, 1}, {0, 5, 0, 0}), dt_f32, dt_f32, dt_f16, false, true, false, gn_fused::none, gn_dist::uniform},
    // Static, natural selection.
    {"fsv16_odd_pad_f32", {{1, 64, 4, 4}}, 32, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f32, dt_f32, false, false, false, gn_fused::none, gn_dist::uniform},
    {"fsv16_odd_pad_f16_sum_relu", {{1, 64, 4, 4}}, 32, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f16, dt_f16, dt_f16, false, false, false, gn_fused::sum_relu, gn_dist::uniform},
    // B * G = 1056 > max_wg on both devices.
    {"fsv16_odd_pad_constant", {{2, 64, 64, 64}}, 2, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f32, dt_f32, false, false, false, gn_fused::none, gn_dist::constant},
    {"fsv16_odd_pad_b33_g32", {{33, 64, 4, 4}}, 32, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f32, dt_f32, false, false, false, gn_fused::none, gn_dist::uniform},
    {"fsv16_odd_pad_f32_f16_sb", {{2, 12, 5, 5}}, 2, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f16, dt_f32, false, false, false, gn_fused::none, gn_dist::uniform},
    {"rank3_fsv16_feature_pad_f16_to_f32", {{2, 40, 33}}, 8, format::b_fs_yx_fsv16, padding({0, 3, 0, 0}, {0, 1, 0, 0}), format::b_fs_yx_fsv16, padding(), dt_f16, dt_f32, dt_f32, false, false, false, gn_fused::none, gn_dist::uniform},
    {"rank6_f32_rank1_f16_sb_sum_relu", {{2, 24, 3, 4, 5, 6}}, 6, format::bfwzyx, padding(), format::bfwzyx, padding(), dt_f32, dt_f16, dt_f32, false, false, true, gn_fused::sum_relu, gn_dist::uniform},
    // The fused eltwise input is i32. validate must not check the dependencies of fused ops.
    {"rank6_f32_fused_sum_i32", {{2, 24, 2, 2, 2, 2}}, 6, format::bfwzyx, padding(), format::bfwzyx, padding(), dt_f32, dt_f32, dt_f32, false, false, false, gn_fused::sum_i32, gn_dist::uniform},
    // Dynamic, natural selection. Executed once per shape. Dynamic bfyx/bfzyx always go to bfyx_opt.
    {"dyn_fsv16_odd_pad_large_mean", {{2, 64, 8, 8}, {1, 64, 16, 16}, {3, 64, 5, 7}}, 4, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f32, dt_f32, true, false, false, gn_fused::none, gn_dist::large_mean},
    // B grows (internal buffers are reallocated), and B * G = 1056 > max_wg in the last run.
    {"dyn_fsv16_odd_pad_f16_sum_relu_b33_g32", {{17, 64, 3, 3}, {4, 64, 5, 5}, {33, 64, 4, 4}}, 32, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f16, dt_f16, dt_f16, true, false, false, gn_fused::sum_relu, gn_dist::uniform},
    {"dyn_fsv16_odd_pad_f32_to_f16_g_eq_c", {{2, 16, 3, 3}, {5, 16, 1, 7}}, 16, format::b_fs_yx_fsv16, fsv16_odd_pad, format::b_fs_yx_fsv16, padding(), dt_f32, dt_f16, dt_f16, true, false, false, gn_fused::none, gn_dist::uniform},
    {"dyn_rank3_fsv16_feature_pad_f16_to_f32", {{2, 40, 17}, {3, 40, 9}}, 8, format::b_fs_yx_fsv16, padding({0, 3, 0, 0}, {0, 1, 0, 0}), format::b_fs_yx_fsv16, padding(), dt_f16, dt_f32, dt_f32, true, false, false, gn_fused::none, gn_dist::uniform},
    {"dyn_rank2_fsv16_feature_pad_rank1_sb_relu", {{3, 40}, {7, 40}}, 8, format::b_fs_yx_fsv16, padding({0, 3, 0, 0}, {0, 1, 0, 0}), format::b_fs_yx_fsv16, padding(), dt_f32, dt_f32, dt_f32, true, false, true, gn_fused::relu, gn_dist::uniform},
    // Empty batch in the middle run.
    {"dyn_rank6_f16_empty_batch", {{2, 24, 2, 2, 3, 3}, {0, 24, 2, 2, 3, 3}, {3, 24, 1, 2, 3, 4}}, 6, format::bfwzyx, padding(), format::bfwzyx, padding(), dt_f16, dt_f16, dt_f16, true, false, false, gn_fused::none, gn_dist::uniform},
    {"dyn_rank6_f16_to_f32_g1", {{2, 12, 2, 3, 3, 5}, {3, 12, 1, 2, 3, 4}}, 1, format::bfwzyx, padding(), format::bfwzyx, padding(), dt_f16, dt_f32, dt_f32, true, false, false, gn_fused::none, gn_dist::uniform},
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(group_normalization_ref,
                         group_normalization_ref_gpu_test,
                         ::testing::ValuesIn(gn_ref_cases),
                         [](const ::testing::TestParamInfo<gn_ref_params>& info) {
                             return info.param.name;
                         });

// Builds a gn node with the given input layout and data types, and returns whether the ref impl accepts it.
bool gn_ref_validate(const ov::PartialShape& shape, format fmt, data_types in_dt, data_types sb_dt, data_types out_dt) {
    auto& engine = get_test_engine();
    const layout in_l{shape, in_dt, fmt};
    const layout sb_l{ov::PartialShape{1, shape[1], 1, 1}, sb_dt, format::bfyx};
    program prog(engine);
    auto& data_node = prog.get_or_create(std::make_shared<input_layout>("data", in_l));
    auto& scale_node = prog.get_or_create(std::make_shared<input_layout>("scale", sb_l));
    auto& bias_node = prog.get_or_create(std::make_shared<input_layout>("bias", sb_l));
    auto gn = std::make_shared<group_normalization>("gn", input_info("data"), input_info("scale"), input_info("bias"), 8, 1e-5);
    gn->output_data_types = {optional_data_type{out_dt}};
    auto& gn_node = prog.get_or_create(gn);
    program_wrapper::add_connection(prog, data_node, gn_node);
    program_wrapper::add_connection(prog, scale_node, gn_node);
    program_wrapper::add_connection(prog, bias_node, gn_node);
    gn_node.recalc_output_layout();
    EXPECT_EQ(gn_node.get_output_layout().format, fmt);
    EXPECT_EQ(gn_node.get_output_layout().data_type, out_dt);

    ov::intel_gpu::ocl::GroupNormalizationRef ref_manager(shape_types::any);
    return ref_manager.validate(gn_node);
}

// The kernel handles blocked formats, but validate accepts only the formats of the opt impls (plus bfwzyx).
// Otherwise format propagation keeps gn in a blocked format and ref replaces the reorder + opt impl path.
TEST(group_normalization_ref, validate_formats) {
    const std::vector<std::tuple<ov::PartialShape, format, bool>> cases = {
        {{2, 32, 4, 4}, format::bfyx, true},
        {{2, 32, 4, 4}, format::b_fs_yx_fsv16, true},
        {{2, 32, 4, 4}, format::bs_fs_yx_bsv32_fsv16, false},
        {{2, 32, 4, 4}, format::b_fs_yx_fsv32, false},
        {{2, 32, 4, 4}, format::byxf, false},
        {{2, 32, 3, 4, 4}, format::bfzyx, true},
        {{2, 32, 3, 4, 4}, format::b_fs_zyx_fsv16, false},
        {{2, 32, 2, 3, 4, 4}, format::bfwzyx, true},
        // The format dimension does not match the rank.
        {{2, 32, 4, 4}, format::bfzyx, false},
    };
    for (const auto& [shape, fmt, expected] : cases) {
        EXPECT_EQ(gn_ref_validate(shape, fmt, data_types::f16, data_types::f16, data_types::f16), expected) << shape << " " << fmt;
    }
}

// GroupNormalization-12 defines data, scale, bias and output as floating point types.
// validate accepts f16 and f32 for each of them, and scale/bias may differ from data.
TEST(group_normalization_ref, validate_data_types) {
    const auto f32 = data_types::f32;
    const auto f16 = data_types::f16;
    const auto i8 = data_types::i8;
    const auto u8 = data_types::u8;
    // in, scale/bias, out, expected
    const std::vector<std::tuple<data_types, data_types, data_types, bool>> cases = {
        {f32, f32, f32, true},
        {f16, f16, f16, true},
        {f16, f32, f16, true},
        {f32, f16, f32, true},
        {f16, f16, f32, true},
        {i8, f16, f16, false},
        {u8, f32, f32, false},
        {f16, i8, f16, false},
        {f32, u8, f32, false},
        {f16, f16, i8, false},
        {f32, f32, u8, false},
    };
    for (const auto& [in_dt, sb_dt, out_dt, expected] : cases) {
        EXPECT_EQ(gn_ref_validate({2, 32, 4, 4}, format::bfyx, in_dt, sb_dt, out_dt), expected) << in_dt << " " << sb_dt << " " << out_dt;
    }
}

}  // anonymous namespace
