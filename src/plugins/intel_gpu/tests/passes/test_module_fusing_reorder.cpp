// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include "program_helpers.h"
#include "layout_optimizer.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include "intel_gpu/primitives/reorder.hpp"
#include <intel_gpu/primitives/data.hpp>

#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;


static void setting_node(program::ptr prog, const primitive_id& id, layout new_layout) {
    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (node_ptr->id() == id)
            node_ptr->set_output_layout(new_layout);
    }
}

// To test removal of reorder for mixed precision of Onednn conv kernel (conv: u8->fp32)
TEST(test_can_fuse_reorder, reorder_for_mixed_type_convolution_fsv32_onednn)
{
    build_options build_opt;
    topology topology;
#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& engine = get_onednn_test_engine();
#else
    auto& engine = get_test_engine();
#endif

    layout reorder_layout(data_types::u8, format::b_fs_yx_fsv32, {1, 32, 2, 2}, padding({0, }, 0));
    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 2, 2} });
    auto weights = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 2, 2} });
    auto bias = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 1, 1} });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(reorder("reorder_input", "input", format::b_fs_yx_fsv32, data_types::u8));
    topology.add(cldnn::convolution("conv", { "reorder_input" }, { "weights" }, { "bias"}, 1, {1, 1}, {0, 0}, {1, 1}, {1, 32, 2, 2}, data_types::f32, false));
    topology.add(reorder("reorder_conv", "conv", reorder_layout));

    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    layout_optimizer lo = layout_optimizer();
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || node_ptr->id() != "reorder_input")  // target reorder
            continue;

        auto& node = node_ptr->as<reorder>();
        auto& input = node.input();
        for (auto usr : node_ptr->get_users()) {
            EXPECT_EQ(false, lo.can_fuse_reorder(input, *usr, node.input().get_output_layout().format, usr->get_output_layout().format));
        }
    }
}

// To test mixed precision of Cldnn conv kernel (conv: u8->fp32)
TEST(test_can_fuse_reorder, reorder_for_mixed_type_convolution_fsv32_cldnn)
{
    build_options build_opt;
    topology topology;
#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& engine = get_onednn_test_engine();
#else
    auto& engine = get_test_engine();
#endif

    layout reorder_layout(data_types::u8, format::b_fs_yx_fsv32, {1, 32, 2, 2}, padding({0, }, 0));
    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 2, 2} });
    auto weights = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 2, 2} });
    auto bias = engine.allocate_memory({ data_types::u8, format::bfyx, {1, 3, 1, 1} });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(reorder("reorder_input", "input", format::b_fs_yx_fsv32, data_types::u8));
    topology.add(cldnn::convolution("conv", { "reorder_input" }, { "weights" }, { "bias"}, 1, {1, 1}, {0, 0}, {1, 1}, {1, 32, 2, 2}, data_types::f32, false));
    topology.add(reorder("reorder_conv", "conv", reorder_layout));

    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    layout_optimizer lo = layout_optimizer();
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, false);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || node_ptr->id() != "reorder_input")  // target reorder
            continue;

        auto& node = node_ptr->as<reorder>();
        auto& input = node.input();
        for (auto usr : node_ptr->get_users()) {
            EXPECT_EQ(true, lo.can_fuse_reorder(input, *usr, node.input().get_output_layout().format, usr->get_output_layout().format));
        }
    }
}

namespace {
struct reorder_test_param {
    format input_format;
    format output_format;
    data_types input_data_type;
    data_types output_data_type;
    tensor in_shape;
    tensor out_shape;
    tensor weight_shape;
    tensor stride;
    tensor pad;
    data_types weights_type;
    format weights_format;
    bool expected_result;
};

}  // namespace namespace

template<typename T>
class ReorderTest : public ::testing::TestWithParam<T> {
public:
#ifdef ENABLE_ONEDNN_FOR_GPU
    cldnn::engine& engine = get_onednn_test_engine();
#else
    cldnn::engine& engine = get_test_engine();
#endif

    layout get_input_layout(T& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{pad_} };
    }

    bool check_supports_immad() {
        return this->engine.get_device_info().supports_immad;
    }
};

// Not to fuse a reorder if the next conv has deep depth input
class test_fused_reorder_deep_depth : public ReorderTest<reorder_test_param> {};
TEST_P(test_fused_reorder_deep_depth, no_removal_for_deep_depth_conv)
{
    build_options build_opt;
    topology topology;
    auto p = GetParam();

    layout conv_layout(p.input_data_type, p.output_format, p.out_shape, padding({0, }, 0));
    layout reorder_layout(p.output_data_type, p.output_format, p.out_shape, padding({0, }, 0));
    auto input = engine.allocate_memory({ p.input_data_type, p.input_format, p.in_shape });
    auto weights = engine.allocate_memory({ p.input_data_type, p.input_format, p.weight_shape });
    auto bias = engine.allocate_memory({ p.input_data_type, p.input_format, p.weight_shape });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(reorder("reorder_input", "input", p.output_format, p.input_data_type));
    topology.add(cldnn::convolution("conv", { "reorder_input" }, { "weights" }));
    topology.add(reorder("reorder_conv", "conv", reorder_layout));

    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    layout_optimizer lo = layout_optimizer();
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);
    setting_node(prog, "conv", conv_layout);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || node_ptr->id() != "reorder_input")  // target reorder
            continue;

        auto& node = node_ptr->as<reorder>();
        auto& input = node.input();
        for (auto usr : node_ptr->get_users()) {
            EXPECT_EQ(p.expected_result, lo.can_fuse_reorder(input, *usr, node.input().get_output_layout().format, usr->get_output_layout().format));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(testing_deep_depth_conv, test_fused_reorder_deep_depth,
                        ::testing::ValuesIn(std::vector<reorder_test_param>{
                                            reorder_test_param{format::bfyx, format::b_fs_yx_fsv32, data_types::u8, data_types::u8, {1, 32, 8, 8}, {1, 32, 8, 8}, {1, 32, 1, 1},
                                                tensor{1}, tensor{0}, data_types::u8, format::goiyx, false},
                                            reorder_test_param{format::bfyx, format::b_fs_yx_fsv16, data_types::f16, data_types::f16, {1, 32, 8, 8}, {1, 32, 8, 8}, {1, 32, 1, 1},
                                                tensor{1}, tensor{0}, data_types::f16, format::goiyx, false},
                                            reorder_test_param{format::bfyx, format::bs_fs_yx_bsv32_fsv32, data_types::u8, data_types::u8, {32, 32, 8, 8}, {32, 32, 8, 8}, {1, 32, 1, 1},
                                                tensor{1}, tensor{0}, data_types::u8, format::goiyx, false},
                                            reorder_test_param{format::bfyx, format::bs_fs_yx_bsv32_fsv16, data_types::f16, data_types::f16, {32, 32, 8, 8}, {32, 32, 8, 8}, {1, 32, 1, 1},
                                                tensor{1}, tensor{0}, data_types::f16, format::goiyx, false},
                                            }));

// To test removal of reorder for first convolution optimizing in cldnn kernel (shallow input depth to deep output depth)
class test_can_fuse_reorder_cldnn : public ReorderTest<reorder_test_param> {};
TEST_P(test_can_fuse_reorder_cldnn, reorder_for_firstconv_cldnn)
{
    build_options build_opt;
    topology topology;
    auto p = GetParam();

    layout reorder_layout(p.output_data_type, p.output_format, p.out_shape, padding({0, }, 0));
    auto input = engine.allocate_memory({ p.input_data_type, p.input_format, p.in_shape });
    auto weights = engine.allocate_memory({ p.input_data_type, p.input_format, p.weight_shape });
    auto bias = engine.allocate_memory({ p.input_data_type, p.input_format, p.weight_shape });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(reorder("reorder_input", "input", p.output_format, p.input_data_type));
    topology.add(cldnn::convolution("conv2", { "reorder_input" }, { "weights" }, { "bias"}, 1, {1, 1}, {0, 0}, {1, 1}, p.out_shape, p.input_data_type, false));
    topology.add(reorder("reorder_conv", "conv2", reorder_layout));

    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    layout_optimizer lo = layout_optimizer();
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, false);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || node_ptr->id() != "reorder_input")  // target reorder
            continue;

        auto& node = node_ptr->as<reorder>();
        auto& input = node.input();
        for (auto usr : node_ptr->get_users()) {
            EXPECT_EQ(p.expected_result, lo.can_fuse_reorder(input, *usr, node.input().get_output_layout().format, usr->get_output_layout().format));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(testing_can_fuse_reorder_first_conv, test_can_fuse_reorder_cldnn,
                        ::testing::ValuesIn(std::vector<reorder_test_param>{
                                            reorder_test_param{format::bfyx, format::b_fs_yx_fsv32, data_types::u8, data_types::u8, {1, 3, 8, 8}, {1, 32, 8, 8}, {1, 3, 1, 1},
                                                tensor{1}, tensor{0}, data_types::u8, format::goiyx, true},
                                            reorder_test_param{format::bfyx, format::b_fs_yx_fsv16, data_types::f16, data_types::f16, {1, 3, 8, 8}, {1, 32, 8, 8}, {1, 3, 1, 1},
                                                tensor{1}, tensor{0}, data_types::f16, format::goiyx, true},
                                            }));

// To test removal of reorder for first convolution optimizing in onednn kernel (shallow input depth to deep output depth)
class test_can_fuse_reorder_onednn : public ReorderTest<reorder_test_param> {};
TEST_P(test_can_fuse_reorder_onednn, reorder_for_firstconv_onednn)
{
    build_options build_opt;
    topology topology;
    auto p = GetParam();

    layout conv_layout(p.input_data_type, p.output_format, p.out_shape, padding({0, }, 0));
    layout reorder_layout(p.output_data_type, p.output_format, p.out_shape, padding({0, }, 0));
    auto input = engine.allocate_memory({ p.input_data_type, p.input_format, p.in_shape });
    auto weights = engine.allocate_memory({ p.input_data_type, p.input_format, p.weight_shape });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(reorder("reorder_input", "input", p.input_format, p.output_data_type));
    topology.add(reorder("reorder_conv", "reorder_input", p.output_format, p.output_data_type));
    topology.add(cldnn::convolution("conv", { "reorder_input" }, { "weights" }));
    topology.add(reorder("reorder_result", "conv", reorder_layout));

    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    layout_optimizer lo = layout_optimizer();
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);
    setting_node(prog, "conv", conv_layout);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || node_ptr->id() != "reorder_conv")  // target reorder
            continue;

        auto& node = node_ptr->as<reorder>();
        auto& input = node.input();
        for (auto usr : node_ptr->get_users()) {
            EXPECT_EQ(p.expected_result, lo.can_fuse_reorder(input, *usr, node.input().get_output_layout().format, usr->get_output_layout().format));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(testing_can_fuse_reorder_first_conv, test_can_fuse_reorder_onednn,
                        ::testing::ValuesIn(std::vector<reorder_test_param>{
                                            reorder_test_param{format::bs_fs_yx_bsv8_fsv4, format::b_fs_yx_fsv32, data_types::f32, data_types::u8, {1, 3, 8, 8}, {1, 32, 8, 8}, {1, 3, 1, 1},
                                                tensor{1}, tensor{0}, data_types::u8, format::goiyx, true},
                                            reorder_test_param{format::bs_fs_yx_bsv8_fsv2, format::b_fs_yx_fsv16, data_types::f32, data_types::f16, {1, 3, 8, 8}, {1, 32, 8, 8}, {1, 3, 1, 1},
                                                tensor{1}, tensor{0}, data_types::f16, format::goiyx, true},
                                            }));
