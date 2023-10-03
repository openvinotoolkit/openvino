// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/binary_convolution.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

struct binary_convolution_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class BinaryConvolutionFusingTest : public BaseFusingTest<binary_convolution_test_params> {
public:
    void execute(binary_convolution_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "conv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(binary_convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(binary_convolution_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

} // namespace

#define CASE_BIN_CONV1 { 1, 16, 4, 5 }, { 1, 16, 4, 5 }, { 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u1, format::b_fs_yx_32fp, data_types::u1, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV2 { 1, 16, 4, 5 }, { 1, 30, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u1, format::b_fs_yx_32fp, data_types::u1, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV3 { 1, 184, 12, 21 }, { 1, 224, 12, 21 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u1, format::b_fs_yx_32fp, data_types::u1, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------------- binary convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_bin_activation : public BinaryConvolutionFusingTest {};
TEST_P(conv_bin_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        activation("activation", input_info("bin_conv_prim"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_activation, ::testing::ValuesIn(std::vector<binary_convolution_test_params>{
    binary_convolution_test_params{ CASE_BIN_CONV1, 2, 3 },
}));

class conv_bin_scale_activation : public BinaryConvolutionFusingTest {};
TEST_P(conv_bin_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        eltwise("scale", { input_info("bin_conv_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_activation, ::testing::ValuesIn(std::vector<binary_convolution_test_params>{
    binary_convolution_test_params{ CASE_BIN_CONV1, 2, 4 },
    binary_convolution_test_params{ CASE_BIN_CONV2, 2, 4 },
}));

class conv_bin_quantize_bin : public BinaryConvolutionFusingTest {};
TEST_P(conv_bin_quantize_bin, channel_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("in_lo", in_thresh),
        data("in_hi", in_thresh),
        data("out_lo", get_mem(get_per_channel_layout(p), -1)),
        data("out_hi", get_mem(get_per_channel_layout(p),  1)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        quantize("quantize_data", input_info("bin_conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 2, data_types::u1),
        reorder("reorder_bfyx", input_info("quantize_data"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_quantize_bin, blob_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_single_element_layout(p), min_random, max_random);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("in_lo", in_thresh),
        data("in_hi", in_thresh),
        data("out_lo", get_mem(get_single_element_layout(p), -1)),
        data("out_hi", get_mem(get_single_element_layout(p), 1)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        quantize("quantize_data", input_info("bin_conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 2, data_types::u1),
        reorder("reorder_bfyx", input_info("quantize_data"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_quantize_bin, ::testing::ValuesIn(std::vector<binary_convolution_test_params>{
    binary_convolution_test_params{ CASE_BIN_CONV1, 2, 3 },
    binary_convolution_test_params{ CASE_BIN_CONV2, 2, 3 },
}));

class conv_bin_scale_conv_dw : public BinaryConvolutionFusingTest {};
TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };

    ov::Strides dw_stride = {2, 2};
    ov::Strides dw_dilation = {1, 1};
    ov::CoordinateDiff dw_pad = p.pad;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        eltwise("scale", { input_info("bin_conv_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        convolution("conv_dw", input_info("scale"), "weights_dw", "", p.out_shape.feature[0], dw_stride, dw_dilation, dw_pad, dw_pad, true),
        reorder("reorder_bfyx", input_info("conv_dw"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };

    ov::Strides dw_stride = {1, 1};
    ov::Strides dw_dilation = {1, 1};
    ov::CoordinateDiff dw_pad = p.pad;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        eltwise("scale", { input_info("bin_conv_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        convolution("conv_dw", input_info("scale"), "weights_dw", "", p.out_shape.feature[0], dw_stride, dw_dilation, dw_pad, dw_pad, true),
        reorder("reorder_bfyx", input_info("conv_dw"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_conv_dw, ::testing::ValuesIn(std::vector<binary_convolution_test_params>{
    binary_convolution_test_params{ CASE_BIN_CONV2, 3, 4 },
    binary_convolution_test_params{ CASE_BIN_CONV3, 3, 4 },
}));

class conv_bin_scale_conv_dw_prelu : public BinaryConvolutionFusingTest {};
TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };

    ov::Strides dw_stride = {2, 2};
    ov::Strides dw_dilation = {1, 1};
    ov::CoordinateDiff dw_pad = p.pad;
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        eltwise("scale", { input_info("bin_conv_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        convolution("conv_dw", input_info("scale"), "weights_dw", "", p.out_shape.feature[0], dw_stride, dw_dilation, dw_pad, dw_pad, true),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        activation("activation", input_info("conv_dw"), "slope_data", activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };

    ov::Strides dw_stride = {1, 1};
    ov::Strides dw_dilation = {1, 1};
    ov::CoordinateDiff dw_pad = p.pad;
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        binary_convolution("bin_conv_prim", input_info("input"), { "weights" }, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
        eltwise("scale", { input_info("bin_conv_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        convolution("conv_dw", input_info("scale"), "weights_dw", "", p.out_shape.feature[0], dw_stride, dw_dilation, dw_pad, dw_pad, true),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        activation("activation", input_info("conv_dw"), "slope_data", activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_conv_dw_prelu, ::testing::ValuesIn(std::vector<binary_convolution_test_params>{
    binary_convolution_test_params{ CASE_BIN_CONV2, 3, 5 },
    binary_convolution_test_params{ CASE_BIN_CONV3, 3, 5 },
}));
