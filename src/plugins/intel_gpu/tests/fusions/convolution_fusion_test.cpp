// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/binary_convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/permute.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

struct convolution_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
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

struct bc_force_kernel_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

struct conv_eltw_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor eltw_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
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

class ConvFusingTest : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
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

    layout get_input_layout(convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class ConvReorderFusingTest : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class ConvEltwTest : public ::BaseFusingTest<conv_eltw_test_params> {
public:

    void execute(conv_eltw_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_prim = [](primitive_info& p) -> bool {
            // Add more ids when needed
            if (p.original_id == "deconv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_prim);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(conv_eltw_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(conv_eltw_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class ConvFusingForceKernelTest : public BaseFusingTest<bc_force_kernel_params> {
    public:
    void execute(bc_force_kernel_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        build_options options;
        options.set_option(build_option::optimize_data(true));
        implementation_desc conv_impl = { p.input_format, p.kernel_name };
        options.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, options);
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

    layout get_input_layout(bc_force_kernel_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(bc_force_kernel_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
class WeightsPrimitiveFusingTestOneDNN : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_imad)
            return;

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        auto impl_forcing_bo = bo_fused.get<build_option_type::force_implementations>();
        const auto& impl_forcing = impl_forcing_bo->forcing;

        auto forcing_format = p.input_format;
        for (auto& forcing : impl_forcing) {
            if (forcing.first == "conv_prim") {
                forcing_format = forcing.second.output_format;
            }
        }

        implementation_desc conv_impl = { forcing_format, "", impl_types::onednn };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
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

    layout get_input_layout(convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

}  // namespace

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_FP32_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_5 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_6 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_7 { 1, 16, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_8 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_9 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_10 { 32, 16, 4, 5, 4 }, { 32, 32, 4, 5, 4 }, { 1, 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_11 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_12 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_13 { 1, 16, 18, 5, 4 }, { 1, 16, 16, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_14 { 1, 3, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx


#define CASE_CONV_FP16_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_5 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::bfyx, data_types::i8, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_6 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_7 { 1, 16, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_8 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_9 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_10 { 32, 16, 4, 5, 4 }, { 32, 32, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_11 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_12 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_13 { 16, 32, 4, 5 }, { 16, 64, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f16, format::fs_b_yx_fsv32, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_CONV_U8S8_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_2 { 1, 15, 5, 5 }, { 1, 30, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_4 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_5 { 1, 16, 5, 5 }, { 1, 32, 5, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_6 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_7 { 1, 64, 7, 7 }, { 1, 32, 7, 7 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_8 { 1, 3, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_9 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_10 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_11 { 32, 15, 4, 5 }, { 32, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_12 { 32, 15, 5, 5 }, { 32, 30, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_13 { 32, 16, 4, 5 }, { 32, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_14 { 32, 17, 4, 5 }, { 32, 17, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_15 { 1, 15, 2, 2 }, { 1, 30, 1, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_CONV_S8S8_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_2 { 1, 15, 5, 5 }, { 1, 30, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_4 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_5 { 1, 16, 5, 5 }, { 1, 32, 5, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_6 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_7  { 1, 64, 7, 7 }, { 1, 32, 7, 7 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_8 { 1, 3, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_9 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_10 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_11 { 1, 4, 1280, 720 }, { 1, 4, 1280, 720 }, { 1, 1, 5, 5 }, tensor{ 1 }, tensor{ { 0, 0, 2, 2 }, 0 }, tensor{ 1 }, 1, data_types::i8, format::b_fs_yx_fsv4, data_types::i8, format::os_is_yx_osv16_isv4, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_12 { 32, 15, 4, 5 }, { 32, 30, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_13 { 32, 15, 5, 5 }, { 32, 30, 3, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_14 { 32, 16, 4, 5 }, { 32, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_15 { 32, 17, 4, 5 }, { 32, 17, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx

#define CASE_CONV3D_U8S8_1 { 1, 15, 5, 4, 5 }, { 1, 30, 3, 2, 3 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_2 { 1, 15, 5, 5, 5 }, { 1, 30, 3, 3, 3 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_3 { 1, 16, 5, 4, 5 }, { 1, 32, 5, 4, 5 }, { 1, 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_4 { 1, 17, 5, 4, 5 }, { 1, 17, 5, 4, 5 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 1 }, 0 }, tensor{ 1 }, 17, data_types::u8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_5 { 1, 3, 5, 4, 5 },  { 1, 32, 5, 4, 5 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 1 }, 0 }, tensor{ 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_CONV3D_S8S8_1 { 1, 15, 5, 4, 5 }, { 1, 30, 3, 2, 3 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_2 { 1, 15, 5, 5, 5 }, { 1, 30, 3, 3, 3 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_3 { 1, 16, 5, 4, 5 }, { 1, 32, 5, 4, 5 }, { 1, 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_4 { 1, 17, 5, 4, 5 }, { 1, 17, 5, 4, 5 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 1 }, 0 }, tensor{ 1 }, 17, data_types::i8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_5 { 1, 3, 5, 4, 5 },  { 1, 18, 5, 4, 5 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 1 }, 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

// in_shape; out_shape; eltw_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_ELTW_FP32_1 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 32, 1, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 1, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 32, 1, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_5 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 32, 2, 1, 1 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_6 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 16, 2, 1, 1 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_7 { 1, 16, 3, 5 }, { 1, 32, 1, 3 }, { 1, 32, 3, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_8 { 1, 32, 3, 5, 4 }, { 1, 16, 1, 3, 2 }, { 1, 1, 2, 1, 1 }, { 1, 1, 3, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx

#define CASE_CONV_ELTW_i8_1 { 1, 16, 3, 5 }, { 1, 32, 1, 3 }, { 1, 32, 3, 1 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_2 { 1, 16, 3, 5, 3 }, { 1, 32, 2, 4, 2 }, { 1, 1, 2, 4, 2 }, { 1, 1, 2, 2, 2 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_3 { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_4 { 1, 16, 1, 4 }, { 1, 16, 1, 2 }, { 1, 16, 1, 1 }, { 1, 1, 1, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_5 { 1, 16, 1, 4, 1 }, { 1, 16, 1, 2, 1 }, { 1, 16, 2, 1, 1 }, { 1, 1, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx

#define CASE_BIN_CONV1 { 1, 16, 4, 5 }, { 1, 16, 4, 5 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ { 0, 0, 1, 1, 0, 0 }, 0 }, tensor{ 1 }, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV2 { 1, 16, 4, 5 }, { 1, 30, 4, 5 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV3 { 1, 184, 12, 21 }, { 1, 224, 12, 21 }, { 1, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx

#define CASE_FC_FP32_1 { 1, 1, 3, 1 }, { 1, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 { 2, 1, 3, 1 }, { 2, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::yxfb, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 { 2, 32, 1, 1 }, { 2, 16, 1, 1 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_1 { 5, 3, 1, 3 }, { 5, 3, 1, 5 }, { 5, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_2 { 2, 1, 1, 1 }, { 2, 1, 1, 32 }, { 32, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_3 { 2, 32, 1, 32 }, { 2, 32, 1, 16 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 { 1, 1, 3, 1 }, { 1, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 { 2, 1, 3, 1 }, { 2, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 { 2, 32, 1, 1 }, { 2, 16, 1, 1 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 { 2, 32, 1, 3 }, { 2, 32, 1, 16 }, { 16, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 { 1, 1, 1, 3 }, { 1, 1, 1, 32 }, { 32, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 { 2, 3, 1, 1 }, { 2, 3, 1, 15 }, { 15, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_4 { 1, 512, 1, 1024 }, { 1, 384, 1, 1024 }, { 1024, 1024, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

#define CASE_NORMALIZE_I8_1 { 1, 2, 3, 3 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FP32 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
/* ----------- NOTE: A part of tests is disabled until all FP kernels don't support fusings ------------ */

class conv_fp32_reorder_fsv16_to_bfyx : public ConvFusingTest {};
TEST_P(conv_fp32_reorder_fsv16_to_bfyx, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        reorder("reorder_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv16"), { "weights" }, p.groups, p.stride, p.pad, p.dilation),
        reorder("reorder_bfyx", input_info("conv_prim"), format::bfyx, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_fsv16_to_bfyx, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_5, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_14, 2, 2 },

    convolution_test_params{ CASE_CONV_FP16_1, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_5, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_13, 2, 2 }
}));

class conv_fp32_reorder_fsv16_to_bfyx_conv : public ConvFusingTest {};
TEST_P(conv_fp32_reorder_fsv16_to_bfyx_conv, basic) {
    auto p = GetParam();

    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };
    auto dw_stride = tensor{ 0, 0, 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        data("weights_dw", {get_mem(dw_weights_layout, -127, 127)}),
        reorder("reorder_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv16"), { "weights" }, p.groups, p.stride, p.pad, p.dilation),
        reorder("reorder_bfyx", input_info("conv_prim"), format::bfyx, data_types::f32),
        convolution("conv_output", input_info("reorder_bfyx"), { "weights_dw" }, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
        activation("activation", input_info("conv_output"), activation_func::abs),
        reorder("reorder_output", input_info("activation"), p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_fsv16_to_bfyx_conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1,  3, 4 },
    convolution_test_params{ CASE_CONV_FP32_2,  3, 4 },
    convolution_test_params{ CASE_CONV_FP32_3,  3, 4 },
    convolution_test_params{ CASE_CONV_FP32_4,  3, 4 },
    convolution_test_params{ CASE_CONV_FP32_5,  3, 4 },
    convolution_test_params{ CASE_CONV_FP32_14, 3, 4 },

    convolution_test_params{ CASE_CONV_FP16_1,  3, 4 },
    convolution_test_params{ CASE_CONV_FP16_2,  3, 4 },
    convolution_test_params{ CASE_CONV_FP16_3,  3, 4 },
    convolution_test_params{ CASE_CONV_FP16_4,  3, 4 },
    convolution_test_params{ CASE_CONV_FP16_5,  3, 4 },
    convolution_test_params{ CASE_CONV_FP16_13, 3, 4 },
}));


class conv_fp32_activation : public ConvFusingTest {};
TEST_P(conv_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));


class conv_fp32_scale : public ConvFusingTest {};
TEST_P(conv_fp32_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 3 },

    // convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_10, 2, 3 },
}));

class conv_fp32_bias : public ConvFusingTest {};
TEST_P(conv_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, std::vector<primitive_id>{}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("add_bias", { input_info("conv_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add_bias"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_bias, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_10, 2, 3 },
}));

class conv_fp32_double_bias : public ConvFusingTest {};
TEST_P(conv_fp32_double_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias1", {get_mem(get_bias_layout(p))}),
        data("bias2", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, std::vector<primitive_id>{}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("add_bias1", { input_info("conv_prim"), input_info("bias1") }, eltwise_mode::sum),
        eltwise("add_bias2", { input_info("add_bias1"), input_info("bias2") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add_bias2"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_double_bias, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
}));

class conv_fp32_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_fp32_prelu_eltwise, basic_sum) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, basic_prod) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_sum) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(layout{ p.data_type, p.input_format, eltw_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_prod) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(layout{ p.data_type, p.input_format, eltw_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_mixed_types) {
    auto p = GetParam();
    auto slope_type = p.default_type == data_types::f32 ? data_types::f16 : data_types::f32;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(layout{ slope_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } })}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_prelu_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 4 },

    // convolution_test_params{ CASE_CONV_FP32_1, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 4 },
}));

class conv_fp32_multi_eltwise_2 : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("eltwise1"), input_info("conv_prim"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 4 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 4 },
}));


class conv_fp32_multi_eltwise_2_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise1_data", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise1_data"), eltwise_mode::sum),
        activation("activation", input_info("eltwise1"), activation_func::clamp, { 0.5f, 2.5f }),
        eltwise("eltwise2", input_info("activation"), input_info("conv_prim"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2_clamp, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 5 },
}));


class conv_fp32_multi_eltwise_4_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_4_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise1_data", {get_mem(get_output_layout(p))}),
        data("eltwise2_data", {get_mem(get_output_layout(p))}),
        data("eltwise4_data", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1_add", input_info("conv_prim"), input_info("eltwise1_data"), eltwise_mode::sum),
        activation("activation", input_info("eltwise1_add"), activation_func::clamp, { 0.5f, 2.5f }),
        eltwise("eltwise2_mul", input_info("activation"), input_info("conv_prim"), eltwise_mode::prod),
        eltwise("eltwise3_div", input_info("eltwise2_mul"), input_info("eltwise2_data"), eltwise_mode::prod),
        eltwise("eltwise4_add", input_info("eltwise3_div"), input_info("eltwise4_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise4_add"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_4_clamp, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 7 },
}));

class conv_fp32_eltwise_fusing_extend_ops : public ConvFusingTest {};
TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern01_simple_sub) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        data("eltwise_data2", {get_mem(get_output_layout(p))}),
        data("eltwise_data4", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_sub", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sub),
        eltwise("eltwise3_prod", input_info("eltwise1_sum"), input_info("eltwise2_sub"), eltwise_mode::prod),
        eltwise("eltwise4_sum", input_info("eltwise3_prod"), input_info("eltwise_data4"), eltwise_mode::sum),
        concatenation("concat", { input_info("eltwise4_sum"), input_info("eltwise4_sum") }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern02_sub_scale) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        data("eltwise_data2", {get_mem(get_output_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_sub", input_info("conv_prim"), input_info("eltwise1_sum"), eltwise_mode::sub),
        eltwise("eltwise3_prod", input_info("eltwise2_sub"), input_info("eltwise_data2"), eltwise_mode::prod),
        scale("scale", input_info("eltwise3_prod"), input_info("scale_data")),
        concatenation("concat", { input_info("scale"), input_info("scale") }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern03_sub_div) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        data("eltwise_data2", {get_mem(get_output_layout(p), 1.0f)}),
        data("eltwise_data3", {get_mem(get_output_layout(p))}),
        data("eltwise_data4", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_div", input_info("eltwise1_sum"), input_info("eltwise_data2"), eltwise_mode::div),
        eltwise("eltwise3_prod", input_info("eltwise2_div"), input_info("eltwise_data3"), eltwise_mode::prod),
        eltwise("eltwise4_sum", input_info("eltwise3_prod"), input_info("eltwise_data4"), eltwise_mode::sum),
        concatenation("concat", { input_info("eltwise4_sum"), input_info("eltwise4_sum") }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_fusing_extend_ops, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 3, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 3, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 3, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 3, 7 },
}));

class conv_fp32_eltwise_fusing_2conv : public ConvFusingTest {};
TEST_P(conv_fp32_eltwise_fusing_2conv, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("bias0", {get_mem(get_bias_layout(p))}),
        data("weights0", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim0", input_info("input"), { "weights0" }, { "bias0" }, p.groups, p.stride, p.pad, p.dilation),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", input_info("conv_prim0"), input_info("conv_prim"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim0"), input_info("conv_prim"), eltwise_mode::sum),
        eltwise("eltwise3", input_info("eltwise1"), input_info("eltwise2"), eltwise_mode::prod),
        concatenation("concat", { input_info("eltwise3"), input_info("eltwise3") }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim0", conv_impl }, { "conv_prim", conv_impl }  }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_fusing_2conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 4, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 4, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 4, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 4, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 4, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 4, 7 },
}));


class conv_fp32_multi_eltwise_3_fusing : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_3_fusing, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        data("eltwise_data2", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sum),
        eltwise("eltwise3", input_info("eltwise1"), input_info("eltwise2"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise3"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_3_fusing, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 5 },
}));



class conv_fp32_multi_eltwise_quantization : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_quantization, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("eltwise1"), input_info("quantize"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_quantization, ::testing::ValuesIn(std::vector<convolution_test_params>{
//  convolution_test_params{ CASE_CONV_FP32_2, 4, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 4, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 4, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 4, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 4, 5 },
}));


class conv_fp32_multi_eltwise_concat : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_concat, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", {get_mem(get_output_layout(p))}),
        data("eltwise_data2", {get_mem(get_output_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("weights", {get_mem(get_weights_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sum),
        concatenation("concat",
            { input_info("eltwise1"), input_info("eltwise2") },
            concatenation::concatenation_axis::along_f,
            data_types::i8,
            "",
            padding{ { 0, 0, 0, 0 }, 0 }),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_concat, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 5, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 5, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 5, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 5, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 5, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 5, 5 },
}));

class conv_fp32_eltwise_b_fs_zyx_fsv16 : public ConvFusingTest {};

TEST_P(conv_fp32_eltwise_b_fs_zyx_fsv16, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise", input_info("conv_prim"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_zyx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    execute(p);
}

class conv_fp32_swish : public ConvFusingTest {};
TEST_P(conv_fp32_swish, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("sigmoid", input_info("conv_prim"), activation_func::logistic),
        eltwise("mul", { input_info("conv_prim"), input_info("sigmoid") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("mul"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_swish, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 4 },

    // convolution_test_params{ CASE_CONV_FP32_1, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 4 },
}));

TEST_P(conv_fp32_eltwise_b_fs_zyx_fsv16, splitted_vector_ops) {
    auto p = GetParam();

    std::vector<std::string> weights_idx;
    for (size_t w = 0; w < p.groups; w++) {
        create_topologies(data("weights" + std::to_string(w), {get_mem(get_weights_layout(p, p.groups))}));
        weights_idx.push_back(("weights" + std::to_string(w)));
    }

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), weights_idx, {}, 1, p.stride, p.pad, p.dilation),
        eltwise("eltwise", input_info("conv_prim"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_zyx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1e-5f;
    //  commented because split mode is disabled
    //  execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_b_fs_zyx_fsv16, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_6, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_7, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_8, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_9, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_11, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_12, 2, 3 },
    // convolution_test_params{ CASE_CONV_FP32_13, 2, 3 }, - leads to mvn_scale_activation_quantize_i8_eltwise_fp32_quantize_i8.basic/11 test failure

    convolution_test_params{ CASE_CONV_FP16_6, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_7, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_8, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_9, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_11, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_12, 2, 3 },
}));

class conv_fp32_quantize_u8_first_conv : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8_first_conv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        reorder("reordered_input", input_info("input"), format::b_fs_yx_fsv16, p.data_type),
        convolution("conv_prim", input_info("reordered_input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8_first_conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_14, 2, 3 },
}));

class conv_fp32_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
}));

class conv_fp32_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 4 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 4 },
}));

class conv_fp32_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 5 },
}));

class conv_fp32_scale_activation_quantize_u8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_u8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  p.default_type),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_u8_eltwise_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 6 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 6 },
}));

class conv_fp32_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        activation("activation_quantize", input_info("quantize"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation_quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 6 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 6 },
}));


class conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_lo1", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("in_hi1", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_lo1", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("out_hi1", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("eltwise_data", {get_mem(layout{ data_types::i8, p.input_format, p.out_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"), input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 7 },
}));

class conv_fp32_activation_eltwise_in_u8_fp32 : public ConvFusingTest {};
TEST_P(conv_fp32_activation_eltwise_in_u8_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(layout{ data_types::i8, p.input_format, p.out_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_in_u8_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 4 }, - eltwise fusing not supported
    convolution_test_params{ CASE_CONV_FP32_2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 4 },
    // convolution_test_params{ CASE_CONV_FP32_5, 2, 4 }, - eltwise fusing not supported
    convolution_test_params{ CASE_CONV_FP32_6, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_7, 2, 4 },
    // convolution_test_params{ CASE_CONV_FP32_8, 2, 4 }, - unknown bug
    convolution_test_params{ CASE_CONV_FP32_9, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 4 },
}));

class conv_fp32_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_fp32_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_1, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_3, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_4, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_5, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_6, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_7, 3, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_8, 3, 4 },
}));

class conv_scale_activation_eltwise_fp32_quantize_i8 : public ConvEltwTest {};
TEST_P(conv_scale_activation_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        data("scale_data", {get_mem(get_per_channel_layout(p))}),
        scale("scale", input_info("conv"), input_info("scale_data")),
        activation("activation", input_info("scale"), activation_func::hyperbolic_tan),
        data("eltwise_data", {get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })}),
        eltwise("eltw", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        data("in_low", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_high", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_low", {get_mem(get_single_element_layout(p), -127, 127)}),
        data("out_high", {get_mem(get_single_element_layout(p), -127, 127)}),
        quantize("quant", input_info("eltw"), input_info("in_low"), input_info("in_high"), input_info("out_low"), input_info("out_high"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_scale_activation_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_1, 2, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_2, 2, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_3, 2, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_4, 2, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_5, 3, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_6, 3, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_7, 3, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_8, 3, 6 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- INT8 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_int8_scale : public ConvFusingTest {};
TEST_P(conv_int8_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_int8_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data"), optional_data_type{ data_types::f16 }),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 3 },
}));

class conv_int8_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_eltwise, fp16_eltwise_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::f16),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 3 },
}));

class conv_int8_scale_shift_swish : public ConvFusingTest {};
TEST_P(conv_int8_scale_shift_swish, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        data("shift_data", {get_mem(get_per_channel_layout(p), 1)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("scale0", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        eltwise("scale1", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        eltwise("shift0", { input_info("scale0"), input_info("shift_data") }, eltwise_mode::sum),
        eltwise("shift1", { input_info("scale1"), input_info("shift_data") }, eltwise_mode::sum),
        activation("sigmoid", input_info("shift0"), activation_func::logistic),
        eltwise("mul", { input_info("shift1"), input_info("sigmoid") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("mul"), p.default_format, data_types::f32)
    );

    tolerance = 1e-3f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_shift_swish, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 8 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 8 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 8 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 8 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 8 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 8 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 8 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 8 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 8 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 8 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 8 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 8 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 8 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 8 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 8 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 8 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 8 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 8 },
}));

class conv_int8_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_prelu_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_int8_prelu_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_prelu_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 4 },
}));

class conv_int8_activation_eltwise_quantize : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise_quantize, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_activation_eltwise_quantize, fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv32, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise_quantize, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 5 },
}));

class conv_int8_activation_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_int8_activation_eltwise, fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv32, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 4 },
}));

class conv_int8_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_int8_quantize_u8, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(conv_int8_quantize_u8, per_tensor) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), -10)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_quantize_u8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 3 },
}));

class conv_int8_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 4 },
}));

class conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_int8" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_S8S8_11, 2, 4 },
}));

class conv_int8_relu_quantize : public ConvFusingTest {};
TEST_P(conv_int8_relu_quantize, i8) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("relu", input_info("conv_prim"), activation_func::relu),
        quantize("quantize", input_info("relu"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

TEST_P(conv_int8_relu_quantize, u8) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("relu", input_info("conv_prim"), activation_func::relu),
        quantize("quantize", input_info("relu"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_relu_quantize, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 4 },
}));

class conv_int8_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 2.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 5 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 5 },
}));

class conv_int8_scale_activation_quantize_i8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 6 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 6 },
}));

class conv_int8_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        activation("activation_quantize", input_info("quantize"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation_quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 6 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 6 },
}));


class conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
// With some input values accuracy error might be = 2, so the test is disabled.
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_lo1", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("in_hi1", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_lo1", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("out_hi1", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("eltwise_data", {get_mem(layout{ data_types::i8, p.input_format, p.out_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"), input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 7 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 7 },
}));

class conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec : public ConvFusingTest {};
TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_lo1", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("in_hi1", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_lo1", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("out_hi1", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("slope_data", {get_mem(get_per_channel_layout(p))}),
        data("eltwise_data", {get_mem(layout{ data_types::i8, format::b_fs_yx_fsv4, p.out_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"), input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops_mixed_types) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_lo1", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("in_hi1", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_lo1", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("out_hi1", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)}),
        data("slope_data", {get_mem(layout{ data_types::f16, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } })}),
        data("eltwise_data", {get_mem(layout{ data_types::u8, format::b_fs_yx_fsv4, p.out_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"), input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_5, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_5, 2, 7 },
}));

class conv_int8_asymmetric_weights : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_weights, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                                           get_weights_layout(p);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(weights_layout)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("w_zp", {get_mem(get_weights_zp_layout(p), 1, 127)}),
        eltwise("w_sub", { input_info("weights"), input_info("w_zp") }, eltwise_mode::sub, data_types::f32),
        convolution("conv_prim", input_info("input"), { "w_sub" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
        reorder("reorder_bfyx", input_info("conv_prim"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 4lu);  // input + weights + bias + w_zp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_weights, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2 },
}));

class conv_int8_asymmetric_data : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_data, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                          get_weights_layout(p);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(weights_layout)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("a_zp", {get_mem(get_activations_zp_layout(p), 1, 127)}),
        eltwise("a_sub", { input_info("input"), input_info("a_zp") }, eltwise_mode::sub, data_types::f32),
        convolution("conv_prim", input_info("a_sub"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
        reorder("reorder_bfyx", input_info("conv_prim"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 5lu);  // input + weights + bias + a_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_data, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 3 },
}));

class conv_int8_asymmetric_data_and_weights : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_data_and_weights, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                          get_weights_layout(p);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(weights_layout)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("a_zp", {get_mem(get_activations_zp_layout(p), 1, 127)}),
        data("w_zp", {get_mem(get_weights_zp_layout(p), 1, 127)}),
        eltwise("a_sub", { input_info("input"), input_info("a_zp") }, eltwise_mode::sub, data_types::f32),
        eltwise("w_sub", { input_info("weights"), input_info("w_zp") }, eltwise_mode::sub, data_types::f32),
        convolution("conv_prim", input_info("a_sub"), { "w_sub" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
        reorder("reorder_bfyx", input_info("conv_prim"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 6lu);  // input + weights + bias + a_zp + w_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_data_and_weights, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 3 },
}));


class conv_i8_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_i8_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_i8_activation_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_1, 3, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_3, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_4, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_5, 3, 4 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------- Force convolution kernel cases ---------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_fp16_activation : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_activation, ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
    bc_force_kernel_params{ CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32" },
}));


class conv_fp16_scale : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        scale("scale", input_info("conv_prim"), input_info("scale_data")),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_scale, ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
    bc_force_kernel_params{ CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32" },
}));


/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------- reorder(bfyx to fs_b_yx_fsv32) + convolution kernel cases -------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define FSV32_CASE_CONV_FP32_1 { 1, 32, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx

// 'reorder_fsv32' is being removed from "remove_redundant_reorders" in the current impl
class conv_fp32_reorder_bfyx_to_fsv32_conv_basic : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_basic, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        reorder("reorder_fsv32", input_info("input"), format::fs_b_yx_fsv32, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv32"), { "weights" }, 1, tensor{ 0, 0, 1, 1 }, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_out", input_info("activation"), format::bfyx, data_types::f32)
    );

    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_basic, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1,  3, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'mean'
class conv_fp32_reorder_bfyx_to_fsv32_conv_mean : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_mean, have_mean) {
    auto p = GetParam();
    memory::ptr mul = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 3, 1, 2 } });
    set_values<float>(mul, { 0.5f, 2.5f, -5.0f, 4.3f, 1.2f, -3.5f });

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("mul", {mul}),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        reorder("reorder_fsv32", input_info("input"), format::fs_b_yx_fsv32, data_types::f32, "mul", reorder_mean_mode::mul),
        convolution("conv_prim", input_info("reorder_fsv32"), { "weights" }, 1, tensor{ 0, 0, 1, 1 }, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs)
    );

    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_mean, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1,  4, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'subtract'
class conv_fp32_reorder_bfyx_to_fsv32_conv_subtract : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_subtract, have_subtract_per_feature) {
    auto p = GetParam();
    const std::vector<float>& values_to_subtract = {
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f
    };

    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(2, 2));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };
    auto dw_stride = tensor{ 0, 0, 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        data("weights_dw", {get_mem(dw_weights_layout, -127, 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, p.groups, p.stride, p.pad, p.dilation),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32, values_to_subtract),
        convolution("conv_output", input_info("reorder_fsv32"), { "weights_dw" }, p.out_shape.feature[0], dw_stride, p.pad, p.dilation)
    );

    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_subtract, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1,  4, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'fused_activation'
class conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation, have_fused_activation) {
    auto p = GetParam();

    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(2, 2));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };
    auto dw_stride = tensor{ 0, 0, 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        data("weights_dw", {get_mem(dw_weights_layout, -127, 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, p.groups, p.stride, p.pad, p.dilation),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32),
        activation("activation_quantize", input_info("reorder_fsv32"), activation_func::relu),
        convolution("conv_prim2", input_info("activation_quantize"), { "weights_dw" }, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim2"), activation_func::abs)
    );

    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim2", conv_impl } }));
    bo_fused.set_option(build_option::force_implementations({ { "activation", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1,  5, 6 }
}));

// 'reorder_fsv32' is being fused even if it has 'padding'
class conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding, have_data_padding) {
    auto p = GetParam();

    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(2, 2));
    auto dw_weights_layout = layout{ p.default_type, format::goiyx, dw_tensor };
    auto dw_stride = tensor{ 0, 0, 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -127, 127)}),
        data("weights_dw", {get_mem(dw_weights_layout, -127, 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, p.groups, p.stride, p.pad, p.dilation),
        reorder("reorder_fsv32", input_info("conv_prim"), layout(data_types::f32, format::fs_b_yx_fsv32, dw_tensor, padding{ { 0, 0, 1, 1 }, 0 })),
        convolution("conv_prim2", input_info("reorder_fsv32"), { "weights_dw" }, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
        reorder("reorder_out", input_info("conv_prim2"), format::fs_b_yx_fsv32, data_types::f32)
    );

    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim2", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1,  5, 5 }
}));

#ifdef ENABLE_ONEDNN_FOR_GPU
class conv_int8_eltwise_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_eltwise_onednn, u8_eltwise_sum_out) {
    auto p = GetParam();

    auto shift_layout = get_output_layout(p);
    shift_layout.data_type = data_types::f32;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), 0, 2)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("shift_data", {get_mem(shift_layout)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("shift", { input_info("conv_prim"), input_info("shift_data") }, eltwise_mode::sum, data_types::f32),
        // Add 'not fusable' primitive to be able to test full size tensor sum
        crop("crop", input_info("shift"), get_output_layout(p).size, { 0, 0, 0, 0 }),
        reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_eltwise_onednn, u8_eltwise_prod_out) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -2, 2)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())} ),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::u8),
        crop("crop", input_info("scale"), get_output_layout(p).size, { 0, 0, 0, 0 }),
        reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },

    convolution_test_params{ CASE_CONV_U8S8_11, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 3, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 3, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 3, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 3, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 3, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 3, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 3, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 3, 4 },
}));

class conv_fp32_activation_abs_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_abs_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_abs_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));

class conv_fp32_activation_mish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_mish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::mish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_mish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));

class conv_fp32_activation_swish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_swish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::swish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_swish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));

class conv_fp32_activation_hswish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_hswish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::hswish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_hswish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));

class conv_fp32_activation_exp_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_exp_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::exp),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_exp_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 3 },
}));

class conv_int8_quantize_u8_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_quantize_u8_onednn, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -2, 2)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), -10, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 0, 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(conv_int8_quantize_u8_onednn, per_tensor) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -2, 2)}),
        data("bias", {get_mem(get_bias_layout(p), 0)}),
        data("in_lo", {get_mem(get_single_element_layout(p), -10)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_quantize_u8_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },
}));

class conv_int8_activation_eltwise_quantize_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_activation_eltwise_quantize_onednn, bsv32_fsv32) {
    auto p = GetParam();
    layout eltwise_layout = get_output_layout(p);
    eltwise_layout.format = format::bs_fs_yx_bsv32_fsv32;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -1, 1)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(eltwise_layout, -0.5, 0.5)}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bs_fs_yx_bsv32_fsv32, "", impl_types::onednn };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise_quantize_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 5 },

    convolution_test_params{ CASE_CONV_S8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 5 },
}));

class conv_int8_scale_shift_swish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_scale_shift_swish_onednn, bsv32_fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -1, 1)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        data("shift_data", {get_mem(get_per_channel_layout(p), 1)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("scale0", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::sum),
        eltwise("shift0", { input_info("scale0"), input_info("shift_data") }, eltwise_mode::sum),
        activation("sigmoid", input_info("shift0"), activation_func::swish),
        eltwise("scale1", { input_info("sigmoid"), input_info("scale_data") }, eltwise_mode::sum),
        eltwise("shift1", { input_info("scale1"), input_info("shift_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("shift1"), p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bs_fs_yx_bsv32_fsv32, "", impl_types::onednn };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_shift_swish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 7 },

    convolution_test_params{ CASE_CONV_U8S8_11, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 7 },
}));

class conv_int8_eltwise_scale_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_eltwise_scale_onednn, u8_eltwise_prod_out_reuse) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p), -2, 2)}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("sum_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
        eltwise("sum", { input_info("conv_prim"), input_info("sum_data") }, eltwise_mode::sum, data_types::f32),
        eltwise("scale", { input_info("sum"), input_info("scale_data") }, eltwise_mode::prod, data_types::f32),
        crop("crop", input_info("scale"), get_output_layout(p).size, { 0, 0, 0, 0 }),
        reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));

    auto forcing_format = p.input_format;
    implementation_desc conv_impl = { forcing_format, "", impl_types::onednn };
    bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    // First network.execute() call
    compare(network_not_fused, network_fused, p);
    // Second network.execute() call to make sure that scales have not been wrongly overwritten within first iteration
    // and don't affect final result of second iteration
    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise_scale_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_15, 2, 5 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------ OneDNN post-ops cases with optimizations ----------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

// Before optimization: eltw_linear + eltw_linear
// After optimization: eltw_linear
// Limitations: no
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_linear:1:-128
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_linear:12.75:-0.5
class post_ops_optimizations_onednn_eltw_linear_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_linear_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), -10)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -128)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_linear_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 3 },
}));

// Before optimization: eltw_non_linear + eltw_linear
// After optimization: eltw_non_linear
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round+eltwise_linear:2.00784+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round:0:0:2.00784+eltwise_clip:0:512
class post_ops_optimizations_onednn_eltw_non_linear_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_non_linear_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), -10)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 512)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::f32),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_non_linear_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 3 },
}));

// Before optimization: binary_add + eltw_linear
// After optimization: binary_add
// Limitations: alpha = 1 and scale = 1 in eltw_linear; binary_add is a constant compile-time buffer
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:binary_add:f32:2+eltwise_linear:1:-127+eltwise_clip:-127:127
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:binary_add:f32:2+eltwise_clip:-127:127
class post_ops_optimizations_onednn_binary_add_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_binary_add_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_binary_add_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 3 },
}));

// Before optimization: binary_mul + eltw_linear
// After optimization: binary_mul
// Limitations: beta = 0 in eltw_linear; binary_mul is a constant compile-time buffer
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:binary_mul:f32:2+eltwise_linear:2.01575+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:binary_mul:f32:2+eltwise_clip:0:512
class post_ops_optimizations_onednn_binary_mul_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_binary_mul_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("eltwise_data", {get_mem(get_per_channel_layout(p), -1, 1)}),
        data("in_lo", {get_mem(get_per_channel_layout(p), 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 512)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise", { input_info("conv_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_binary_mul_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 4 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 4 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 4 },
}));

// Before optimization: o_scale + eltw_linear
// After optimization: o_scale
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:eltwise_linear:2.01575+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:eltwise_clip:0:512
class post_ops_optimizations_onednn_oscale_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_oscale_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 512)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_oscale_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 3 },
}));

// Before optimization: eltw_any + sum + eltw_linear
// After optimization: eltw_any + sum
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_relu+sum:1:0:u8+eltwise_linear:12.7+eltwise_clip:0:127
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_relu:0:0:12.7+sum:12.7:0:u8+eltwise_clip:0:127
class post_ops_optimizations_onednn_eltw_any_sum_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_any_sum_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("eltwise_data", {get_mem(get_output_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum),
        quantize("quantize", input_info("sum"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 128, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_any_sum_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 5 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 5 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 5 },
}));

// Input range uses in 2 cases: not per-tensor output range or out_lo > out_hi
// Here's out_lo > out_hi and no optimizations
// DNNL_VERBOSE log: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round+eltwise_linear:-1:127
class post_ops_optimizations_input_range : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_input_range, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), -10)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 10)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), -128)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_input_range, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 3 },
}));


// input:b_fs_yx_fsv32:u8 X weight:bfyx:i8 + eltwise_sum:b_fs_yx_fsv32:u8
// After optimization: eltwise_any + binary_add
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_tanh+binary_add:u8:14:aBcd32b+eltwise_linear:1
class post_ops_optimizations_onednn_binary_add_full_tensor : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_binary_add_full_tensor, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("in_hi", {get_mem(get_single_element_layout(p), 255)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        data("eltwise_data", {get_mem(get_output_layout(p), 0, 255)}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::hyperbolic_tan),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum),
        quantize("quantize", input_info("sum"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_U8S8_FT_BINARY_ADD_1 { 1, 32, 4, 4 }, { 1, 16, 4, 4 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0, 0, 1, 1, 0 }, tensor{ 1 }, 1, data_types::u8, format::b_fs_yx_fsv32, data_types::i8, format::bfyx, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_binary_add_full_tensor, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_FT_BINARY_ADD_1, 2, 5 },
}));


// input:b_fs_yx_fsv16:f16 X weight:bfyx:f16 + eltwise_sum:b_fs_yx_fsv16:f16
// After optimization: eltwise_any + sum
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_tanh+sum:1:0:f16
class post_ops_optimizations_onednn_sum_full_tensor : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_sum_full_tensor, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
        activation("activation", input_info("conv_prim"), activation_func::hyperbolic_tan),
        data("eltwise_data", {get_mem(get_output_layout(p), 0, 255)}),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

#define CASE_CONV_F16F16_FT_ELTW_SUM_1 { 1, 32, 4, 4 }, { 1, 16, 4, 4 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0, 0, 1, 1, 0 }, tensor{ 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_sum_full_tensor, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_F16F16_FT_ELTW_SUM_1, 2, 4 },
}));

#endif  // ENABLE_ONEDNN_FOR_GPU
