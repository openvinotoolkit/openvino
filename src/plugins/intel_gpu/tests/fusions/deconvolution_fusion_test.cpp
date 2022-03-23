// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/deconvolution.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

struct deconv_test_params {
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

struct deconv_eltw_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor eltw_shape;
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

class DeconvolutionFusingTest : public ::BaseFusingTest<deconv_test_params> {
public:
    void execute(deconv_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "deconv")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(deconv_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(deconv_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class ConvEltwTest : public ::BaseFusingTest<deconv_eltw_test_params> {
public:

    void execute(deconv_eltw_test_params& p) {
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

    layout get_input_layout(deconv_eltw_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(deconv_eltw_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Deconvolution cases ----------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_DECONV_FP32_1 { 1, 15, 4, 5 }, { 1, 30, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_5 { 1, 15, 4, 5 }, { 1, 30, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_6 { 1, 16, 4, 5 }, { 1, 32, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_7 { 1, 16, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_8 { 1, 32, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx

#define CASE_DECONV_FP16_1 { 1, 15, 4, 5 }, { 1, 30, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_2 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_5 { 1, 15, 4, 5 }, { 1, 30, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_6 { 1, 16, 4, 5 }, { 1, 32, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_7 { 1, 16, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_8 { 1, 32, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx

#define CASE_DECONV_S8S8_1 { 1, 15, 4, 5 }, { 1, 30, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_2 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::i8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_5 { 1, 15, 4, 5 }, { 1, 30, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_6 { 1, 16, 4, 5 }, { 1, 32, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_7 { 1, 16, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_8 { 1, 32, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 32, data_types::i8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx

#define CASE_DECONV_U8S8_1 { 1, 15, 4, 5 }, { 1, 30, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_2 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::u8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_5 { 1, 15, 4, 5 }, { 1, 30, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_6 { 1, 16, 4, 5 }, { 1, 32, 9, 11 }, { 1, 1, 3, 3 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_7 { 1, 16, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_8 { 1, 32, 4, 5 }, { 1, 32, 7, 9 }, { 1, 1, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 32, data_types::u8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx


// 3D
// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_DECONV_FP32_3D_1 { 1, 15, 4, 5, 3 }, { 1, 30, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_2 { 1, 16, 4, 5, 3 }, { 1, 32, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_3 { 1, 16, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_4 { 1, 32, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32,  format::gs_oizyx_gsv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_5 { 1, 15, 4, 5, 3 }, { 1, 30, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_6 { 1, 16, 4, 5, 3 }, { 1, 32, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_7 { 1, 16, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_8 { 1, 32, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32,  format::gs_oizyx_gsv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_9 { 16, 16, 4, 5, 3 }, { 16, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx

#define CASE_DECONV_FP16_3D_1 { 1, 15, 4, 5, 3 }, { 1, 30, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::oizyx, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_2 { 1, 16, 4, 5, 3 }, { 1, 32, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_3 { 1, 16, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_4 { 1, 32, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16,  format::gs_oizyx_gsv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_5 { 1, 15, 4, 5, 3 }, { 1, 30, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::oizyx, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_6 { 1, 16, 4, 5, 3 }, { 1, 32, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_7 { 1, 16, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_8 { 1, 32, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16,  format::gs_oizyx_gsv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_9 { 16, 16, 4, 5, 3 }, { 16, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx

#define CASE_DECONV_S8S8_3D_1 { 1, 15, 4, 5, 3 }, { 1, 30, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_2 { 1, 16, 4, 5, 3 }, { 1, 32, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_3 { 1, 16, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_4 { 1, 32, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_5 { 1, 15, 4, 5, 3 }, { 1, 30, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_6 { 1, 16, 4, 5, 3 }, { 1, 32, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_7 { 1, 16, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_8 { 1, 32, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx

#define CASE_DECONV_U8S8_3D_1 { 1, 15, 4, 5, 3 }, { 1, 30, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_2 { 1, 16, 4, 5, 3 }, { 1, 32, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_3 { 1, 16, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_4 { 1, 32, 4, 5, 3 }, { 1, 32, 4, 5, 3 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_5 { 1, 15, 4, 5, 3 }, { 1, 30, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_6 { 1, 16, 4, 5, 3 }, { 1, 32, 9, 11, 7 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_7 { 1, 16, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_8 { 1, 32, 4, 5, 3 }, { 1, 32, 7, 9, 5 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx

#define CASE_DECONV_ELTW_FP32_1 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 32, 1, 1 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 6, 7 }, { 1, 1, 1, 1 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_4 { 1, 15, 4, 5, 3 }, { 1, 30, 6, 7, 5 }, { 1, 1, 6, 7, 5 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_5 { 1, 15, 4, 5, 4 }, { 1, 30, 6, 7, 6 }, { 1, 30, 6, 1, 6 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_6 { 1, 32, 2, 2, 2 }, { 1, 16, 4, 4, 4 }, { 1, 16, 1, 4, 1 }, { 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_7 { 1, 16, 3, 5 }, { 1, 32, 5, 7 }, { 1, 32, 1, 7 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_8 { 1, 32, 4, 5 }, { 1, 32, 7, 9 }, { 1, 32, 1, 1 }, { 1, 1, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx

#define CASE_DECONV_ELTW_i8_1 { 1, 16, 3, 5 }, { 1, 32, 5, 7 }, { 1, 32, 5, 1 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_i8_2 { 1, 32, 4, 5, 3 }, { 1, 32, 6, 7, 5 }, { 1, 32, 1, 1, 1 }, { 1, 1, 3, 3, 3 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 1, 1 }, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_i8_3 { 1, 5, 5, 5, 5 }, { 1, 5, 5, 5, 5 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_i8_4 { 1, 16, 1, 4 }, { 1, 16, 1, 6 }, { 1, 16, 1, 1 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_i8_5 { 1, 16, 2, 4 }, { 1, 16, 4, 6 }, { 1, 16, 4, 1 }, { 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx

class deconv_actv : public DeconvolutionFusingTest {};
TEST_P(deconv_actv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        activation("act", "deconv", activation_func::relu),
        reorder("out", "act", p.default_format, data_types::f32)
    );
    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_actv, ::testing::ValuesIn(std::vector<deconv_test_params>{
    deconv_test_params{ CASE_DECONV_FP32_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP16_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },
    // Here and below this test case and CASE_DECONV_S8S8_4 are commented because they fail for z_pad=0 which is unexpected
    // deconv_test_params{ CASE_DECONV_U8S8_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },
    // deconv_test_params{ CASE_DECONV_S8S8_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_9, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_9, 2, 3 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 3 },
}));


class deconv_bias : public DeconvolutionFusingTest {};
TEST_P(deconv_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        eltwise("bias_add", { "deconv", "bias" }, eltwise_mode::sum),
        reorder("out", "bias_add", p.default_format, data_types::f32)
    );

    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_bias, ::testing::ValuesIn(std::vector<deconv_test_params>{
    deconv_test_params{ CASE_DECONV_FP32_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP16_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
}));

class deconv_scale : public DeconvolutionFusingTest {};
TEST_P(deconv_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data"),
        reorder("out", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(deconv_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data", optional_data_type{ data_types::f16 }),
        reorder("out", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale, ::testing::ValuesIn(std::vector<deconv_test_params>{
    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },
    // deconv_test_params{ CASE_DECONV_U8S8_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },
    // deconv_test_params{ CASE_DECONV_S8S8_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 3 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 3 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 3 },
}));

class deconv_actv_eltw_actv : public DeconvolutionFusingTest {};
TEST_P(deconv_actv_eltw_actv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("eltw_data", get_mem(get_output_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        activation("act1", "deconv", activation_func::relu),
        eltwise("eltw", { "act1", "eltw_data" }, eltwise_mode::sum),
        activation("act2", "eltw", activation_func::relu),
        reorder("out", "act2", p.default_format, data_types::f32)
    );

    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_actv_eltw_actv, ::testing::ValuesIn(std::vector<deconv_test_params>{
    // Some fusings disabled under deconvolution -> convolution optimization
    deconv_test_params{ CASE_DECONV_FP32_1, 3, 5 },
    deconv_test_params{ CASE_DECONV_FP32_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_FP16_1, 3, 5 },
    deconv_test_params{ CASE_DECONV_FP16_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 5 },
    // deconv_test_params{ CASE_DECONV_U8S8_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 5 },
    // deconv_test_params{ CASE_DECONV_S8S8_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_9, 2, 5 },

    deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_9, 2, 5 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 5 },
}));

class deconv_scale_actv_quant_i8 : public DeconvolutionFusingTest {};
TEST_P(deconv_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.f/p.kernel.count())),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data"),
        activation("actv", "scale", activation_func::softsign),
        quantize("quant", "actv", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("out", "quant", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_actv_quant_i8, ::testing::ValuesIn(std::vector<deconv_test_params>{
    deconv_test_params{ CASE_DECONV_FP32_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_FP16_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 5 },
    // deconv_test_params{ CASE_DECONV_U8S8_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 5 },
    // deconv_test_params{ CASE_DECONV_S8S8_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 5 },
    // FIXME no quantize implementation for bs_fs_yx_bsv16_fsv16 format AND add_required_reorders pass completely ruins data types
    // add_required_reorders pass tries to reorder everything to output type if no format exists, this ruins fp32 -> int8 quantize
    //deconv_test_params{ CASE_DECONV_FP32_3D_9, 3, 5 },

    deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 5 },
    //deconv_test_params{ CASE_DECONV_FP16_3D_9, 3, 5 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 5 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 5 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 5 },
}));

class deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8 : public DeconvolutionFusingTest {};
TEST_P(deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale1_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
        data("in1_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in1_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out1_lo", get_mem(get_single_element_layout(p), 0)),
        data("out1_hi", get_mem(get_single_element_layout(p), 255)),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.out_shape))),
        data("scale2_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
        data("in2_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in2_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out2_lo", get_mem(get_single_element_layout(p), -127)),
        data("out2_hi", get_mem(get_single_element_layout(p), 127)),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale1", "deconv", "scale1_data"),
        activation("actv1", "scale1", activation_func::relu),
        quantize("quant1", "actv1", "in1_lo", "in1_hi", "out1_lo", "out1_hi", 256, data_types::u8),
        eltwise("eltw", { "quant1", "eltw_data" }, eltwise_mode::sum, p.default_type),
        scale("scale2", "eltw", "scale2_data"),
        activation("actv2", "scale2", activation_func::relu),
        quantize("quant2", "actv2", "in2_lo", "in2_hi", "out2_lo", "out2_hi", 255, data_types::i8),
        reorder("out", "quant2", p.default_format, data_types::f32)
    );

    tolerance = 2.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8, ::testing::ValuesIn(std::vector<deconv_test_params>{
    deconv_test_params{ CASE_DECONV_FP32_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_2, 2, 9 },
    // deconv_test_params{ CASE_DECONV_FP32_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_8, 2, 9 },

    deconv_test_params{ CASE_DECONV_FP16_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_8, 2, 9 },

    deconv_test_params{ CASE_DECONV_U8S8_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3, 2, 9 },
    // deconv_test_params{ CASE_DECONV_U8S8_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_8, 2, 9 },

    deconv_test_params{ CASE_DECONV_S8S8_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3, 2, 9 },
    // deconv_test_params{ CASE_DECONV_S8S8_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_8, 2, 9 },

    deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 9 },
    // deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 9 },
    // Commented out due to sporadic CI failures
    // deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 9 },
    // deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 9 },
    // deconv_test_params{ CASE_DECONV_FP32_3D_9, 6, 9 },

    deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 9 },
    // deconv_test_params{ CASE_DECONV_FP16_3D_9, 6, 9 },

    deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 9 },

    deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 9 },
    deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 9 },
}));

class deconv_scale_activation_quantize_i8_eltwise_quantize_u8 : public ConvEltwTest {};
TEST_P(deconv_scale_activation_quantize_i8_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        deconvolution("deconv_prim", "input", { "weights" }, p.groups, p.stride, p.pad),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
        scale("scale", "deconv_prim", "scale_data"),
        activation("activation", "scale", activation_func::relu),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        quantize("quant", "activation", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
        eltwise("eltw", { "quant", "eltwise_data" }, eltwise_mode::sum, p.default_type),
        data("in_low2", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high2", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low2", get_mem(get_single_element_layout(p), 0)),
        data("out_high2", get_mem(get_single_element_layout(p), 255)),
        quantize("quant2", "eltw", "in_low2", "in_high2", "out_low2", "out_high2", 256, data_types::u8),
        reorder("reorder_bfyx", "quant2", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_activation_quantize_i8_eltwise_quantize_u8, ::testing::ValuesIn(std::vector<deconv_eltw_test_params>{
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_1, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_2, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_3, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_4, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_5, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_6, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_7, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_8, 2, 7 },

    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_1, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_2, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_3, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_4, 2, 7 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_5, 2, 7 },

}));

class deconv_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(deconv_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
        deconvolution("deconv_prim", "input", { "weights" }, p.groups, p.stride, p.pad),
        activation("activation", "deconv_prim", activation_func::relu),
        eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum, p.default_type),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_activation_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<deconv_eltw_test_params>{
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_1, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_2, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_3, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_4, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_5, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_6, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_7, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_FP32_8, 2, 4 },

    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_1, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_2, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_3, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_4, 2, 4 },
    deconv_eltw_test_params{ CASE_DECONV_ELTW_i8_5, 2, 4 },
}));
