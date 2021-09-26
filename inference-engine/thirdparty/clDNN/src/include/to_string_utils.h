// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/tensor.hpp"
#include "cldnn/runtime/layout.hpp"
#include "cldnn/runtime/device.hpp"
#include "cldnn/primitives/primitive.hpp"


#include <string>
#include <sstream>
#include <memory>

namespace cldnn {

inline std::string to_string_hex(int val) {
    std::stringstream stream;
    stream << "0x" << std::uppercase << std::hex << val;
    return stream.str();
}

inline std::string bool_to_str(bool cond) { return cond ? "true" : "false"; }

inline std::string get_extr_type(const std::string& str) {
    auto begin = str.find('<');
    auto end = str.find('>');

    if (begin == std::string::npos || end == std::string::npos)
        return {};

    return str.substr(begin + 1, (end - begin) - 1);
}

inline std::string dt_to_str(data_types dt) {
    switch (dt) {
        case data_types::bin:
            return "bin";
        case data_types::i8:
            return "i8";
        case data_types::u8:
            return "u8";
        case data_types::i32:
            return "i32";
        case data_types::i64:
            return "i64";
        case data_types::f16:
            return "f16";
        case data_types::f32:
            return "f32";
        default:
            return "unknown (" + std::to_string(typename std::underlying_type<data_types>::type(dt)) + ")";
    }
}

inline std::string fmt_to_str(format fmt) {
    switch (fmt.value) {
        case format::yxfb:
            return "yxfb";
        case format::byxf:
            return "byxf";
        case format::bfyx:
            return "bfyx";
        case format::fyxb:
            return "fyxb";
        case format::b_fs_yx_fsv16:
            return "b_fs_yx_fsv16";
        case format::b_fs_yx_fsv32:
            return "b_fs_yx_fsv32";
        case format::b_fs_zyx_fsv32:
            return "b_fs_zyx_fsv32";
        case format::bs_xs_xsv8_bsv8:
            return "bs_xs_xsv8_bsv8";
        case format::bs_xs_xsv8_bsv16:
            return "bs_xs_xsv8_bsv16";
        case format::bs_x_bsv16:
            return "bs_x_bsv16";
        case format::winograd_2x3_s1_data:
            return "winograd_2x3_s1_data";
        case format::b_fs_yx_fsv4:
            return "b_fs_yx_fsv4";
        case format::b_fs_yx_32fp:
            return "b_fs_yx_32fp";
        case format::bfzyx:
            return "bfzyx";
        case format::bfwzyx:
            return "bfwzyx";
        case format::fs_b_yx_fsv32:
            return "fs_b_yx_fsv32";
        case format::bs_fs_yx_bsv16_fsv16:
            return "bs_fs_yx_bsv16_fsv16";
        case format::bs_fs_yx_bsv32_fsv16:
            return "bs_fs_yx_bsv32_fsv16";
        case format::bs_fs_yx_bsv4_fsv2:
            return "bs_fs_yx_bsv4_fsv2";
        case format::bs_fs_yx_bsv4_fsv4:
            return "bs_fs_yx_bsv4_fsv4";
        case format::bs_fs_yx_bsv32_fsv32:
            return "bs_fs_yx_bsv32_fsv32";
        case format::b_fs_zyx_fsv16:
            return "b_fs_zyx_fsv16";
        case format::bs_fs_zyx_bsv16_fsv16:
            return "bs_fs_zyx_bsv16_fsv16";
        case format::image_2d_rgba:
            return "image_2d_rgba";

        case format::oiyx:
            return "oiyx";
        case format::ioyx:
            return "ioyx";
        case format::yxio:
            return "yxio";
        case format::oizyx:
            return "oizyx";
        case format::iozyx:
            return "iozyx";
        case format::winograd_2x3_s1_weights:
            return "winograd_2x3_s1_weights";
        case format::winograd_2x3_s1_fused_weights:
            return "winograd_2x3_s1_fused_weights";
        case format::winograd_6x3_s1_fused_weights:
            return "winograd_6x3_s1_fused_weights";
        case format::image_2d_weights_c4_fyx_b:
            return "image_2d_weights_c4_fyx_b";
        case format::image_2d_weights_c1_b_fyx:
            return "image_2d_weights_c1_b_fyx";
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            return "image_2d_weights_winograd_6x3_s1_fbxyb";
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            return "image_2d_weights_winograd_6x3_s1_xfbyb";
        case format::os_yxi_osv16:
            return "os_yxi_osv16";
        case format::o_is_yx_isv16:
            return "o_is_yx_isv16";
        case format::os_iyx_osv16:
            return "os_iyx_osv16";
        case format::os_is_yx_osv16_isv16:
            return "os_is_yx_osv16_isv16";
        case format::os_iyx_osv32:
            return "os_iyx_osv32";
        case format::os_iyx_osv64:
            return "os_iyx_osv64";
        case format::is_o_yx_isv32:
            return "is_o_yx_isv32";
        case format::os_is_yx_isv16_osv16:
            return "os_is_yx_isv16_osv16";
        case format::os_is_yx_isa8_osv8_isv4:
            return "os_is_yx_isa8_osv8_isv4";
        case format::os_is_yx_isa8_osv16_isv4:
            return "os_is_yx_isa8_osv16_isv4";
        case format::os_is_zyx_isa8_osv8_isv4:
            return "os_is_zyx_isa8_osv8_isv4";
        case format::os_is_zyx_isa8_osv16_isv4:
            return "os_is_zyx_isa8_osv16_isv4";
        case format::os_is_yx_osa4_isa8_osv8_isv2:
            return "os_is_yx_osa4_isa8_osv8_isv2";
        case format::g_os_is_yx_osa4_isa8_osv8_isv2:
            return "g_os_is_yx_osa4_isa8_osv8_isv2";
        case format::g_os_is_yx_osa4_isa8_osv8_isv4:
            return "g_os_is_yx_osa4_isa8_osv8_isv4";
        case format::os_is_yx_osa4_isa8_osv8_isv4:
            return "os_is_yx_osa4_isa8_osv8_isv4";
        case format::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_isa8_osv8_isv4_swizzled_by_4";
        case format::is_o32_yx_isv32_swizzled_by_4:
            return "is_o32_yx_isv32_swizzled_by_4";
        case format::os_is_yx_osv8_isv2:
            return "os_is_yx_osv8_isv2";
        case format::os_is_yx_osv8_isv4:
            return "os_is_yx_osv8_isv4";
        case format::os_is_yx_osv16_isv4:
            return "os_is_yx_osv16_isv4";
        case format::os_is_yx_osv32_isv4_swizzled_by_2:
            return "os_is_yx_osv32_isv4_swizzled_by_2";
        case format::os_is_yx_osv32_isv4:
            return "os_is_yx_osv32_isv4";
        case format::os_is_zyx_osv32_isv4:
            return "os_is_zyx_osv32_isv4";
        case format::os_is_y_x8_osv8_isv4:
            return "os_is_y_x8_osv8_isv4";
        case format::os_is_yx_osv32_isv32p:
            return "os_is_yx_osv32_isv32p";
        case format::os_is_zyx_isv16_osv16:
            return "os_is_zyx_isv16_osv16";
        case format::is_os_zyx_isv16_osv16:
            return "is_os_zyx_isv16_osv16";
        case format::is_os_yx_isv16_osv16:
            return "is_os_yx_isv16_osv16";
        case format::os_is_osv32_isv32_swizzled_by_4:
            return "os_is_osv32_isv32_swizzled_by_4";
        case format::os_is_zyx_isv8_osv16_isv2:
            return "os_is_zyx_isv8_osv16_isv2";
        case format::os_zyxi_osv16:
            return "os_zyxi_osv16";

        case format::goiyx:
            return "goiyx";
        case format::goizyx:
            return "goizyx";
        case format::gioyx:
            return "gioyx";
        case format::giozyx:
            return "giozyx";
        case format::g_os_iyx_osv16:
            return "g_os_iyx_osv16";
        case format::g_os_iyx_osv32:
            return "g_os_iyx_osv32";
        case format::gs_oiyx_gsv16:
            return "gs_oiyx_gsv16";
        case format::gs_oiyx_gsv32:
            return "gs_oiyx_gsv32";
        case format::g_is_os_zyx_isv16_osv16:
            return "g_is_os_zyx_isv16_osv16";
        case format::g_is_os_yx_isv16_osv16:
            return "g_is_os_yx_isv16_osv16";
        case format::g_os_is_zyx_isv8_osv16_isv2:
            return "g_os_is_zyx_isv8_osv16_isv2";
        case format::g_os_is_yx_isv8_osv16_isv2:
            return "g_os_is_yx_isv8_osv16_isv2";
        case format::g_os_is_zyx_isv16_osv16:
            return "g_os_is_zyx_isv16_osv16";
        case format::g_os_is_yx_osv16_isv4:
            return "g_os_is_yx_osv16_isv4";
        case format::g_os_is_zyx_osv16_isv16:
            return "g_os_is_zyx_osv16_isv16";
        case format::g_os_zyx_is_osv16_isv4:
            return "g_os_zyx_is_osv16_isv4";
        case format::g_os_zyx_is_osv16_isv16:
            return "g_os_zyx_is_osv16_isv16";
        case format::g_os_zyx_is_osv16_isv32:
            return "g_os_zyx_is_osv16_isv32";
        case format::g_os_zyx_is_osv32_isv4:
            return "g_os_zyx_is_osv32_isv4";
        case format::g_os_zyx_is_osv32_isv16:
            return "g_os_zyx_is_osv32_isv16";
        case format::g_os_zyx_is_osv32_isv32:
            return "g_os_zyx_is_osv32_isv32";
        case format::gs_oi_yxs_gsv32_yxsv4:
            return "gs_oi_yxs_gsv32_yxsv4";
        default:
            return "unknown (" + std::to_string(fmt.value) + ")";
    }
}

inline std::string type_to_str(std::shared_ptr<const primitive> primitive) { return primitive->type_string(); }

inline std::string allocation_type_to_str(allocation_type type) {
    switch (type) {
    case allocation_type::cl_mem: return "cl_mem";
    case allocation_type::usm_host: return "usm_host";
    case allocation_type::usm_shared: return "usm_shared";
    case allocation_type::usm_device: return "usm_device";
    default: return "unknown";
    }
}

}  // namespace cldnn
