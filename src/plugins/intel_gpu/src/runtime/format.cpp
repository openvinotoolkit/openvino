// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/format.hpp"
#include "openvino/core/except.hpp"

#include <list>
#include <vector>
#include <algorithm>

#define FMT_TRAITS(fmt, ...) {format::fmt, {#fmt, __VA_ARGS__}}

namespace cldnn {

static const std::map<format::type, format_traits> format_traits_map {
        // B - number of Batch dimensions
        // F - number of Feature dimensions
        // S - number of Spatial dimensions
        // G - number of Group dimensions
        // Order - dims changing order from rare to often
        // Inner order - dims order for internal storage in _sizes array
        // Block sizes - vector of pairs of dimension number (by inner order) and block size ordered from rare to often
        //         Format                 B  F  S  G   Dims order               Order     Inner order   Block sizes
        FMT_TRAITS(yxfb,                  1, 1, 2, 0, {2, 3, 1, 0},             "yxfb",     "bfxy",     {}),
        FMT_TRAITS(byxf,                  1, 1, 2, 0, {0, 2, 3, 1},             "byxf",     "bfxy",     {}),
        FMT_TRAITS(bfyx,                  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {}),
        FMT_TRAITS(fbyx,                  1, 1, 2, 0, {1, 0, 2, 3},             "fbyx",     "bfxy",     {}),
        FMT_TRAITS(fyxb,                  1, 1, 2, 0, {1, 2, 3, 0},             "fyxb",     "bfxy",     {}),
        FMT_TRAITS(byfx,                  1, 1, 2, 0, {0, 2, 1, 3},             "byfx",     "bfxy",     {}),
        FMT_TRAITS(bxfy,                  1, 1, 2, 0, {0, 3, 1, 2},             "bxfy",     "bfxy",     {}),
        FMT_TRAITS(b_fs_yx_fsv2,          1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 2}}),
        FMT_TRAITS(b_fs_yx_fsv4,          1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 4}}),
        FMT_TRAITS(b_fs_yx_fsv8,          1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 8}}),
        FMT_TRAITS(b_fs_yx_fsv16,         1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 16}}),
        FMT_TRAITS(b_fs_yx_fsv32,         1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 32}}),
        FMT_TRAITS(b_fs_zyx_fsv2,         1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{1, 2}}),
        FMT_TRAITS(b_fs_zyx_fsv4,         1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{1, 4}}),
        FMT_TRAITS(b_fs_zyx_fsv8,         1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{1, 8}}),
        FMT_TRAITS(b_fs_zyx_fsv32,        1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{1, 32}}),
        FMT_TRAITS(bs_fs_fsv8_bsv8,       1, 1, 0, 0, {0, 1},                   "bf",       "bf",       {{0, 8}, {1, 8}}),
        FMT_TRAITS(bs_fs_fsv8_bsv16,      1, 1, 0, 0, {0, 1},                   "bf",       "bf",       {{0, 16}, {1, 8}}),
        FMT_TRAITS(bs_f_bsv16,            1, 1, 0, 0, {0, 1},                   "bf",       "bf",       {{0, 16}}),
        FMT_TRAITS(winograd_2x3_s1_data,  1, 1, 2, 0, {0, 2, 3, 1},             "bxyf",     "bfxy",     {}),
        FMT_TRAITS(bzyxf,                 1, 1, 3, 0, {0, 2, 3, 4, 1},          "bzyxf",    "bfxyz",    {}),
        FMT_TRAITS(bfzyx,                 1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {}),
        FMT_TRAITS(bfwzyx,                1, 1, 4, 0, {0, 1, 2, 3, 4, 5},       "bfwzyx",   "bfxyzw",   {}),
        FMT_TRAITS(bfuwzyx,               1, 1, 5, 0, {0, 1, 2, 3, 4, 5, 6},    "bfuwzyx",  "bfxyzwu",  {}),
        FMT_TRAITS(bfvuwzyx,              1, 1, 6, 0, {0, 1, 2, 3, 4, 5, 6, 7}, "bfvuwzyx", "bfxyzwuv", {}),
        FMT_TRAITS(fs_b_yx_fsv32,         1, 1, 2, 0, {1, 0, 2, 3},             "fbyx",     "bfxy",     {{1, 32}}),
        FMT_TRAITS(b_fs_yx_32fp,          1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{1, 32}}),
        FMT_TRAITS(b_fs_zyx_fsv16,        1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{1, 16}}),
        FMT_TRAITS(bs_fs_zyx_bsv16_fsv32, 1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 16 }, {1, 32}}),
        FMT_TRAITS(bs_fs_zyx_bsv16_fsv16, 1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 16 }, {1, 16}}),
        FMT_TRAITS(bs_fs_yx_bsv16_fsv16,  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 16 }, {1, 16}}),
        FMT_TRAITS(bs_fs_yx_bsv16_fsv32,  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 16 }, {1, 32}}),
        FMT_TRAITS(bs_fs_yx_bsv4_fsv4,    1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 4 }, {1, 4}}),
        FMT_TRAITS(bs_fs_yx_bsv8_fsv4,    1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 8 }, {1, 4}}),
        FMT_TRAITS(bs_fs_zyx_bsv8_fsv4,   1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 8 }, {1, 4}}),
        FMT_TRAITS(bs_fs_yx_bsv16_fsv8,   1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 16 }, {1, 8}}),
        FMT_TRAITS(bs_fs_zyx_bsv16_fsv8,  1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 16 }, {1, 8}}),
        FMT_TRAITS(bs_fs_yx_bsv16_fsv4,   1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 16 }, {1, 4}}),
        FMT_TRAITS(bs_fs_zyx_bsv16_fsv4,  1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 16 }, {1, 4}}),
        FMT_TRAITS(bs_fs_yx_bsv16_fsv2,   1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 16 }, {1, 2}}),
        FMT_TRAITS(bs_fs_zyx_bsv16_fsv2,  1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 16 }, {1, 2}}),
        FMT_TRAITS(bs_fs_yx_bsv8_fsv2,    1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 8 }, {1, 2}}),
        FMT_TRAITS(bs_fs_zyx_bsv8_fsv2,   1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 8 }, {1, 2}}),
        FMT_TRAITS(bs_fs_yx_bsv4_fsv2,    1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 4 }, {1, 2}}),
        FMT_TRAITS(bs_fs_zyx_bsv4_fsv4,   1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 4 }, {1, 4}}),
        FMT_TRAITS(bs_fs_zyx_bsv4_fsv2,   1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 4 }, {1, 2}}),
        FMT_TRAITS(bs_fs_zyx_bsv32_fsv32, 1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 32 }, {1, 32}}),
        FMT_TRAITS(bs_fs_zyx_bsv32_fsv16, 1, 1, 3, 0, {0, 1, 2, 3, 4},          "bfzyx",    "bfxyz",    {{0, 32 }, {1, 16}}),
        FMT_TRAITS(bs_fs_yx_bsv32_fsv32,  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 32 }, {1, 32}}),
        FMT_TRAITS(bs_fs_yx_bsv32_fsv16,  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {{0, 32 }, {1, 16}}),
        FMT_TRAITS(nv12,                  1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {}),
        FMT_TRAITS(image_2d_rgba,         1, 1, 2, 0, {0, 1, 2, 3},             "bfyx",     "bfxy",     {}),

        FMT_TRAITS(oiyx,                                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {}),
        FMT_TRAITS(ioyx,                                         1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {}),
        FMT_TRAITS(iyxo,                                         1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {}),
        FMT_TRAITS(oyxi,                                         1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {}),
        FMT_TRAITS(oyix,                                         1, 1, 2, 0, {0, 2, 1, 3},    "oyix",   "oixy",  {}),
        FMT_TRAITS(oxiy,                                         1, 1, 2, 0, {0, 3, 1, 2},    "oxiy",   "oixy",  {}),
        FMT_TRAITS(yxio,                                         1, 1, 2, 0, {2, 3, 1, 0},    "yxio",   "oixy",  {}),
        FMT_TRAITS(oizyx,                                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {}),
        FMT_TRAITS(iozyx,                                        1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {}),
        FMT_TRAITS(os_is_yx_isv16_osv16,                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 16}, {0, 16}}),
        FMT_TRAITS(o_is_yx_isv4,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 4}}),
        FMT_TRAITS(o_is_yx_isv16,                                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 16}}),
        FMT_TRAITS(o_is_zyx_isv16,                               1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 16}}),
        FMT_TRAITS(os_yxi_osv16,                                 1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {{0, 16}}),
        FMT_TRAITS(os_iyx_osv16,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 16}}),
        FMT_TRAITS(os_iyx_osv32,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}}),
        FMT_TRAITS(os_iyx_osv64,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 64}}),
        FMT_TRAITS(winograd_2x3_s1_weights,                      1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {}),
        FMT_TRAITS(winograd_2x3_s1_fused_weights,                1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy",  {}),
        FMT_TRAITS(winograd_6x3_s1_fused_weights,                1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy",  {}),
        FMT_TRAITS(image_2d_weights_winograd_6x3_s1_fbxyb,       1, 1, 2, 0, {1, 0, 3, 2},    "ioxy",   "oixy",  {}),
        FMT_TRAITS(image_2d_weights_winograd_6x3_s1_xfbyb,       1, 1, 2, 0, {3, 1, 0, 2},    "xioy",   "oixy",  {}),
        FMT_TRAITS(image_2d_weights_c4_fyx_b,                    1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {}),
        FMT_TRAITS(image_2d_weights_c1_b_fyx,                    1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {}),
        FMT_TRAITS(lstm_weights_dio,                             1, 1, 2, 0, {0, 1, 3, 2},    "oixy",   "oixy",  {}),
        FMT_TRAITS(os_is_yx_isa8_osv16_isv4,                     1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 8}, {0, 16}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv2,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 4}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv4,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 4}, {1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv2,                1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 4}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv4,                1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 4}, {1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osa2_isa8_osv16_isv2,                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 2}, {1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(os_is_yx_osa2_isa8_osv16_isv4,                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 2}, {1, 8}, {0, 16}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osa2_isa8_osv8_isv2,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 2}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_osa2_isa8_osv8_isv2,                1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 2}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_isa8_osv8_isv4,                     1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_isa8_osv16_isv4,                    1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 16}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 32}}),
        FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 32}}),
        FMT_TRAITS(is_os_yx_osa4_isa8_osv8_isv4,                 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{0, 4}, {1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(is_os_yx_isa2_osa8_isv8_osv2,                 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 2}, {0, 8}, {1, 8}, {0, 2}}),
        FMT_TRAITS(is_os_yx_isa4_osa8_isv8_osv4,                 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 4}, {0, 8}, {1, 8}, {0, 4}}),
        FMT_TRAITS(is_o_yx_isv32,                                1, 1, 2, 0, {1, 0, 2, 3},    "oyxi",   "oixy",  {{1, 32}}),
        FMT_TRAITS(is_o32_yx_isv32_swizzled_by_4,                1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy",  {{0, 32}, {1, 32}}),
        FMT_TRAITS(os_is_y_x8_osv8_isv4,                         1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy",  {{0, 8}, {1, 4}, {2, 8}}),
        FMT_TRAITS(os_is_y_x8_osv8_isv4_swizzled_by_4,           1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy",  {{0, 8}, {1, 4}, {2, 8}}),
        FMT_TRAITS(os_is_yx_osv2_isv16,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 2}, {1, 16}}),
        FMT_TRAITS(os_is_yx_osv4_isv16,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 4}, {1, 16}}),
        FMT_TRAITS(os_is_yx_osv16_isv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 16}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osv8_isv4,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_osv8_isv4,                          1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osv8_isv2,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_osv8_isv2,                          1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_osv16_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 16}, {1, 16}}),
        FMT_TRAITS(os_is_yx_osv32_isv4_swizzled_by_2,            1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osv32_isv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_osv32_isv4,                         1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 4}}),
        FMT_TRAITS(os_is_yx_osv32_isv32p,                        1, 1, 1, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 32}}),
        FMT_TRAITS(os_is_zyx_isv16_osv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 16}, {0, 16}}),
        FMT_TRAITS(is_os_zyx_isv16_osv16,                        1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {{1, 16}, {0, 16}}),
        FMT_TRAITS(is_os_yx_osv8_isv4,                           1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{0, 8}, {1, 4}}),
        FMT_TRAITS(is_os_yx_isv16_osv16,                         1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 16}, {0, 16}}),
        FMT_TRAITS(is_os_yx_isv16_osv8,                          1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 16}, {0, 8}}),
        FMT_TRAITS(is_os_yx_isv16_osv4,                          1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 16}, {0, 4}}),
        FMT_TRAITS(is_os_yx_isv16_osv2,                          1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 16}, {0, 2}}),
        FMT_TRAITS(is_os_zyx_isa8_osv8_isv2,                     1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(is_os_zyx_isa8_osv8_isv4,                     1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_isa8_osv8_isv4,                     1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_zyx_isa8_osv8_isv2,                     1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_zyx_isa8_osv16_isv4,                    1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 16}, {1, 4}}),
        FMT_TRAITS(is_os_yx_isa8_osv8_isv2,                      1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(is_os_yx_isa8_osv8_isv4,                      1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(is_os_yx_osa8_isv16_osv4,                     1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {{0, 8}, {1, 16}, {0, 4}}),
        FMT_TRAITS(os_is_yx_isa8_osv8_isv4,                      1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(os_is_yx_isa8_osv8_isv4_swizzled_by_4,        1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 32}, {0, 32}}),
        FMT_TRAITS(os_is_yx_isa8_osv8_isv2,                      1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(os_is_osv32_isv32_swizzled_by_4,              1, 1, 0, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 32}}),
        FMT_TRAITS(os_is_zyx_isv8_osv16_isv2,                    1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(os_zyxi_osv16,                                1, 1, 3, 0, {0, 2, 3, 4, 1}, "ozyxi",  "oixyz", {{0, 16}}),
        FMT_TRAITS(os_is_yx_isv8_osv16_isv2,                     1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(os_is_yx_osv16_isv2,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 16}, {1, 2}}),
        FMT_TRAITS(os_is_yx_osv16_isv16,                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 16}, {1, 16}}),
        FMT_TRAITS(os_is_zyx_osv32_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 16}}),
        FMT_TRAITS(os_is_zyx_osv64_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 64}, {1, 16}}),
        FMT_TRAITS(os_iyx_osv8,                                  1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 8}}),
        FMT_TRAITS(os_iyx_osv32__ai32,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}}),
        FMT_TRAITS(i_yxs_os_yxsv2_osv16,                         1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{0, 16}}),
        FMT_TRAITS(iy_xs_os_xsv2_osv8__ao32,                     1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{2, 2}, {0, 8}}),
        FMT_TRAITS(iy_xs_os_xsv2_osv16__ao32,                    1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{2, 2}, {0, 16}}),
        FMT_TRAITS(os_i_yxs_osv4_yxsv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 4}}),
        FMT_TRAITS(os_i_osv16,                                   1, 1, 0, 0, {0, 1},          "oi",     "oi",    {{0, 16}}),
        FMT_TRAITS(os_i_osv16__ai8,                              1, 1, 0, 0, {0, 1},          "oi",     "oi",    {{1, 8}, {0, 16}}),
        FMT_TRAITS(os_i_osv8__ai8,                               1, 1, 0, 0, {0, 1},          "oi",     "oi",    {{1, 8}, {0, 8}}),
        FMT_TRAITS(os_y_is_x_osv8_isv2,                          1, 1, 2, 0, {0, 2, 1, 3},    "oyix",   "oixy",  {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_y_is_x_osv8_isv4,                          1, 1, 2, 0, {0, 2, 1, 3},    "oyix",   "oixy",  {{0, 8}, {1, 4}}),
        FMT_TRAITS(os_yx_is_osv8_isv2,                           1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_yx_is_osv8_isv4,                           1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {{0, 8}, {1, 4}}),
        FMT_TRAITS(os_yx_is_osv16_isv2,                          1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {{0, 16}, {1, 2}}),
        FMT_TRAITS(os_zyx_is_osv8_isv2,                          1, 1, 3, 0, {0, 2, 3, 4, 1}, "ozyxi",  "oixyz", {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_zyx_is_osv8_isv4,                          1, 1, 3, 0, {0, 2, 3, 4, 1}, "ozyxi",  "oixyz", {{0, 8}, {1, 4}}),
        FMT_TRAITS(os_zy_is_x_osv8_isv2,                         1, 1, 3, 0, {0, 2, 3, 1, 4}, "ozyix",  "oixyz", {{0, 8}, {1, 2}}),
        FMT_TRAITS(os_zy_is_x_osv8_isv4,                         1, 1, 3, 0, {0, 2, 3, 1, 4}, "ozyix",  "oixyz", {{0, 8}, {1, 4}}),

        FMT_TRAITS(goiyx,                                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {}),
        FMT_TRAITS(gioyx,                                        1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy????g", {}),
        FMT_TRAITS(goizyx,                                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {}),
        FMT_TRAITS(giozyx,                                       1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz???g", {}),
        FMT_TRAITS(g_os_iyx_osv8,                                1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 8}}),
        FMT_TRAITS(g_os_iyx_osv16,                               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 16}}),
        FMT_TRAITS(g_os_iyx_osv32,                               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 32}}),
        FMT_TRAITS(gs_oiyx_gsv8,                                 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 8}}),
        FMT_TRAITS(gs_oizyx_gsv8,                                1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{8, 8}}),
        FMT_TRAITS(gs_oiyx_gsv16,                                1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 16}}),
        FMT_TRAITS(gs_oizyx_gsv16,                               1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{8, 16}}),
        FMT_TRAITS(gs_oiyx_gsv32,                                1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 32}}),
        FMT_TRAITS(gs_oizyx_gsv32,                               1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{8, 32}}),
        FMT_TRAITS(gyxio,                                        1, 1, 2, 1, {0, 3, 4, 2, 1},    "gyxio",  "oixy????g", {}),
        FMT_TRAITS(g_is_os_zyx_isv16_osv16,                      1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz???g", {{1, 16}, {0, 16}}),
        FMT_TRAITS(g_is_os_yx_isv16_osv16,                       1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy????g", {{1, 16}, {0, 16}}),
        FMT_TRAITS(g_os_is_zyx_isv8_osv16_isv2,                  1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(g_os_is_yx_isv8_osv16_isv2,                   1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(g_os_is_zyx_isv16_osv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{0, 16}, {1, 16}}),
        FMT_TRAITS(g_os_zy_is_x_osv8_isv2,                       1, 1, 3, 1, {0, 1, 3, 4, 2, 5}, "gozyix", "oixyz???g", {{0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_zy_is_x_osv8_isv4,                       1, 1, 3, 1, {0, 1, 3, 4, 2, 5}, "gozyix", "oixyz???g", {{0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_zyx_is_osv8_isv2,                        1, 1, 3, 1, {0, 1, 3, 4, 5, 2}, "gozyxi", "oixyz???g", {{0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_zyx_is_osv8_isv4,                        1, 1, 3, 1, {0, 1, 3, 4, 5, 2}, "gozyxi", "oixyz???g", {{0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_yx_osv8_isv2,                         1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_yx_osv8_isv4,                         1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_yx_osv16_isv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 16}, {1, 4}}),
        FMT_TRAITS(g_os_is_zyx_osv16_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{0, 16}, {1, 16}}),
        FMT_TRAITS(g_os_zyx_is_osv16_isv4,                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 16}, {1, 4}}),
        FMT_TRAITS(g_os_zyx_is_osv16_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 16}, {1, 16}}),
        FMT_TRAITS(g_os_zyx_is_osv16_isv32,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 16}, {1, 32}}),
        FMT_TRAITS(g_os_zyx_is_osv32_isv4,                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 32}, {1, 4}}),
        FMT_TRAITS(g_os_zyx_is_osv32_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 32}, {1, 16}}),
        FMT_TRAITS(g_os_zyx_is_osv32_isv32,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz???g", {{0, 32}, {1, 32}}),
        FMT_TRAITS(g_os_is_yx_isa8_osv8_isv2,                    1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_yx_isa8_osv8_isv4,                    1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_zyx_isa8_osv8_isv2,                   1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_zyx_isa8_osv8_isv4,                   1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_yx_osa2_isa8_osv8_isv2,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 2}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_yx_osa4_isa8_osv8_isv4,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 4}, {1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_zyx_osa4_isa8_osv8_isv4,              1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{0, 4}, {1, 8}, {0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_is_yx_osa4_isa8_osv8_isv2,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 4}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_zyx_osa4_isa8_osv8_isv2,              1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz???g", {{0, 4}, {1, 8}, {0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_is_yx_osa2_isa8_osv16_isv4,              1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 2}, {1, 8}, {0, 16}, {1, 4}}),
        FMT_TRAITS(g_os_is_yx_osa2_isa8_osv16_isv2,              1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{0, 2}, {1, 8}, {0, 16}, {1, 2}}),
        FMT_TRAITS(gs_oi_yxs_gsv4_yxsv4,                         1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 4}}),
        FMT_TRAITS(gs_oi_yxs_gsv16_yxsv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 16}}),
        FMT_TRAITS(gs_oi_yxs_gsv32_yxsv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{8, 32}}),
        FMT_TRAITS(g_os_is_yx_isv16_osv16,                       1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy????g", {{1, 16}, {0, 16}}),
        FMT_TRAITS(gi_yxs_os_yxsv2_osv16,                        1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy????g", {{0, 16}}),
        FMT_TRAITS(giy_xs_os_xsv2_osv8__ao32,                    1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy????g", {{2, 2}, {0, 8}}),
        FMT_TRAITS(giy_xs_os_xsv2_osv16__ao32,                   1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy????g", {{2, 2}, {0, 16}}),
        FMT_TRAITS(g_os_yx_is_osv8_isv2,                         1, 1, 2, 1, {0, 1, 3, 4, 2},    "goyxi",  "oixy????g", {{0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_yx_is_osv8_isv4,                         1, 1, 2, 1, {0, 1, 3, 4, 2},    "goyxi",  "oixy????g", {{0, 8}, {1, 4}}),
        FMT_TRAITS(g_os_y_is_x_osv8_isv2,                        1, 1, 2, 1, {0, 1, 3, 2, 4},    "goyix",  "oixy????g", {{0, 8}, {1, 2}}),
        FMT_TRAITS(g_os_y_is_x_osv8_isv4,                        1, 1, 2, 1, {0, 1, 3, 2, 4},    "goyix",  "oixy????g", {{0, 8}, {1, 4}}),
};

const format_traits& format::traits(type fmt) {
    OPENVINO_ASSERT(format_traits_map.find(fmt) != format_traits_map.end(), "[GPU] Format description is missing in fmt traits");
    return format_traits_map.at(fmt);
}

std::string format::to_string() const {
    if (value == any) {
        return "any";
    }
    return traits(value).str;
}

format format::get_default_format(size_t rank, bool is_weights, bool is_grouped) {
    auto default_fmt = cldnn::format::bfyx;
    if (is_weights) {
        if (is_grouped) {
            if (rank == 5) {
                default_fmt = cldnn::format::goiyx;
            } else if (rank == 6) {
                default_fmt = cldnn::format::goizyx;
            }
        } else {
            if (rank == 4) {
                default_fmt = cldnn::format::oiyx;
            } else if (rank == 5) {
                default_fmt = cldnn::format::oizyx;
            }
        }
    } else {
        if (rank == 5) {
            default_fmt = cldnn::format::bfzyx;
        } else if (rank == 6) {
            default_fmt = cldnn::format::bfwzyx;
        } else if (rank == 7) {
            default_fmt = cldnn::format::bfuwzyx;
        } else if (rank == 8) {
            default_fmt = cldnn::format::bfvuwzyx;
        }
    }
    return default_fmt;
}

bool format::is_default_format(type fmt) {
    return fmt == get_default_format(dimension(fmt));
}

format format::adjust_to_rank(format fmt, size_t new_rank) {
    // TODO: remove as soon as rank extension is not needed anymore
    new_rank = std::max<size_t>(new_rank, 4);

    auto current_traits = format::traits(fmt);
    auto current_order = current_traits._order;
    auto current_blocking = current_traits.block_sizes;
    auto current_rank = current_order.size();
    if (new_rank == current_rank)
        return fmt;

    auto is_adjustable = [](const format& fmt) -> bool {
        return !format::is_weights_format(fmt) &&
               !format::is_image_2d(fmt) &&
               !format::is_winograd(fmt) &&
               fmt != format::b_fs_yx_32fp;
    };

    // Skip special formats as order + blocking desc may be not enough to properly match them
    OPENVINO_ASSERT(is_adjustable(fmt), "Format ", fmt, " is not adjustable");

    auto align_order = [](std::vector<size_t>& order, size_t current_rank, size_t new_rank) {
        auto max_element_it = std::max_element(order.begin(), order.end());
        for (size_t i = current_rank; i < new_rank; i++) {
            max_element_it = std::next(max_element_it);
            max_element_it = order.insert(max_element_it, i);
        }
    };

    if (new_rank > current_rank) {
        align_order(current_order, current_rank, new_rank);
    }

    for (auto& kv : format_traits_map) {
        auto candidate_tag = kv.first;
        auto candidate_traits = kv.second;
        auto candidate_order = candidate_traits._order;
        auto candidate_blocking = candidate_traits.block_sizes;
        auto candidate_rank = candidate_traits.order.size();

        if (candidate_rank != new_rank || !is_adjustable(candidate_tag))
            continue;

        bool same_blocking_scheme = candidate_blocking == current_blocking;
        bool same_dims_scheme = current_traits.batch_num == candidate_traits.batch_num &&
                                current_traits.group_num == candidate_traits.group_num &&
                                current_traits.feature_num == candidate_traits.feature_num;

        if (!same_blocking_scheme || !same_dims_scheme)
            continue;

        if (current_rank > candidate_rank) {
            align_order(candidate_order, candidate_rank, current_rank);
        }

        if (candidate_order == current_order)
            return candidate_tag;
    }

    OPENVINO_ASSERT(false, "Can't adjust format ", fmt.to_string(), " to the new rank (", new_rank, ")");
}

// First : block_idx, Second : block_size
const std::vector<std::pair<size_t, int>> format::per_axis_block_size(format fmt) {
    std::vector<std::pair<size_t, int>> sizes_for_dims;
    for (const auto& block : fmt.block_sizes()) {
        auto it = std::find_if(sizes_for_dims.begin(), sizes_for_dims.end(),
                [&block](const std::pair<size_t, int>& ele) { return ele.first == block.first; });
        if (it != sizes_for_dims.end())
            it->second *= block.second;  // the axis is double blocked
        else
            sizes_for_dims.push_back({block.first, block.second});
    }

    return sizes_for_dims;
}

format format::find_format(const std::vector<uint64_t>& order,
                           const std::vector<std::pair<size_t, int>>& block_sizes,
                           bool is_weights,
                           bool is_grouped,
                           bool is_image_2d,
                           bool is_winograd,
                           bool is_nv12) {
    auto is_suitable_traits = [&](const std::pair<format::type, format_traits>& traits) -> bool {
        return traits.second._order == order &&
               traits.second.block_sizes == block_sizes &&
               format::is_weights_format(traits.first) == is_weights &&
               format::is_grouped(traits.first) == is_grouped &&
               format::is_image_2d(traits.first) == is_image_2d &&
               format::is_winograd(traits.first) == is_winograd &&
               format::is_nv12(traits.first) == is_nv12;
    };

    std::vector<format> finded_formats;
    for (auto& traits : format_traits_map) {
        if (is_suitable_traits(traits))
            finded_formats.emplace_back(traits.first);
    }

    OPENVINO_ASSERT(!finded_formats.empty(), "[GPU] Cannot find a format with the specified parameters");
    OPENVINO_ASSERT(finded_formats.size() == 1, "[GPU] Cannot find a format. Specified parameters are ambiguous");

    return finded_formats.front();
}

}  // namespace cldnn
