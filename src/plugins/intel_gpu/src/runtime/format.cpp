// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/format.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

const format_traits& format::traits(type fmt) {
    #define FMT_TRAITS(fmt, ...) {fmt, {#fmt, __VA_ARGS__}}

    static const std::map<type, format_traits> traits {
            // B - number of Batch dimensions
            // F - number of Feature dimensions
            // S - number of Spatial dimensions
            // G - number of Group dimensions
            // Order - dims changing order from rare to often
            // Inner order - dims order for internal storage in _sizes array
            // Block sizes - vector of pairs of dimension number (by inner order) and block size ordered from rare to often
            //         Format                 B  F  S  G   Dims order         Order   Inner order Block sizes
            FMT_TRAITS(yxfb,                  1, 1, 2, 0, {2, 3, 1, 0},       "yxfb",   "bfxy?",  {}),
            FMT_TRAITS(byxf,                  1, 1, 2, 0, {0, 2, 3, 1},       "byxf",   "bfxy?",  {}),
            FMT_TRAITS(bfyx,                  1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}),
            FMT_TRAITS(fyxb,                  1, 1, 2, 0, {1, 2, 3, 0},       "fyxb",   "bfxy?",  {}),
            FMT_TRAITS(b_fs_yx_fsv16,         1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy",   {{1, 16}}),
            FMT_TRAITS(b_fs_yx_fsv32,         1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy",   {{1, 32}}),
            FMT_TRAITS(b_fs_zyx_fsv32,        1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{1, 32}}),
            FMT_TRAITS(bs_xs_xsv8_bsv8,       1, 0, 1, 0, {0, 1},             "bx",     "b?x??",  {{2, 8}, {0, 8}}),
            FMT_TRAITS(bs_xs_xsv8_bsv16,      1, 0, 1, 0, {0, 1},             "bx",     "b?x??",  {{2, 8}, {0, 16}}),
            FMT_TRAITS(bs_x_bsv16,            1, 1, 1, 0, {0, 1},             "bx",     "b?x??",  {{0, 16}}),
            FMT_TRAITS(winograd_2x3_s1_data,  1, 1, 2, 0, {0, 2, 3, 1},       "bxyf",   "bfxy?",  {}),
            FMT_TRAITS(b_fs_yx_fsv4,          1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{1, 4}}),
            FMT_TRAITS(bfzyx,                 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {}),
            FMT_TRAITS(bfwzyx,                1, 1, 4, 0, {0, 1, 2, 3, 4, 5}, "bfwzyx", "bfxyzw", {}),
            FMT_TRAITS(fs_b_yx_fsv32,         1, 1, 2, 0, {1, 0, 2, 3},       "fbyx",   "bfxy?",  {{1, 32}}),
            FMT_TRAITS(b_fs_yx_32fp,          1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}),
            FMT_TRAITS(b_fs_zyx_fsv16,        1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{1, 16}}),
            FMT_TRAITS(bs_fs_zyx_bsv16_fsv16, 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 16 }, {1, 16}}),
            FMT_TRAITS(bs_fs_yx_bsv16_fsv16,  1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 16 }, {1, 16}}),
            FMT_TRAITS(bs_fs_yx_bsv4_fsv4,    1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 4 }, {1, 4}}),
            FMT_TRAITS(bs_fs_yx_bsv8_fsv4,    1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 8 }, {1, 4}}),
            FMT_TRAITS(bs_fs_yx_bsv8_fsv2,    1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 8 }, {1, 2}}),
            FMT_TRAITS(bs_fs_yx_bsv4_fsv2,    1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 4 }, {1, 2}}),
            FMT_TRAITS(bs_fs_zyx_bsv4_fsv4,   1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 4 }, {1, 4}}),
            FMT_TRAITS(bs_fs_zyx_bsv4_fsv2,   1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 4 }, {1, 2}}),
            FMT_TRAITS(bs_fs_zyx_bsv32_fsv32, 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 32 }, {1, 32}}),
            FMT_TRAITS(bs_fs_zyx_bsv32_fsv16, 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 32 }, {1, 16}}),
            FMT_TRAITS(bs_fs_yx_bsv32_fsv32,  1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 32 }, {1, 32}}),
            FMT_TRAITS(bs_fs_yx_bsv32_fsv16,  1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 32 }, {1, 16}}),
            FMT_TRAITS(nv12,                  1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}),
            FMT_TRAITS(image_2d_rgba,         1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}),

            FMT_TRAITS(oiyx,                                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {}),
            FMT_TRAITS(ioyx,                                         1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",  {}),
            FMT_TRAITS(iyxo,                                         1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {}),
            FMT_TRAITS(oyxi,                                         1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",  {}),
            FMT_TRAITS(yxio,                                         1, 1, 2, 0, {2, 3, 1, 0},    "yxio",   "oixy?", {}),
            FMT_TRAITS(oizyx,                                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {}),
            FMT_TRAITS(iozyx,                                        1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {}),
            FMT_TRAITS(os_is_yx_isv16_osv16,                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 16}, {0, 16}}),
            FMT_TRAITS(o_is_yx_isv16,                                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{1, 16}}),
            FMT_TRAITS(os_yxi_osv16,                                 1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy?", {{0, 16}}),
            FMT_TRAITS(os_iyx_osv16,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{0, 16}}),
            FMT_TRAITS(os_iyx_osv32,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{0, 32}}),
            FMT_TRAITS(os_iyx_osv64,                                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{0, 64}}),
            FMT_TRAITS(winograd_2x3_s1_weights,                      1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(winograd_2x3_s1_fused_weights,                1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?", {}),
            FMT_TRAITS(winograd_6x3_s1_fused_weights,                1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?", {}),
            FMT_TRAITS(image_2d_weights_winograd_6x3_s1_fbxyb,       1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?", {}),
            FMT_TRAITS(image_2d_weights_winograd_6x3_s1_xfbyb,       1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?", {}),
            FMT_TRAITS(image_2d_weights_c4_fyx_b,                    1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(image_2d_weights_c1_b_fyx,                    1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(lstm_weights_dio,                             1, 1, 2, 0, {0, 1, 3, 2},    "oixy",   "oixy?", {}),
            FMT_TRAITS(os_is_yx_isa8_osv8_isv4,                      1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(os_is_yx_isa8_osv16_isv4,                     1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(os_is_yx_isa8_osv8_isv4_swizzled_by_4,        1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {}),
            FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv2,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{0, 32}, {1, 16}}),
            FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv4,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 32}}),
            FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv2,                1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 16}}),
            FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv4,                1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 32}}),
            FMT_TRAITS(os_is_yx_osa2_isa8_osv16_isv2,                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 16}}),
            FMT_TRAITS(os_is_yx_osa2_isa8_osv16_isv4,                1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}, {1, 32}}),
            FMT_TRAITS(os_is_yx_osa2_isa8_osv8_isv2,                 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 16}, {1, 16}}),
            FMT_TRAITS(os_is_zyx_isa8_osv8_isv4,                     1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 8}, {1, 4}}),
            FMT_TRAITS(os_is_zyx_isa8_osv16_isv4,                    1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 16}, {1, 4}}),
            FMT_TRAITS(os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?", {{0, 32}, {1, 32}}),
            FMT_TRAITS(os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 32}}),
            FMT_TRAITS(is_os_yx_isa2_osa8_isv8_osv2,                 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "ioxy?", {{1, 16}, {0, 16}}),
            FMT_TRAITS(is_os_yx_isa4_osa8_isv8_osv4,                 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "ioxy?", {{1, 32}, {0, 32}}),
            FMT_TRAITS(is_o_yx_isv32,                                1, 1, 2, 0, {1, 0, 2, 3},    "oyxi",   "oixy?", {{1, 32}}),
            FMT_TRAITS(is_o32_yx_isv32_swizzled_by_4,                1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?", {}),
            FMT_TRAITS(os_is_y_x8_osv8_isv4,                         1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?", {}),
            FMT_TRAITS(os_is_y_x8_osv8_isv4_swizzled_by_4,           1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?", {}),
            FMT_TRAITS(os_is_yx_osv16_isv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?", {{0, 16}, {1, 4}}),
            FMT_TRAITS(os_is_yx_osv8_isv4,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 4}, {0, 8}}),
            FMT_TRAITS(os_is_yx_osv8_isv2,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 2}, {0, 8}}),
            FMT_TRAITS(os_is_zyx_osv16_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 16}, {1, 16}}),
            FMT_TRAITS(os_is_yx_osv32_isv4_swizzled_by_2,            1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?", {{0, 32}, {1, 4}}),
            FMT_TRAITS(os_is_yx_osv32_isv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?", {{0, 32}, {1, 4}}),
            FMT_TRAITS(os_is_zyx_osv32_isv4,                         1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 4}}),
            FMT_TRAITS(os_is_yx_osv32_isv32p,                        1, 1, 1, 0, {0, 1, 2, 3},    "oixy",   "oixy?", {}),
            FMT_TRAITS(os_is_zyx_isv16_osv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 16}, {1, 16}}),
            FMT_TRAITS(is_os_zyx_isv16_osv16,                        1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz", {{1, 16}, {0, 16}}),
            FMT_TRAITS(is_os_yx_isv16_osv16,                         1, 1, 2, 0, {1, 0, 2, 3, 4}, "ioyx",   "oixy",  {{1, 16}, {0, 16}}),
            FMT_TRAITS(os_is_osv32_isv32_swizzled_by_4,              1, 1, 0, 0, {0, 1, 2, 3},    "oixy",   "oixy?", {{0, 32}, {1, 32}}),
            FMT_TRAITS(os_is_zyx_isv8_osv16_isv2,                    1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{1, 8}, {0, 16}, {1, 2}}),
            FMT_TRAITS(os_zyxi_osv16,                                1, 1, 3, 0, {0, 2, 3, 4, 1}, "ozyxi",  "oixyz", {{0, 16}}),
            FMT_TRAITS(os_is_yx_isv8_osv16_isv2,                     1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 8}, {0, 16}, {1, 2}}),
            FMT_TRAITS(os_is_yx_osv16_isv16,                         1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{1, 16}, {0, 16}}),
            FMT_TRAITS(os_is_zyx_osv32_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 32}, {1, 16}}),
            FMT_TRAITS(os_is_zyx_osv64_isv16,                        1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz", {{0, 64}, {1, 16}}),
            FMT_TRAITS(os_iyx_osv32__ai32,                           1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 32}}),
            FMT_TRAITS(i_yxs_os_yxsv2_osv16,                         1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{0, 16}}),
            FMT_TRAITS(iy_xs_os_xsv2_osv8__ao32,                     1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{2, 2}, {0, 8}}),
            FMT_TRAITS(iy_xs_os_xsv2_osv16__ao32,                    1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",  {{2, 2}, {0, 16}}),
            FMT_TRAITS(os_i_yxs_osv4_yxsv4,                          1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",  {{0, 4}}),
            FMT_TRAITS(os_i_osv16__ai8,                              1, 1, 0, 0, {0, 1},          "oi",     "oi??",  {{1, 8}, {0, 16}}),
            FMT_TRAITS(os_i_osv8__ai8,                               1, 1, 0, 0, {0, 1},          "oi",     "oi??",  {{1, 8}, {0, 8}}),

            FMT_TRAITS(goiyx,                                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {}),
            FMT_TRAITS(gioyx,                                        1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy??g", {}),
            FMT_TRAITS(goizyx,                                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {}),
            FMT_TRAITS(giozyx,                                       1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz?g", {}),
            FMT_TRAITS(g_os_iyx_osv16,                               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 16}}),
            FMT_TRAITS(g_os_iyx_osv32,                               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 32}}),
            FMT_TRAITS(gs_oiyx_gsv16,                                1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{6, 16}}),
            FMT_TRAITS(gs_oizyx_gsv16,                               1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{6, 16}}),
            FMT_TRAITS(gs_oiyx_gsv32,                                1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{6, 32}}),
            FMT_TRAITS(gyxio,                                        1, 1, 2, 1, {0, 3, 4, 2, 1},    "gyxio",  "oixy??g", {}),
            FMT_TRAITS(g_is_os_zyx_isv16_osv16,                      1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz?g", {{1, 16}, {0, 16}}),
            FMT_TRAITS(g_is_os_yx_isv16_osv16,                       1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy??g", {{1, 16}, {0, 16}}),
            FMT_TRAITS(g_os_is_zyx_isv8_osv16_isv2,                  1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{1, 8}, {0, 16}, {1, 2}}),
            FMT_TRAITS(g_os_is_yx_isv8_osv16_isv2,                   1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{1, 8}, {0, 16}, {1, 2}}),
            FMT_TRAITS(g_os_is_zyx_isv16_osv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{0, 16}, {1, 16}}),
            FMT_TRAITS(g_os_is_yx_osv16_isv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goixy",  "oixy??g", {{0, 16}, {1, 4}}),
            FMT_TRAITS(g_os_is_zyx_osv16_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{0, 16}, {1, 16}}),
            FMT_TRAITS(g_os_zyx_is_osv16_isv4,                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 16}, {1, 4}}),
            FMT_TRAITS(g_os_zyx_is_osv16_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 16}, {1, 16}}),
            FMT_TRAITS(g_os_zyx_is_osv16_isv32,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 16}, {1, 32}}),
            FMT_TRAITS(g_os_zyx_is_osv32_isv4,                       1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 32}, {1, 4}}),
            FMT_TRAITS(g_os_zyx_is_osv32_isv16,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 32}, {1, 16}}),
            FMT_TRAITS(g_os_zyx_is_osv32_isv32,                      1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g", {{0, 32}, {1, 32}}),
            FMT_TRAITS(g_os_is_yx_osa2_isa8_osv8_isv2,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 16}, {1, 16}}),
            FMT_TRAITS(g_os_is_yx_osa4_isa8_osv8_isv4,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 32}, {1, 32}}),
            FMT_TRAITS(g_os_is_zyx_osa4_isa8_osv8_isv4,              1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{0, 32}, {1, 32}}),
            FMT_TRAITS(g_os_is_yx_osa4_isa8_osv8_isv2,               1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 32}, {1, 16}}),
            FMT_TRAITS(g_os_is_zyx_osa4_isa8_osv8_isv2,              1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g", {{0, 32}, {1, 16}}),
            FMT_TRAITS(g_os_is_yx_osa2_isa8_osv16_isv4,              1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 32}, {1, 32}}),
            FMT_TRAITS(g_os_is_yx_osa2_isa8_osv16_isv2,              1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{0, 32}, {1, 16}}),
            FMT_TRAITS(gs_oi_yxs_gsv4_yxsv4,                         1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{6, 4}}),
            FMT_TRAITS(gs_oi_yxs_gsv16_yxsv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{6, 16}}),
            FMT_TRAITS(gs_oi_yxs_gsv32_yxsv4,                        1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{6, 32}}),
            FMT_TRAITS(g_os_is_yx_isv16_osv16,                       1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g", {{1, 16}, {0, 16}}),
            FMT_TRAITS(gi_yxs_os_yxsv2_osv16,                        1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g", {{0, 16}}),
            FMT_TRAITS(giy_xs_os_xsv2_osv8__ao32,                    1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g", {{2, 2}, {0, 8}}),
            FMT_TRAITS(giy_xs_os_xsv2_osv16__ao32,                   1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g", {{2, 2}, {0, 16}}),
    };
    if (traits.find(fmt) == traits.end()) {
        throw std::runtime_error("[GPU] Format description is missing in fmt traits");
    }
    return traits.at(fmt);
}

std::string format::to_string() const {
    if (value == any) {
        return "any";
    }
    return traits(value).str;
}

}  // namespace cldnn
