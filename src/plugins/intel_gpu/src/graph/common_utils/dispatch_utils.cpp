// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dispatch_utils.hpp"

namespace ov::intel_gpu {

using namespace cldnn;

std::vector<size_t> get_optimal_lws(const std::vector<size_t>& gws, const cldnn::device_info& info,
                                    cldnn::format input_fmt, cldnn::format output_fmt,
                                    std::vector<std::vector<ChannelName>> dims_by_gws) {
    enum axis { x, y, z, w, u, v, f, b, unused_axis };

    // GWS/LWS priority order should be considered for better local WGS setting
    // and as a result more optimized data reading/writing inside kernels
    std::vector<size_t> priority_order = { 0, 1, 2 };
    std::vector<size_t> layout_order = { x, y, z, w, u, v, f, b };

    const size_t gws_dims_num = priority_order.size();
    const size_t axis_num = layout_order.size();
    size_t first_axis_idx = 0;

    std::vector<size_t> axis_by_gws = { unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis };
    for (size_t gws_idx = 0; gws_idx < gws_dims_num; gws_idx++) {
        for (size_t axis_idx = 0; axis_idx < dims_by_gws[gws_idx].size(); axis_idx++) {
            axis_by_gws[static_cast<size_t>(dims_by_gws[gws_idx][axis_idx])] = gws_idx;
        }
    }

    auto calculate_optimized_priority_order = [&]() -> void {
        while (axis_by_gws[layout_order[first_axis_idx]] == unused_axis)
            first_axis_idx++;

        for (size_t gws_idx = 0; gws_idx < gws_dims_num; gws_idx++) {
            for (size_t axis_idx = first_axis_idx; axis_idx < axis_num; axis_idx++) {
                if (axis_by_gws[layout_order[axis_idx]] != unused_axis) {
                    bool is_already_exists = false;
                    if (axis_idx > 0) {
                        for (int i = static_cast<int>(axis_idx) - 1; i >= 0; i--) {
                            if (axis_by_gws[layout_order[axis_idx]] == axis_by_gws[layout_order[i]]) {
                                is_already_exists = true;
                                break;
                            }
                        }
                    }
                    first_axis_idx++;
                    if (!is_already_exists) {
                        priority_order[gws_idx] = axis_by_gws[layout_order[axis_idx]];
                        break;
                    }
                }
            }
        }
    };

    auto one_layout = input_fmt == output_fmt;

    auto simple_planar_layout = cldnn::format::is_simple_data_format(output_fmt);

    auto blocked_fsv_layout = output_fmt == format::b_fs_yx_fsv2 || output_fmt == format::b_fs_zyx_fsv2 ||
                              output_fmt == format::b_fs_yx_fsv4 || output_fmt == format::b_fs_zyx_fsv4 ||
                              output_fmt == format::b_fs_yx_fsv8 || output_fmt == format::b_fs_zyx_fsv8 ||
                              output_fmt == format::b_fs_yx_fsv16 || output_fmt == format::b_fs_zyx_fsv16 ||
                              output_fmt == format::b_fs_yx_fsv32 || output_fmt == format::b_fs_zyx_fsv32 ||
                              output_fmt == format::fs_b_yx_fsv32;

    auto blocked_bsv_fsv_layout = output_fmt == format::bs_fs_yx_bsv16_fsv2 || output_fmt == format::bs_fs_zyx_bsv16_fsv2 ||
                                  output_fmt == format::bs_fs_yx_bsv16_fsv4 || output_fmt == format::bs_fs_zyx_bsv16_fsv4 ||
                                  output_fmt == format::bs_fs_yx_bsv16_fsv8 || output_fmt == format::bs_fs_zyx_bsv16_fsv8 ||
                                  output_fmt == format::bs_fs_yx_bsv16_fsv16 || output_fmt == format::bs_fs_yx_bsv16_fsv32 ||
                                  output_fmt == format::bs_fs_yx_bsv32_fsv16 || output_fmt == format::bs_fs_yx_bsv32_fsv32 ||
                                  output_fmt == format::bs_fs_zyx_bsv16_fsv16 || output_fmt == format::bs_fs_zyx_bsv16_fsv32 ||
                                  output_fmt == format::bs_fs_zyx_bsv32_fsv16 || output_fmt == format::bs_fs_zyx_bsv32_fsv32;

    auto try_change_priority_order = (simple_planar_layout || blocked_fsv_layout || blocked_bsv_fsv_layout) && one_layout;

    if (try_change_priority_order) {
        if (simple_planar_layout) {
            switch (output_fmt) {
                case format::bfyx:
                    layout_order = { x, y, f, b, z, w, u, v };
                    break;
                case format::yxfb:
                    layout_order = { b, f, x, y, z, w, u, v };
                    break;
                case format::byxf:
                    layout_order = { f, x, y, b, z, w, u, v };
                    break;
                case format::byfx:
                    layout_order = { x, f, y, b, z, w, u, v };
                    break;
                case format::bxfy:
                    layout_order = { y, f, x, b, z, w, u, v };
                    break;
                case format::fbyx:
                    layout_order = { x, y, b, f, z, w, u, v };
                    break;
                case format::fyxb:
                    layout_order = { b, x, y, f, z, w, u, v };
                    break;
                case format::bfxy:
                    layout_order = { y, x, f, b, z, w, u, v };
                    break;
                case format::bfzyx:
                    layout_order = { x, y, z, f, b, w, u, v };
                    break;
                case format::bzyxf:
                    layout_order = { f, x, y, z, b, w, u, v };
                    break;
                case format::bfwzyx:
                    layout_order = { x, y, z, w, f, b, u, v };
                    break;
                case format::bfuwzyx:
                    layout_order = { x, y, z, w, u, f, b, v };
                    break;
                case format::bfvuwzyx:
                    layout_order = { x, y, z, w, u, v , f, b };
                    break;
                default:
                    layout_order = { x, y, z, w, u, v, f, b };
                    break;
            }
        } else if (blocked_fsv_layout) {
            if (output_fmt == format::b_fs_yx_fsv2 || output_fmt == format::b_fs_yx_fsv4 || output_fmt == format::b_fs_yx_fsv8 ||
                output_fmt == format::b_fs_yx_fsv16 || output_fmt == format::b_fs_yx_fsv32) {
                layout_order = { f, x, y, b, z, w, u, v };
            } else if (output_fmt == format::b_fs_zyx_fsv2 || output_fmt == format::b_fs_zyx_fsv4 || output_fmt == format::b_fs_zyx_fsv8 ||
                       output_fmt == format::b_fs_zyx_fsv16 || output_fmt == format::b_fs_zyx_fsv32) {
                layout_order = { f, x, y, z, b, w, u, v };
            } else { // output_fmt == format::fs_b_yx_fsv32
                layout_order = { f, x, y, b, z, w, u, v };
            }
        } else if (blocked_bsv_fsv_layout) {
            layout_order = { f, b, x, y, z, w, u, v };
        }

        calculate_optimized_priority_order();

        // Revert basic priority if something is wrong
        if (priority_order[0] == priority_order[1] || priority_order[0] == priority_order[2] || priority_order[1] == priority_order[2] ||
            priority_order[0] > 2 || priority_order[1] > 2 || priority_order[2] > 2) {
            priority_order[0] = 0;
            priority_order[1] = 1;
            priority_order[2] = 2;
        }
    }

    size_t lws_max = info.max_work_group_size;
    const size_t optimal_lws_values[] = { 1024, 960, 896, 832, 768, 704, 640, 576,
                                          512, 480, 448, 416, 384, 352, 320, 288,
                                          256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1 };
    const size_t suboptimal_lws_values[] = { 1024, 960, 896, 832, 768, 704, 640, 576,
                                             512, 480, 448, 416, 384, 352, 320, 288,
                                             256, 227, 224, 192, 160, 128, 96, 64, 32, 16,
                                             15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    size_t first_lws_idx = lws_max == 1024 ? 0:
                           lws_max == 512 ?  8:
                                            16;
    // Reduces max local wgs for some cases on Gen12+ devices
    if (lws_max >= 512) {
        auto two_dims_are_odd_and_equal = (gws[0] % 2 && gws[0] > 7 && (gws[0] == gws[1] || gws[0] == gws[2])) ||
                                          (gws[1] % 2 && gws[1] > 7 && gws[1] == gws[2]);

        // Known cases when lws_max = 256 works better than lws_max > 256
        auto max_wgs_exception1 = gws[priority_order[0]] == 1278 && gws[priority_order[1]] == 718 && gws[priority_order[2]] % 10 == 0;
        auto max_wgs_exception2 = gws[priority_order[0]] == 28 && gws[priority_order[1]] == 168 && gws[priority_order[2]] == 128;
        auto max_wgs_exception3 = gws[priority_order[0]] == 1000 && gws[priority_order[1]] == 1 && gws[priority_order[2]] == 64;
        auto max_wgs_exception4 = gws[priority_order[0]] == 180 && gws[priority_order[1]] == 320 && gws[priority_order[2]] == 56;
        auto max_wgs_exception5 = gws[priority_order[0]] == 1 && gws[priority_order[1]] > 256 && gws[priority_order[2]] == 1;
        auto max_wgs_exception6 = gws[priority_order[0]] == 64 && gws[priority_order[1]] == 16 && gws[priority_order[2]] == 1 &&
                                  priority_order[1] == 2 && priority_order[2] == 1;
        if (two_dims_are_odd_and_equal || max_wgs_exception1 || max_wgs_exception2 || max_wgs_exception3 || max_wgs_exception4 ||
            max_wgs_exception5 || max_wgs_exception6) {
            lws_max = 256;
            first_lws_idx = 16;
        }
    }

    size_t total_lws = 1;
    size_t total_gws = 1;
    std::vector<size_t> lws = { 1, 1, 1 };

    for (size_t i = 0; i < gws.size(); ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = first_lws_idx;
        size_t max_optimal_lws0_value = lws_max;
        if (try_change_priority_order && axis_by_gws[f] != unused_axis) {
            if (output_fmt == format::b_fs_yx_fsv16 || output_fmt == format::b_fs_zyx_fsv16 || output_fmt == format::fs_b_yx_fsv32) {
                max_optimal_lws0_value = 16;
            } else if (output_fmt == format::b_fs_yx_fsv32 || output_fmt == format::b_fs_zyx_fsv32) {
                max_optimal_lws0_value = 32;
            } else if ((output_fmt == format::bs_fs_yx_bsv16_fsv16 || output_fmt == format::bs_fs_zyx_bsv16_fsv16) &&
                       (axis_by_gws[b] == axis_by_gws[f])) {
                max_optimal_lws0_value = 256;
            } else if ((output_fmt == format::bs_fs_yx_bsv16_fsv16 || output_fmt == format::bs_fs_zyx_bsv16_fsv16) &&
                       (axis_by_gws[b] != axis_by_gws[f]) && (axis_by_gws[b] != unused_axis)) {
                max_optimal_lws0_value = 16;
            } else if ((output_fmt == format::bs_fs_yx_bsv32_fsv32 || output_fmt == format::bs_fs_zyx_bsv32_fsv32) &&
                       (axis_by_gws[b] != axis_by_gws[f]) && (axis_by_gws[b] != unused_axis)) {
                max_optimal_lws0_value = 32;
            }
        }

        auto can_use_suboptimal_lws1 = (i == 1) && ((gws[priority_order[0]] % 32 == 0) || (gws[priority_order[0]] == 1 && gws[priority_order[2]] % 16 != 0));
        auto can_use_suboptimal_lws2 = (i == 2) && (total_lws == total_gws);
        const size_t* lws_values = can_use_suboptimal_lws1 || can_use_suboptimal_lws2 ?
                                   suboptimal_lws_values :
                                   optimal_lws_values;

        while (rest_lws < lws_values[lws_idx]) lws_idx++;
        if (i == 0) {
            while (lws_values[lws_idx] > max_optimal_lws0_value) lws_idx++;
        }
        while (gws[priority_order[i]] % lws_values[lws_idx]) lws_idx++;

        // else statement cannot be interpreted, it causes dg2 perf degradation, so added dg2(1024 lws_max) in if statement
        if (lws_max == 256 || lws_max == 1024 || total_lws == total_gws) {
            lws[priority_order[i]] = lws_values[lws_idx];
        } else {
            lws[priority_order[i]] = i == 2 && gws[priority_order[0]] != 1 ? 1 : lws_values[lws_idx];
            if (total_gws > 100 && total_lws < 8 && i == 2)
                lws[priority_order[i]] = lws_values[lws_idx];
        }

        total_lws *= lws_values[lws_idx];
        total_gws *= gws[priority_order[i]];
    }

    // For cases with lws { 1, 1, 1 } try to use suboptimal values to increase work group size
    if (lws[0] == 1 && lws[1] == 1 && lws[2] == 1) {
        total_lws = 1;
        for (size_t i = 0; i < gws.size(); ++i) {
            auto rest_lws = lws_max / total_lws;
            size_t lws_idx = first_lws_idx;

            const size_t* lws_values = suboptimal_lws_values;

            while (rest_lws < lws_values[lws_idx]) lws_idx++;
            while (gws[priority_order[i]] % lws_values[lws_idx]) lws_idx++;

            lws[priority_order[i]] = lws_values[lws_idx];

            total_lws *= lws_values[lws_idx];
        }
    }

    return lws;
}

}  // namespace ov::intel_gpu
