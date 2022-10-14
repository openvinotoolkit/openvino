// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "program_helpers.h"
#include "binary_convolution_inst.h"
#include "mvn_inst.h"
#include "to_string_utils.h"
#include "pooling_inst.h"
#include "reshape_inst.h"

#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>
#include <tuple>

using namespace cldnn;

// ToDo remove friendship relation from program

reorder_inputs::reorder_inputs(layout_optimizer& lo_ref, reorder_factory& rf_ref) : base_pass("reorder_inputs"), _lo(lo_ref), _rf(rf_ref) {}

void reorder_inputs::run(program& p) { run(p, _lo, _rf); }

namespace {

std::map<program_node*, format::type> get_preferred_formats(program& p, layout_optimizer& lo) {
    GPU_DEBUG_GET_INSTANCE(debug_config);

    std::map<program_node*, format::type> fmt_map;

#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t onednn_impls_counter = 0;
    size_t all_impls_counter = 0;
    const float onednn_min_threshold = 0.09f;
    bool should_update_fmt_map = false;

    // Calculate onednn kernels number and all kernels number inside the network
    for (auto n : p.get_processing_order()) {
        if (!n->is_in_data_flow())
            continue;

        auto ex = lo.get_preferred_format(*n);
        auto impl = lo.get_preferred_impl_type(*n, ex);
        fmt_map[n] = ex;

        n->set_preferred_impl_type(impl);

        if (impl == impl_types::onednn)
            onednn_impls_counter++;

        all_impls_counter++;
    }

    float onednn_usage_ratio = all_impls_counter ? static_cast<float>(onednn_impls_counter) / static_cast<float>(all_impls_counter) : 0.f;

    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        GPU_DEBUG_COUT << "----------------------------------------------" << std::endl;
        GPU_DEBUG_COUT << "Onednn kernels number: " << onednn_impls_counter << " from " << all_impls_counter
                       << " (" << onednn_usage_ratio * 100.f << "%)" << std::endl;
        GPU_DEBUG_COUT << "Onednn usage threshold: " << onednn_min_threshold * 100.f << "%" << std::endl;
    }

    // Reverted to cldnn way for cases when onednn kernels number inside the whole network is extremely low =>
    // improvements from onednn usage less than losses due to unoptimized formats for cldnn kernels, extra reorders, etc.
    if (onednn_usage_ratio < onednn_min_threshold && lo.get_optimization_attributes().use_onednn_impls) {
        should_update_fmt_map = true;
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, 0);
        GPU_DEBUG_IF(debug_config->verbose >= 1) {
            GPU_DEBUG_COUT << "The return to clDNN implementations" << std::endl;
        }
    }

    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        GPU_DEBUG_COUT << "----------------------------------------------" << std::endl;
    }
#endif // ENABLE_ONEDNN_FOR_GPU

#ifdef ENABLE_ONEDNN_FOR_GPU
    if (should_update_fmt_map)
#endif
    {
        for (auto n : p.get_processing_order()) {
            if (!n->is_in_data_flow())
                continue;

            auto ex = lo.get_preferred_format(*n);
            auto impl = lo.get_preferred_impl_type(*n, ex);
            fmt_map[n] = ex;

            n->set_preferred_impl_type(impl);
        }
    }
    return fmt_map;
}

enum class direction_e {
    forwards = 0,
    backwards = 1
};

inline constexpr direction_e reverse(direction_e dir) {
    return dir == direction_e::forwards ? direction_e::backwards : direction_e::forwards;
}

template <direction_e dir = direction_e::forwards>
struct travel_direction_wrapper {
    static const std::list<program_node*>& next_nodes(program_node* node) {
        return node->get_users();
    }

    template <typename T>
    static T& first(T& current, T& /*next*/) { return current; }

    template <typename T>
    static T& second(T& /*current*/, T& next) { return next; }
};

template <>
struct travel_direction_wrapper<direction_e::backwards> {
    static const std::vector<program_node*>& next_nodes(program_node* node) {
        return node->get_dependencies();
    }

    template <typename T>
    static T& first(T& /*current*/, T& next) { return next; }

    template <typename T>
    static T& second(T& current, T& /*next*/) { return current; }
};

template <direction_e dir>
bool can_propagate_formats_rec(
    const std::map<program_node*, format::type>& fmt_map,
    layout_optimizer& lo,
    program_node* prev,
    program_node* node,
    format::type fmt) {

    auto sel_fmt = fmt_map.at(node);
    if (fmt == sel_fmt)
        return true;

    auto first_node = travel_direction_wrapper<dir>::first(prev, node);
    auto second_node = travel_direction_wrapper<dir>::second(prev, node);
    auto first_fmt = travel_direction_wrapper<dir>::first(fmt, sel_fmt);
    auto second_fmt = travel_direction_wrapper<dir>::second(fmt, sel_fmt);

    if (lo.can_fuse_reorder(*first_node,
                            *second_node,
                            first_fmt,
                            second_fmt))
        return true;

    if (sel_fmt != format::any)
        return false;

    if (!lo.is_format_supported(*node, fmt))
        return false;

    auto reverse_reorders = std::count_if(
        travel_direction_wrapper<reverse(dir)>::next_nodes(node).begin(),
        travel_direction_wrapper<reverse(dir)>::next_nodes(node).end(),
        [&](program_node* rev) {
        return rev->is_in_data_flow() && fmt_map.at(rev) != fmt && rev != prev;
    });

    if (reverse_reorders > 0)
        return false;

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!next->is_in_data_flow())
            continue;
        if (!can_propagate_formats_rec<dir>(fmt_map, lo, node, next, fmt))
            return false;
    }

    return true;
}

template <direction_e dir>
void propagate_formats_rec(std::map<program_node*, format::type>& fmt_map,
                           layout_optimizer& lo,
                           program_node* prev,
                           program_node* node,
                           format::type fmt) {
    auto sel_fmt = fmt_map.at(node);
    if (sel_fmt == fmt)
        return;

    auto first_node = travel_direction_wrapper<dir>::first(prev, node);
    auto second_node = travel_direction_wrapper<dir>::second(prev, node);
    auto first_fmt = travel_direction_wrapper<dir>::first(fmt, sel_fmt);
    auto second_fmt = travel_direction_wrapper<dir>::second(fmt, sel_fmt);

    if (lo.can_fuse_reorder(*first_node,
                            *second_node,
                            first_fmt,
                            second_fmt))
        return;

    fmt_map.at(node) = fmt;

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!next->is_in_data_flow())
            continue;
        propagate_formats_rec<dir>(fmt_map, lo, node, next, fmt);
    }
}

template <direction_e dir>
void propagate_formats_in_dir(std::map<program_node*, format::type>& fmt_map,
                         layout_optimizer& lo,
                         program_node* node) {
    auto fmt = fmt_map.at(node);

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!next->is_in_data_flow())
            continue;
        if (!can_propagate_formats_rec<dir>(fmt_map, lo, node, next, fmt))
            return;
    }

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!next->is_in_data_flow())
            continue;
        propagate_formats_rec<dir>(fmt_map, lo, node, next, fmt);
    }
}

void propagate_formats(program& p, std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo) {
    auto it = p.get_processing_order().begin();
    while (it != p.get_processing_order().end()) {
        auto node = *it++;

        if (fmt_map.count(node) == 0 || fmt_map.at(node) == format::any)
            continue;

        propagate_formats_in_dir<direction_e::forwards>(fmt_map, lo, node);
        propagate_formats_in_dir<direction_e::backwards>(fmt_map, lo, node);
    }
}

struct reorder_cnt {
    size_t number;
    size_t total_sizes;
};

template <direction_e dir>
reorder_cnt count_reorders_in_dir(const std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo, program_node* node) {
    size_t cnt = 0;
    size_t size = 0;
    auto sel_fmt = fmt_map.at(node);

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!next->is_in_data_flow())
            continue;

        auto next_fmt = fmt_map.at(next);

        if (next_fmt == format::any ||
            (sel_fmt != next_fmt &&
             !lo.can_fuse_reorder(*travel_direction_wrapper<dir>::first(node, next),
                                  *travel_direction_wrapper<dir>::second(node, next),
                                  travel_direction_wrapper<dir>::first(sel_fmt, next_fmt),
                                  travel_direction_wrapper<dir>::second(sel_fmt, next_fmt)))) {
            cnt += 1;
            auto l = travel_direction_wrapper<dir>::first(node, next)->get_output_layout();
            if (l.is_static())
                size += l.count();
        }
    }

    return { cnt, size };
}

reorder_cnt count_reorders(const std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo, program_node* node) {
    auto fwd = count_reorders_in_dir<direction_e::forwards>(fmt_map, lo, node);
    auto bwd = count_reorders_in_dir<direction_e::backwards>(fmt_map, lo, node);

    return { fwd.number + bwd.number, fwd.total_sizes + bwd.total_sizes };
}

void minimize_local_reorders(program& p, std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo) {
    for (auto node : p.get_processing_order()) {
        if (!node->is_in_data_flow())
            continue;

        auto preferred_format = lo.get_preferred_format(*node);

        if (preferred_format != format::any) {
            if (preferred_format == format::b_fs_yx_fsv4 &&
                (node->get_output_layout().data_type == data_types::i8 || node->get_output_layout().data_type == data_types::u8)) {
                std::set<format::type> io_formats;
                for (auto user : node->get_users()) {
                    io_formats.insert(fmt_map.at(user));
                }
                for (auto dep : node->get_dependencies()) {
                    if (!dep->is_in_data_flow())
                        continue;
                    io_formats.insert(fmt_map.at(dep));
                }
                if (!(io_formats.size() == 1 && io_formats.count(preferred_format) == 0))
                    continue;
            } else {
                continue;
            }
        }

        if (fmt_map.at(node) == format::any) {
            auto out_fmt = node->get_output_layout().format;
            if (lo.is_format_supported(*node, out_fmt)) {
                fmt_map.at(node) = out_fmt;
            }
        }

        auto sel_fmt = fmt_map.at(node);
        auto best_reorder_cnt = count_reorders(fmt_map, lo, node);
        auto best_format = sel_fmt;

        if (best_reorder_cnt.number == 0)
            continue;

        std::set<format::type> local_formats;

        for (auto user : node->get_users()) {
            auto user_fmt = fmt_map.at(user);

            if (user_fmt != format::any &&
                lo.is_format_supported(*node, user_fmt)) {
                local_formats.insert(user_fmt);
            }
        }

        for (auto dep : node->get_dependencies()) {
            if (!dep->is_in_data_flow())
                continue;

            auto dep_fmt = fmt_map.at(dep);

            if (dep_fmt != format::any &&
                lo.is_format_supported(*node, dep_fmt)) {
                local_formats.insert(dep_fmt);
            }
        }

        if (local_formats.empty())
            continue;

        for (auto new_fmt : local_formats) {
            fmt_map.at(node) = new_fmt;

            auto reorders_cnt = count_reorders(fmt_map, lo, node);

            if (reorders_cnt.number < best_reorder_cnt.number ||
                (reorders_cnt.number == best_reorder_cnt.number && reorders_cnt.total_sizes < best_reorder_cnt.total_sizes)) {
                best_reorder_cnt = reorders_cnt;
                best_format = new_fmt;
            }
        }

        fmt_map.at(node) = best_format;
    }
}

static format get_target_output_format(layout_optimizer& lo, const std::map<program_node*, format::type>& fmt_map, program_node *node) {
    // 1. Check required_output
    if (lo.get_optimization_attributes().use_onednn_impls) {
        // If onednn is not used, need to ignore get_required_layout result as it is from onednn
        auto ret = node->get_required_output();

        if (ret != format::any)
            return ret;
    }

    // 2. Check fmt
    if (fmt_map.count(node) > 0)
        return fmt_map.at(node);

    // 3. Use output_layout
    return node->get_output_layout().format;
}

static format get_target_input0_format(layout_optimizer& lo, const std::map<program_node*, format::type>& fmt_map, program_node *node) {
    // 1. Check required_input
    if (lo.get_optimization_attributes().use_onednn_impls) {
        // If onednn is not used, need to ignore get_required_layout result as it is from onednn
        auto ret = node->get_required_input0();
        if (ret != format::any)
            return ret;
    }

    // 2. Check fmt
    if (fmt_map.count(node) > 0)
        return fmt_map.at(node);

    // 3. Use output_layout
    return node->get_output_layout().format;
}

const char *dir_msg(direction_e dir) {
    if (dir == direction_e::forwards)
        return "forward";
    else
        return "backward";
}

// If there is layout mismatch between two layers, add reorder
template <direction_e dir>
void insert_reorders_in_dir(program& p, const std::map<program_node*, format::type>& fmt_map, reorder_factory& rf, layout_optimizer& lo, program_node* node) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto fmt = fmt_map.at(node);

    auto next_cpy = travel_direction_wrapper<dir>::next_nodes(node);
    for (auto next : next_cpy) {
        if (!next->is_in_data_flow())
            continue;

        if (fmt_map.count(next) > 0 && fmt_map.at(next) == fmt)
            continue;

        // We have three (potentially) conflicting information here for format
        //    node->get_output_layout().format : It is not up-to-date at this moment. It is just the default format (bfyx)
        //    fmt_map.at(node).format          : It is queried with get_preferred_layout. However, it has only output format.
        //    node.get_required_input0/output  : If it is valid(!= any), it is up-to-date. It has input format, too.
        // So the priority is required_input0/output --> fmt_map --> output_layout().format

        auto predecessor = travel_direction_wrapper<dir>::first(node, next);
        auto successor = travel_direction_wrapper<dir>::second(node, next);
        auto in_layout = predecessor->get_output_layout();
        auto out_layout = in_layout;
        in_layout.format = get_target_output_format(lo, fmt_map, predecessor);
        auto target_input0_format = get_target_input0_format(lo, fmt_map, successor);

        for (auto& fused_prim : successor->get_fused_primitives()) {
            // If it is input of fused node, use output layout instead of input layout
            if (successor->get_dependencies().size() <= fused_prim.dep_start_idx)
                continue;
            auto& dependency = successor->get_dependency(fused_prim.dep_start_idx);
            if (&dependency == predecessor) {
                target_input0_format = get_target_output_format(lo, fmt_map, successor);
                GPU_DEBUG_IF(debug_config->verbose >= 2) {
                    GPU_DEBUG_COUT << __func__ << ":" << __LINE__ << ": Use output format of successor " << successor->id() << " : "
                                << fmt_to_str(target_input0_format) << std::endl;
                }
                break;
            }
        }

        out_layout.format = target_input0_format;
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << __func__ << ":" << __LINE__ << ":" << dir_msg(dir) << "  " << node->id() << " --> " << next->id() << " ## "
                    << fmt_to_str(in_layout.format) << " --> " << fmt_to_str(out_layout.format) << std::endl;
        }

        auto reorder_pair = rf.get_reorder(travel_direction_wrapper<dir>::first(node, next)->id(),
                                           in_layout,
                                           out_layout);
        auto reorder = reorder_pair.first;

        if (reorder && (in_layout.format != format::any && out_layout.format != format::any)) {
            auto& reorder_node = p.get_or_create(reorder);
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                GPU_DEBUG_COUT << __func__ << ":" << __LINE__ << ":" << dir_msg(dir) << "  " << reorder_node.id()
                        << "  Reorder is added" << std::endl;
            }
            p.add_intermediate(reorder_node,
                               *travel_direction_wrapper<dir>::second(node, next),
                               *travel_direction_wrapper<dir>::first(node, next),
                               !reorder_pair.second);
        }
    }
}

void insert_reorders(program& p, const std::map<program_node*, format::type>& fmt_map, reorder_factory& rf, layout_optimizer& lo) {
    auto fwd_it = p.get_processing_order().begin();
    while (fwd_it != p.get_processing_order().end()) {
        auto node = *(fwd_it++);

        if (fmt_map.count(node) != 1)
            continue;

        auto fmt = fmt_map.at(node);
        if (fmt == format::any || format::is_image(fmt))
            continue;

        insert_reorders_in_dir<direction_e::forwards>(p, fmt_map, rf, lo, node);
    }

    auto bwd_it = p.get_processing_order().rbegin();
    while (bwd_it != p.get_processing_order().rend()) {
        auto node = *(bwd_it++);

        if (fmt_map.count(node) != 1)
            continue;

        auto fmt = fmt_map.at(node);
        if (fmt == format::any || format::is_image(fmt))
            continue;

        insert_reorders_in_dir<direction_e::backwards>(p, fmt_map, rf, lo, node);
    }
}

}  // namespace

void reorder_inputs::run(program& p, layout_optimizer& lo, reorder_factory& rf) {
    GPU_DEBUG_GET_INSTANCE(debug_config);

    auto fmt_map = get_preferred_formats(p, lo);

    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "[clDNN][reorder_inputs] Preferred formats:" << std::endl;
        for (auto& node_fmt : fmt_map) {
            if (node_fmt.second != format::any) {
                GPU_DEBUG_COUT << "[clDNN][reorder_inputs]   " << node_fmt.first->id() << " " << fmt_to_str(node_fmt.second) << std::endl;
            }
        }
    }

    propagate_formats(p, fmt_map, lo);
    minimize_local_reorders(p, fmt_map, lo);

    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "[clDNN][reorder_inputs] Selected formats:" << std::endl;
        for (auto node_ptr : p.get_processing_order()) {
            if (fmt_map.count(node_ptr) == 0)
                continue;

            auto fmt = fmt_map.at(node_ptr);
            GPU_DEBUG_COUT << "[clDNN][reorder_inputs]   " << node_ptr->id() << " " << fmt_to_str(fmt) << std::endl;
        }
    }

    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        reorder_cnt total_reorder_count = std::accumulate(
            p.get_processing_order().begin(),
            p.get_processing_order().end(),
            reorder_cnt{ 0, 0 },
            [&](reorder_cnt& total, program_node* node) {
            if (fmt_map.count(node) == 0 || fmt_map.at(node) == format::any)
                return total;
            auto count = count_reorders(fmt_map, lo, node);
            return reorder_cnt{ total.number + count.number, total.total_sizes + count.total_sizes };
        });
        // Divide results by two as above function will each reorder from both sides
        GPU_DEBUG_COUT << "[clDNN][reorder_inputs] Total number of reorders: " << total_reorder_count.number / 2 << std::endl;
        GPU_DEBUG_COUT << "[clDNN][reorder_inputs] Total elements count of all reorders: " << total_reorder_count.total_sizes / 2 << std::endl;

        // Count number of reorders that will be fused
        size_t nodes_with_fusing = 0;
        for (auto node_ptr : p.get_processing_order()) {
            if (fmt_map.count(node_ptr) == 0 || fmt_map.at(node_ptr) == format::any)
                continue;
            for (auto prev_ptr : travel_direction_wrapper<direction_e::backwards>::next_nodes(node_ptr)) {
                if (!prev_ptr->is_in_data_flow() || fmt_map.at(prev_ptr) == fmt_map.at(node_ptr))
                    continue;
                if (lo.can_fuse_reorder(*prev_ptr, *node_ptr, fmt_map.at(prev_ptr), fmt_map.at(node_ptr))) {
                    nodes_with_fusing += 1;
                    break;
                }
            }
        }
        GPU_DEBUG_COUT << "[clDNN][reorder_inputs] Number of nodes with fused reorders: " << nodes_with_fusing << std::endl;
        GPU_DEBUG_COUT << "----------------------------------------------" << std::endl;
    }

    insert_reorders(p, fmt_map, rf, lo);

    for (auto n : p.get_processing_order()) {
        n->recalc_output_layout(true);
    }

    const auto reorder_input_detection_output = [&p, &rf](typed_program_node<detection_output>& detection_output_node) {
        if (detection_output_node.get_preferred_impl_type() == impl_types::cpu) {
            auto detection_output_prim = detection_output_node.get_primitive();

            for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++) {
                auto& input = detection_output_node.get_dependency(i);
                auto new_input = rf.get_reorder(input.id(),
                                                input.get_output_layout(),
                                                layout{ data_types::f32, format::bfyx, input.get_output_layout().get_tensor() });

                if (new_input.first) {
                    p.add_intermediate(new_input.first, detection_output_node, i, !new_input.second);
                }
            }
        }
    };

    const auto reorder_input_binary_convolution = [&p, &rf](typed_program_node<binary_convolution>& binary_conv_node) {
        auto& input = binary_conv_node.input();
        auto input_layout = input.get_output_layout();
        auto new_layout = input_layout;
        new_layout.data_type = data_types::bin;

        auto reorder = rf.get_reorder(input.id(), input_layout, new_layout);

        if (reorder.first) {
            p.add_intermediate(reorder.first, binary_conv_node, 0, !reorder.second);
        }
    };

    const auto reorder_input_and_weights_deconvolution = [&p, &lo, &rf](typed_program_node<deconvolution>& deconv_node) {
        auto& input = deconv_node.input();
        auto input_layout = input.get_output_layout();
        auto new_format = lo.get_preferred_format(deconv_node);
        if (new_format == format::b_fs_zyx_fsv16 || new_format == format::bs_fs_zyx_bsv16_fsv16) {
            auto reorder = rf.get_reorder(input.id(), input_layout,
                layout{ input_layout.data_type, new_format, input_layout.get_tensor() });
            if (reorder.first) {
                p.add_intermediate(reorder.first, deconv_node, 0, !reorder.second);
            }
        }

        auto& weights = deconv_node.weights();
        auto weights_layout = weights.get_output_layout();
        if (!format::is_simple_data_format(weights_layout.format) && !weights.is_type<data>() && !weights.is_constant()) {
            auto dims = weights_layout.format.dimension();
            auto preferred_format = dims <= 4 ? format::bfyx : dims == 5 ? format::bfzyx : format::bfwzyx;
            auto reorder = rf.get_reorder(weights.id(), weights_layout,
                layout{ weights_layout.data_type, preferred_format, weights_layout.get_tensor() });
            if (reorder.first) {
                p.add_intermediate(reorder.first, deconv_node, 1, !reorder.second);
                p.get_or_create(reorder.first).recalc_output_layout(false);
            }
        }
    };

    const auto reorder_convolution = [&p, &lo, &rf, &debug_config](typed_program_node<convolution>& conv_node) {
        {
            // reorder weights convolution
            auto& weights = conv_node.weights();
            auto weights_layout = weights.get_output_layout();
            if (!format::is_simple_data_format(weights_layout.format) && !weights.is_type<data>() && !weights.is_constant()) {
                auto dims = weights_layout.format.dimension();
                auto preferred_format = dims <= 4 ? format::bfyx : dims == 5 ? format::bfzyx : format::bfwzyx;
                auto reorder = rf.get_reorder(weights.id(), weights_layout,
                    layout{ weights_layout.data_type, preferred_format, weights_layout.get_tensor() });
                if (reorder.first) {
                    p.add_intermediate(reorder.first, conv_node, 1, !reorder.second);
                    p.get_or_create(reorder.first).recalc_output_layout(false);
                }
            }
        }

        {
            // Change input data type of conv node from i32 to f32
            auto& input = conv_node.input();
            auto input_layout = input.get_output_layout();
            if (input_layout.data_type == data_types::i32) {
                auto new_layout = input_layout;
                new_layout.data_type = data_types::f32;
                auto new_input = rf.get_reorder(input.id(), input_layout, new_layout);
                if (new_input.first) {
                    p.add_intermediate(new_input.first, conv_node, 0, !new_input.second);
                    p.get_or_create(new_input.first).recalc_output_layout(true);
                }
            }

            // Change weights type i32 to f32
            auto& weights = conv_node.weights();
            auto weights_layout = weights.get_output_layout();
            if (weights_layout.data_type == data_types::i32) {
                auto new_layout = weights_layout;
                new_layout.data_type = data_types::f32;
                auto new_input = rf.get_reorder(weights.id(), weights_layout, new_layout);
                if (new_input.first) {
                    p.add_intermediate(new_input.first, conv_node, 1, !new_input.second);
                    p.get_or_create(new_input.first).recalc_output_layout(false);
                }
            }
        }

        // For supporting optimized onednn first conv, the input format from prev reorder to this conv is changed to a recommended format by onednn.
        auto& input = conv_node.input();
        auto input_layout = input.get_output_layout();

        if (conv_node.impl_type == impl_types::onednn && input_layout.format != conv_node.get_required_input0()) {
            auto new_layout = input_layout;
            new_layout.format = conv_node.get_required_input0();
            auto new_input = rf.get_reorder(input.id(), input_layout, new_layout);
            if (new_input.first)
                p.add_intermediate(new_input.first, conv_node, 0, !new_input.second);
        }

        // When the conv node is of onednn impl type and eltwise sum with full tensor is fused,
        // changes the input format of eltwise sum post-op to use binary add.
        if (conv_node.get_preferred_impl_type() == impl_types::onednn) {
            onednn_add_fusing_helpers::for_eltwise(conv_node, eltwise_mode::sum,
                [&](const program_node& p_node, const fused_primitive_desc& desc) {
                    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(p_node, desc);
                    if (fusing_type == add_fusing_type::binary_per_tensor) {
                        auto& dep_node = p_node.get_dependency(desc.dep_start_idx);
                        auto d_layout = dep_node.get_output_layout();
                        auto d_format = d_layout.format;
                        auto expected_format = format::any;

                        if (data_type_traits::is_i8_u8(d_layout.data_type)) {
                            if (d_format == format::b_fs_yx_fsv16)
                                expected_format = format::b_fs_yx_fsv32;
                            else if (d_format == format::bs_fs_yx_bsv32_fsv16)
                                expected_format = format::bs_fs_yx_bsv32_fsv32;
                        } else if (data_type_traits::is_floating_point(d_layout.data_type)) {
                            if (d_format == format::b_fs_yx_fsv32)
                                expected_format = format::b_fs_yx_fsv16;
                            else if (d_format == format::bs_fs_yx_bsv32_fsv32)
                                expected_format = format::bs_fs_yx_bsv32_fsv16;
                        }

                        if (expected_format != format::any && d_layout.format != expected_format) {
                            auto new_layout = d_layout;
                            new_layout.format = expected_format;
                            auto new_input = rf.get_reorder(dep_node.id(), d_layout, new_layout);
                            if (new_input.first) {
                                p.add_intermediate(new_input.first, conv_node, desc.dep_start_idx, !new_input.second);
                            }
                            conv_node.get_dependency(desc.dep_start_idx).set_output_layout(new_layout, false);
                        }
                    }
                });
        }
    };

    const auto reorder_input_fully_connected = [&p, &lo, &rf](typed_program_node<fully_connected>& fc_node) {
        auto& weights = fc_node.weights();
        auto& input = fc_node.input();
        auto input_layout = input.get_output_layout();
        // Change input data type of fully-connected node from i32 to f32
        if (input_layout.data_type == data_types::i32) {
            auto new_layout = input_layout;
            new_layout.data_type = data_types::f32;
            auto new_input = rf.get_reorder(input.id(), input_layout, new_layout);
            if (new_input.first) {
               p.add_intermediate(new_input.first, fc_node, 0);
            }
        }

        // Change weights type i32 to f32
        auto weights_layout = weights.get_output_layout();
        if (weights_layout.data_type == data_types::i32) {
            auto new_layout = weights_layout;
            new_layout.data_type = data_types::f32;
            auto new_input = rf.get_reorder(weights.id(), weights_layout, new_layout);
            if (new_input.first) {
               p.add_intermediate(new_input.first, fc_node, 1);
            }
        }
    };

    const auto reorder_input_pooling = [&p, &rf](typed_program_node<pooling>& pooling_node) {
        // Change input data type of pooling node from i32 to f32
        auto& input = pooling_node.input();
        auto input_layout = input.get_output_layout();
        if (pooling_node.get_primitive()->mode == pooling_mode::max && input_layout.data_type == data_types::i32) {
            auto new_layout = input_layout;
            new_layout.data_type = data_types::f32;
            auto new_input = rf.get_reorder(input.id(), input_layout, new_layout);
            if (new_input.first) {
               p.add_intermediate(new_input.first, pooling_node, 0);
            }
        }
    };

    for (auto& prim : p.get_processing_order()) {
        program_helpers::do_for_types<detection_output, binary_convolution, deconvolution, convolution, fully_connected, pooling>(
            *prim,
            reorder_input_detection_output,
            reorder_input_binary_convolution,
            reorder_input_and_weights_deconvolution,
            reorder_convolution,
            reorder_input_fully_connected,
            reorder_input_pooling);
    }

    for (auto n : p.get_processing_order()) {
        if (n->is_in_data_flow() && fmt_map.count(n) != 0) {
            n->get_output_layout(); // There might be some invalid output layout
            auto preferred_impl = lo.get_preferred_impl_type(*n, fmt_map.at(n));
            n->set_preferred_impl_type(preferred_impl);
        }
    }
}
