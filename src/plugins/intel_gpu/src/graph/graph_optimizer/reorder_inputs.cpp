// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_helpers.h"
#include "to_string_utils.h"
#include "pooling_inst.h"
#include "fully_connected_inst.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "gemm_inst.h"
#include "broadcast_inst.h"
#include <impls/onednn/utils.hpp>
#endif

#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>

using namespace cldnn;

// ToDo remove friendship relation from program

reorder_inputs::reorder_inputs(reorder_factory& rf_ref) : base_pass("reorder_inputs"), _rf(rf_ref) {}

void reorder_inputs::run(program& p) { run(p, _rf); }

namespace {

std::map<program_node*, format::type> get_preferred_formats(program& p, layout_optimizer& lo) {
    std::map<program_node*, format::type> fmt_map;

#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t onednn_impls_counter = 0;
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
    }

    if (!lo.is_empty_onednn_impls_optimization_attribute() && onednn_impls_counter < 1) {
        should_update_fmt_map = true;
        lo.clear_onednn_impls_optimization_attribute();
        GPU_DEBUG_LOG << "Disable oneDNN implementations globally" << std::endl;
    }

    if (should_update_fmt_map)
#endif // ENABLE_ONEDNN_FOR_GPU
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
    static const T& first(const T& current, const T& /*next*/) { return current; }

    template <typename T>
    static const T& second(const T& /*current*/, const T& next) { return next; }
};

template <>
struct travel_direction_wrapper<direction_e::backwards> {
    static const std::vector<std::pair<program_node*, int32_t>>& next_nodes(program_node* node) {
        return node->get_dependencies();
    }

    template <typename T>
    static const T& first(const T& /*current*/, const T& next) { return next; }

    template <typename T>
    static const T& second(const T& current, const T& /*next*/) { return current; }
};

static inline program_node* get_node(program_node *node)                             { return node; }

static inline program_node* get_node(const std::pair<program_node *, int32_t> &node) { return node.first; }

static format get_target_output_format(layout_optimizer& lo, const std::map<program_node*, format::type>& fmt_map, program_node *node, program_node *next) {
    auto user_idx = next->get_dependency_output_port(*node);

    // 1. Check selected preferred_output_format
    auto ret = node->get_preferred_output_fmt(user_idx);
    if (ret != format::any)
        return ret;

    // 2. Check fmt
    if (fmt_map.count(node) > 0)
        return fmt_map.at(node);

    // 3. Use output_layout
    return node->get_output_layout().format;
}

static format get_target_input_format(layout_optimizer& lo, const std::map<program_node*, format::type>& fmt_map, program_node *node, program_node *prev) {
    auto dep_idx = node->get_dependency_index(*prev);

    // 1. Check selected preferred_input_format
    auto ret = node->get_preferred_input_fmt(dep_idx);
    if (ret != format::any)
        return ret;

    // 2. Check fmt
    if (fmt_map.count(node) > 0)
        return fmt_map.at(node);

    // 3. Use output_layout
    return node->get_output_layout().format;
}


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

    auto predecessor = travel_direction_wrapper<dir>::first(prev, node);
    auto successor = travel_direction_wrapper<dir>::second(prev, node);
    auto first_fmt = get_target_output_format(lo, fmt_map, predecessor, successor);
    auto second_fmt = get_target_input_format(lo, fmt_map, successor, predecessor);

    if (lo.can_fuse_reorder(*predecessor,
                            *successor,
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
        [&](auto rev) {
        return get_node(rev)->is_in_data_flow() && fmt_map.at(get_node(rev)) != fmt && get_node(rev) != prev;
    });

    if (reverse_reorders > 0)
        return false;

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!get_node(next)->is_in_data_flow())
            continue;
        if (!can_propagate_formats_rec<dir>(fmt_map, lo, node, get_node(next), fmt))
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

    auto predecessor = travel_direction_wrapper<dir>::first(prev, node);
    auto successor = travel_direction_wrapper<dir>::second(prev, node);
    auto first_fmt = get_target_output_format(lo, fmt_map, predecessor, successor);
    auto second_fmt = get_target_input_format(lo, fmt_map, successor, predecessor);

    if (lo.can_fuse_reorder(*predecessor,
                            *successor,
                            first_fmt,
                            second_fmt))
        return;

    fmt = travel_direction_wrapper<dir>::first(first_fmt, second_fmt);
    fmt_map.at(node) = fmt;
    GPU_DEBUG_LOG << "Propagate_formats_rec: " << node->id() << " - " << fmt_to_str(fmt) << std::endl;

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!get_node(next)->is_in_data_flow())
            continue;
        if (!can_propagate_formats_rec<dir>(fmt_map, lo, node, get_node(next), fmt))
            continue;
        propagate_formats_rec<dir>(fmt_map, lo, node, get_node(next), fmt);
    }
}

template <direction_e dir>
void propagate_formats_in_dir(std::map<program_node*, format::type>& fmt_map,
                              layout_optimizer& lo,
                              program_node* node) {
    auto fmt = fmt_map.at(node);

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!get_node(next)->is_in_data_flow())
            continue;
        if (!can_propagate_formats_rec<dir>(fmt_map, lo, node, get_node(next), fmt))
            return;
    }

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!get_node(next)->is_in_data_flow())
            continue;
        propagate_formats_rec<dir>(fmt_map, lo, node, get_node(next), fmt);
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

    for (auto next : travel_direction_wrapper<dir>::next_nodes(node)) {
        if (!get_node(next)->is_in_data_flow())
            continue;

        auto predecessor = travel_direction_wrapper<dir>::first(node, get_node(next));
        auto successor = travel_direction_wrapper<dir>::second(node, get_node(next));
        auto first_fmt = get_target_output_format(lo, fmt_map, predecessor, successor);
        auto second_fmt = get_target_input_format(lo, fmt_map, successor, predecessor);

        if (second_fmt == format::any ||
            (first_fmt != second_fmt &&
             !lo.can_fuse_reorder(*predecessor,
                                  *successor,
                                  first_fmt, second_fmt))) {
            cnt += 1;
            auto l = travel_direction_wrapper<dir>::first(node, get_node(next))->get_output_layout();
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
                for (const auto& dep : node->get_dependencies()) {
                    if (!dep.first->is_in_data_flow())
                        continue;
                    io_formats.insert(fmt_map.at(dep.first));
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
            auto user_fmt = get_target_input_format(lo, fmt_map, user, node);

            if (user_fmt != format::any &&
                lo.is_format_supported(*node, user_fmt)) {
                local_formats.insert(user_fmt);
            }
        }

        for (const auto& dep : node->get_dependencies()) {
            if (!dep.first->is_in_data_flow())
                continue;

            auto dep_fmt = get_target_output_format(lo, fmt_map, dep.first, node);

            if (dep_fmt != format::any &&
                lo.is_format_supported(*node, dep_fmt)) {
                local_formats.insert(dep_fmt);
            }
        }

        if (local_formats.empty())
            continue;

        for (auto new_fmt : local_formats) {
            // Avoid setting of formats which will require transform from higher rank to smaller one which requires dimension squeeze
            // TODO: Needs to be updated once we improve layout assignment logic
            if (fmt_map.at(node) != format::any && format::dimension(fmt_map.at(node)) > format::dimension(new_fmt))
                continue;
            fmt_map.at(node) = new_fmt;

            auto reorders_cnt = count_reorders(fmt_map, lo, node);

            if (reorders_cnt.number < best_reorder_cnt.number ||
                (reorders_cnt.number == best_reorder_cnt.number && reorders_cnt.total_sizes < best_reorder_cnt.total_sizes
                                                                && !node->get_output_layout().is_dynamic())) {
                best_reorder_cnt = reorders_cnt;
                best_format = new_fmt;
            }
        }

        fmt_map.at(node) = best_format;
    }
}


const char *dir_msg(direction_e dir) {
    if (dir == direction_e::forwards)
        return "forward";
    else
        return "backward";
}

static bool is_weights_dependency(program_node* predecessor, program_node* successor) {
    bool is_weights_dep = false;
    if (successor->is_type<convolution>() || successor->is_type<deconvolution>() || successor->is_type<fully_connected>()) {
        size_t dep_idx = successor->get_dependency_index(*predecessor);
        is_weights_dep = dep_idx == successor->get_primitive()->input_size();
    }
    return is_weights_dep;
}

// If there is layout mismatch between two layers, add reorder
template <direction_e dir>
void insert_reorders_in_dir(program& p, const std::map<program_node*, format::type>& fmt_map, reorder_factory& rf, layout_optimizer& lo, program_node* node) {
    auto next_cpy = travel_direction_wrapper<dir>::next_nodes(node);
    for (auto next : next_cpy) {
        if (!get_node(next)->is_in_data_flow())
            continue;

        // We have three (potentially) conflicting information here for format
        //    node->get_output_layout().format : It is not up-to-date at this moment. It is just the default format (bfyx)
        //    fmt_map.at(node).format          : It is queried with get_preferred_layout. However, it has only output format.
        //    node.get_preferred_output_fmt    : If it is valid(!= any), it is up-to-date.
        // So the priority is preferred_input/output_format --> fmt_map --> output_layout().format
        auto predecessor = travel_direction_wrapper<dir>::first(node, get_node(next));
        auto successor = travel_direction_wrapper<dir>::second(node, get_node(next));
        if (is_weights_dependency(predecessor, successor))
            continue;
        auto port_idx = successor->get_dependency_output_port(*predecessor);
        auto in_layout = predecessor->get_output_layout(false, port_idx);
        auto out_layout = in_layout;

        in_layout.format = get_target_output_format(lo, fmt_map, predecessor, successor);
        out_layout.format = get_target_input_format(lo, fmt_map, successor, predecessor);
        if (in_layout.format == out_layout.format)
            continue;

        GPU_DEBUG_LOG << dir_msg(dir) << "  " << node->id() << " --> " << get_node(next)->id() << " ## "
                      << fmt_to_str(in_layout.format) << " --> " << fmt_to_str(out_layout.format) << std::endl;

        if (in_layout.format == format::any || out_layout.format == format::any ||
            in_layout.format == format::custom || out_layout.format == format::custom)
            continue;

        auto reorder_pair = rf.get_reorder(predecessor->id(),
                                           port_idx,
                                           in_layout,
                                           out_layout);

        auto reorder = reorder_pair.first;
        if (reorder && (in_layout.format != format::any && out_layout.format != format::any)) {
            auto& reorder_node = p.get_or_create(reorder);
            GPU_DEBUG_LOG << dir_msg(dir) << "  " << reorder_node.id() << "  Reorder is added" << std::endl;
            p.add_intermediate(reorder_node,
                               *travel_direction_wrapper<dir>::second(node, get_node(next)),
                               *travel_direction_wrapper<dir>::first(node, get_node(next)),
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

void reorder_inputs::run(program& p, reorder_factory& rf) {
    auto& lo = p.get_layout_optimizer();

    auto fmt_map = get_preferred_formats(p, lo);

    GPU_DEBUG_LOG_PASS << "Preferred formats:" << std::endl;
    for (auto& node_fmt : fmt_map) {
        if (node_fmt.second != format::any) {
            GPU_DEBUG_LOG_PASS << "  " << node_fmt.first->id() << " " << fmt_to_str(node_fmt.second) << std::endl;
        }
    }

    propagate_formats(p, fmt_map, lo);
    minimize_local_reorders(p, fmt_map, lo);

    GPU_DEBUG_LOG_PASS << "Selected formats:" << std::endl;
    for (auto node_ptr : p.get_processing_order()) {
        if (fmt_map.count(node_ptr) == 0)
            continue;

        auto fmt = fmt_map.at(node_ptr);
        GPU_DEBUG_LOG_PASS << "  " << node_ptr->id() << " " << fmt_to_str(fmt) << std::endl;
    }

    GPU_DEBUG_IF(p.get_config().get_verbose() >= 2) {
        reorder_cnt total_reorder_count =
            std::accumulate(p.get_processing_order().begin(),
                            p.get_processing_order().end(),
                            reorder_cnt{0, 0},
                            [&](reorder_cnt total, program_node* node) {
                                if (fmt_map.count(node) == 0 || fmt_map.at(node) == format::any)
                                    return total;
                                auto count = count_reorders(fmt_map, lo, node);
                                return reorder_cnt{total.number + count.number, total.total_sizes + count.total_sizes};
                            });
        // Divide results by two as above function will each reorder from both sides
        GPU_DEBUG_LOG_PASS << "Total number of reorders: " << total_reorder_count.number / 2 << std::endl;
        GPU_DEBUG_LOG_PASS << "Total elements count of all reorders: " << total_reorder_count.total_sizes / 2 << std::endl;

        // Count number of reorders that will be fused
        size_t nodes_with_fusing = 0;
        for (auto node_ptr : p.get_processing_order()) {
            if (fmt_map.count(node_ptr) == 0 || fmt_map.at(node_ptr) == format::any)
                continue;
            for (const auto& prev_ptr : travel_direction_wrapper<direction_e::backwards>::next_nodes(node_ptr)) {
                if (!prev_ptr.first->is_in_data_flow() || fmt_map.at(prev_ptr.first) == fmt_map.at(node_ptr))
                    continue;
                if (lo.can_fuse_reorder(*prev_ptr.first, *node_ptr, fmt_map.at(prev_ptr.first), fmt_map.at(node_ptr))) {
                    nodes_with_fusing += 1;
                    break;
                }
            }
        }
        GPU_DEBUG_LOG_PASS << "Number of nodes with fused reorders: " << nodes_with_fusing << std::endl;
        GPU_DEBUG_LOG_PASS << "----------------------------------------------" << std::endl;
    }

    insert_reorders(p, fmt_map, rf, lo);

    for (auto n : p.get_processing_order()) {
        n->recalc_output_layouts(true);
    }

    const auto reorder_input_detection_output = [&p, &rf](typed_program_node<detection_output>& detection_output_node) {
        if (detection_output_node.get_preferred_impl_type() == impl_types::cpu) {
            auto detection_output_prim = detection_output_node.get_primitive();

            for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++) {
                auto& input = detection_output_node.get_dependency(i);
                auto input_layout = input.get_output_layout();
                auto new_input = rf.get_reorder(input.id(),
                                                input_layout,
                                                layout{ input_layout.get_partial_shape(), data_types::f32, format::bfyx });

                if (new_input.first) {
                    p.add_intermediate(new_input.first, detection_output_node, i, !new_input.second);
                    detection_output_node.recalc_output_layouts();
                }
            }
        }
    };

    const auto reorder_input_and_weights_deconvolution = [&p, &lo, &rf](typed_program_node<deconvolution>& deconv_node) {
        auto& input = deconv_node.input();
        auto input_layout = input.get_output_layout();
        auto new_format = lo.get_preferred_format(deconv_node);
        if (new_format == format::b_fs_zyx_fsv16 || new_format == format::bs_fs_zyx_bsv16_fsv16) {
            auto reorder = rf.get_reorder(input.id(), input_layout,
                layout{ input_layout.get_partial_shape(), input_layout.data_type, new_format });
            if (reorder.first) {
                p.add_intermediate(reorder.first, deconv_node, 0, !reorder.second);
                deconv_node.recalc_output_layouts();
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
                p.get_or_create(reorder.first).recalc_output_layouts(false);
            }
        }
    };

    const auto reorder_convolution = [&p, &rf](typed_program_node<convolution>& conv_node) {
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
                    p.get_or_create(reorder.first).recalc_output_layouts(false);
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
                    p.get_or_create(new_input.first).recalc_output_layouts(true);
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
                    p.get_or_create(new_input.first).recalc_output_layouts(false);
                }
            }
        }

        // For supporting optimized onednn first conv, the input format from prev reorder to this conv is changed to a recommended format by onednn.
        auto& input = conv_node.input();
        auto input_layout = input.get_output_layout();
        if (conv_node.impl_type == impl_types::onednn && input_layout.format != conv_node.get_preferred_input_fmt()) {
            // Data input format does NOT match with an output format of previous node
            auto new_layout = input_layout;
            new_layout.format = conv_node.get_preferred_input_fmt();
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
                    if (fusing_type == add_fusing_type::binary_per_tensor && desc.has_outer_dep()) {
                        auto& dep_node = p_node.get_dependency(desc.outer_dep_start_idx);
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
                                p.add_intermediate(new_input.first, conv_node, desc.outer_dep_start_idx, !new_input.second);
                            }
                            conv_node.get_dependency(desc.outer_dep_start_idx).set_output_layout(new_layout, false);
                        }
                    }
                });
        }
    };

    const auto reorder_input_fully_connected = [&p, &rf](typed_program_node<fully_connected>& fc_node) {
        auto& weights = fc_node.weights();
        auto& input = fc_node.input();
        auto input_layout = input.get_output_layout();
        // Change input data type of fully-connected node from i32 to f32
        if (input_layout.data_type == data_types::i32) {
            auto new_layout = input_layout;
            new_layout.data_type = data_types::f32;
            auto new_input = rf.get_reorder(input.id(), input_layout, new_layout);
            if (new_input.first) {
               p.add_intermediate(new_input.first, fc_node, 0, !new_input.second);
               fc_node.recalc_output_layouts();
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
        auto dep = pooling_node.get_dependency_with_port(0);
        const auto& input = dep.first;
        auto input_layout = input->get_output_layout();
        if (pooling_node.get_primitive()->mode == pooling_mode::max && input_layout.data_type == data_types::i32) {
            auto new_layout = input_layout;
            new_layout.data_type = data_types::f32;
            auto new_input = rf.get_reorder(input->id(), dep.second, input_layout, new_layout);
            if (new_input.first) {
               p.add_intermediate(new_input.first, pooling_node, 0);
               pooling_node.recalc_output_layouts();
            }
        }
    };

#ifdef ENABLE_ONEDNN_FOR_GPU
    const auto reorder_input_gemm = [&p, &rf](typed_program_node<gemm>& gemm_node) {
        if (gemm_node.get_preferred_impl_type() != impl_types::onednn || gemm_node.is_dynamic()
            || gemm_node.get_preferred_input_fmts().size() < 2) {
            return;
        }

        for (size_t idx = 0; idx < 2; ++idx) {
            auto fmt = gemm_node.get_preferred_input_fmts()[idx];
            if (fmt != format::type::any && !format::is_simple_data_format(fmt)) {
                return;
            }
        }

        for (size_t idx = 0; idx < 2; idx++) {
            auto dep = gemm_node.get_dependency_with_port(idx);
            const auto& input = dep.first;
            auto input_layout = input->get_output_layout();

            if (input_layout.is_dynamic())
                continue;

            if (!input->is_constant() && !format::is_simple_data_format(input_layout.format)) {
                auto new_layout = input_layout;
                new_layout.format = format::get_default_format(input_layout.get_rank());
                auto new_input = rf.get_reorder(input->id(), dep.second, input_layout, new_layout);
                if (new_input.first) {
                    p.add_intermediate(new_input.first, gemm_node, idx, !new_input.second);
                }
            }
        }
    };
#endif // ENABLE_ONEDNN_FOR_GPU

    for (auto& prim : p.get_processing_order()) {
        program_helpers::do_for_types<detection_output, deconvolution, convolution, fully_connected, pooling>(
            *prim,
            reorder_input_detection_output,
            reorder_input_and_weights_deconvolution,
            reorder_convolution,
            reorder_input_fully_connected,
            reorder_input_pooling);

#ifdef ENABLE_ONEDNN_FOR_GPU
        program_helpers::do_for_types<gemm>(
            *prim,
            reorder_input_gemm);
#endif // ENABLE_ONEDNN_FOR_GPU
    }

    for (auto n : p.get_processing_order()) {
        if (n->is_in_data_flow() && fmt_map.count(n) != 0) {
            n->get_output_layout(); // There might be some invalid output layout
            auto preferred_impl = lo.get_preferred_impl_type(*n, fmt_map.at(n));
            n->set_preferred_impl_type(preferred_impl);
        }
    }

    // WA for OneDNN PRelu activation fusions: convert activation's slope buffer to expected f32 data type
    for (auto& node : p.get_processing_order()) {
        if (node->get_preferred_impl_type() == impl_types::onednn) {
            auto fused_prims = node->get_fused_primitives();
            for (auto& fused_desc : fused_prims) {
                if (!fused_desc.is_type<activation>())
                    continue;

                auto activation_desc = fused_desc.typed_desc<activation>();
                if (activation_desc->activation_function == cldnn::activation_func::relu_negative_slope &&
                    !activation_desc->additional_params_input.empty()) {
                    const auto expected_dt = data_types::f32;
                    const auto dep_idx = fused_desc.outer_dep_start_idx;
                    const auto orig_layout = node->get_dependency(dep_idx).get_output_layout();
                    if (orig_layout.data_type == expected_dt)
                        continue;

                    auto new_layout = orig_layout;
                    new_layout.data_type = expected_dt;
                    auto new_input = rf.get_reorder(node->get_dependency(dep_idx).id(), orig_layout, new_layout);
                    if (new_input.first)
                        p.add_intermediate(new_input.first, *node, dep_idx, !new_input.second);
                }
            }
        }
    }

    // WA for OneDNN binary add fusions: we need to broadcast batch dimension to avoid situation with
    // batch dimension mismatch in OneDNN tensor descriptors as follow:
    // * Gemm output shape: (b,f,y,x) -> OneDNN shape: (b*f,y,x)
    // * Gemm fused op shape: (1,f,y,x) -> OneDNN shape: (1*f,y,x)
    // If batch dimension of gemm output is not equal to 1, then OneDNN will not be able to broadcast fused op data
    // correctly and we need to do it manually
#ifdef ENABLE_ONEDNN_FOR_GPU
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<gemm>() && node->get_preferred_impl_type() == impl_types::onednn) {
            for (const auto& fused_prim : node->get_fused_primitives()) {
                if (fused_prim.is_type<eltwise>() &&
                    one_of(fused_prim.typed_desc<eltwise>()->mode, {eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::prod})) {
                    auto& data = node->get_dependency(fused_prim.outer_dep_start_idx);

                    auto gemm_layout = node->get_output_layout();
                    auto data_layout = data.get_output_layout();

                    if (gemm_layout.is_dynamic() || data_layout.is_dynamic())
                        continue;

                    auto gemm_dims = onednn::convert_gemm_tensor(gemm_layout.get_tensor(),
                                                                 cldnn::format::dimension(gemm_layout.format),
                                                                 false);

                    auto data_dims = onednn::convert_gemm_tensor(data_layout.get_tensor(),
                                                                 cldnn::format::dimension(data_layout.format),
                                                                 false);

                    if (gemm_dims[0] == data_dims[0])
                        continue;

                    auto data_shape = data_layout.get_shape();
                    if (data_shape.size() && shape_size(data_shape) == 1ul)
                        continue;

                    static size_t idx = 0;
                    const auto prim_id = "broadcast:" + data.id() + "_broadcasted" + std::to_string(idx++);
                    auto broadcast_prim = std::make_shared<cldnn::broadcast>(prim_id, cldnn::input_info(data.id()), gemm_layout.get_shape(),
                                                                            ov::AxisSet{}, ov::op::BroadcastType::NUMPY);

                    auto& broadcast_node = p.get_or_create(broadcast_prim);
                    p.add_intermediate(broadcast_node, *node, fused_prim.outer_dep_start_idx, true);
                    broadcast_node.recalc_output_layouts(false);
                }
            }
        } else if (node->is_type<fully_connected>() && node->get_preferred_impl_type() == impl_types::onednn) {
            for (const auto& fused_prim : node->get_fused_primitives()) {
                if (fused_prim.is_type<eltwise>() &&
                    one_of(fused_prim.typed_desc<eltwise>()->mode, {eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::prod})) {
                    auto fc_layout = node->get_output_layout();
                    auto& data = node->get_dependency(fused_prim.outer_dep_start_idx);
                    auto data_layout = data.get_output_layout();

                    if (fc_layout.is_dynamic() || data_layout.is_dynamic())
                        continue;

                    // fc_b     | fc_f      | data_b    | data_f    | broadcast condition
                    // ---------+-----------+-----------+-----------+--------------------
                    // 1        | 1         | 1         | 1         | no broadcast
                    // 1        | 1         | 1         | N         | N/A
                    // 1        | 1         | N         | 1         | N/A
                    // 1        | 1         | N         | N         | N/A
                    // 1        | N         | 1         | 1         | implicit broadcast
                    // 1        | N         | 1         | N         | no broadcast
                    // 1        | N         | N         | 1         | N/A
                    // 1        | N         | N         | N         | N/A
                    // N        | 1         | 1         | 1         | implicit broadcast
                    // N        | 1         | 1         | N         | N/A
                    // N        | 1         | N         | 1         | no broadcast
                    // N        | 1         | N         | N         | N/A
                    // N        | N         | 1         | 1         | implicit broadcast
                    // N        | N         | 1         | N         | explicit broadcast
                    // N        | N         | N         | 1         | explicit broadcast
                    // N        | N         | N         | N         | no broadcast
                    if ((fc_layout.batch() == 1 || fc_layout.feature() == 1) ||
                        (data_layout.batch() == 1 && data_layout.feature() == 1) ||
                        (fc_layout.count() == data_layout.count())) {
                        continue;
                    }

                    static size_t idx = 0;
                    const auto prim_id = "broadcast:" + data.id() + "_broadcasted" + std::to_string(idx++);
                    auto broadcast_prim = std::make_shared<cldnn::broadcast>(prim_id, cldnn::input_info(data.id()), fc_layout.get_shape(),
                                                                            ov::AxisSet{}, ov::op::BroadcastType::NUMPY);

                    auto& broadcast_node = p.get_or_create(broadcast_prim);
                    p.add_intermediate(broadcast_node, *node, fused_prim.outer_dep_start_idx, true);
                    broadcast_node.recalc_output_layouts(false);
                }
            }
        }
    }
#endif
}
