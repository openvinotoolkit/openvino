// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include "binary_convolution_inst.h"
#include "mvn_inst.h"

#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>

#define CLDNN_REORDER_INPUTS_VERBOSE 0

// Prints overall statistics of performed selection, such as number of reorders required.
#define CLDNN_REORDER_INPUTS_VERBOSE_STATISTICS          (CLDNN_REORDER_INPUTS_VERBOSE > 0)
// Prints special cases and work-arounds matched.
#define CLDNN_REORDER_INPUTS_VERBOSE_PATTERN_MATCH       (CLDNN_REORDER_INPUTS_VERBOSE > 1)
// Prints full list of preferred formats for each node.
#define CLDNN_REORDER_INPUTS_VERBOSE_PREFERRED           (CLDNN_REORDER_INPUTS_VERBOSE > 2)
// Prints full list of selected formats for each node.
#define CLDNN_REORDER_INPUTS_VERBOSE_FORMATS             (CLDNN_REORDER_INPUTS_VERBOSE > 2)

#if CLDNN_REORDER_INPUTS_VERBOSE
#include "to_string_utils.h"
#include <iostream>
#define CLDNN_REORDER_INPUTS_LOG(x) std::cout << "[clDNN][reorder_inputs] " << x << std::endl
#endif

#if CLDNN_REORDER_INPUTS_VERBOSE_PATTERN_MATCH
#define CLDNN_REORDER_INPUTS_PATTERN_MATCH_LOG(desc, id) CLDNN_REORDER_INPUTS_LOG(id << " matched for pattern: " << desc)
#else
#define CLDNN_REORDER_INPUTS_PATTERN_MATCH_LOG(desc, id) do { } while (false)
#endif

using namespace cldnn;

// ToDo remove friendship relation from program_impl

reorder_inputs::reorder_inputs(layout_optimizer& lo_ref, reorder_factory& rf_ref) : base_pass("reorder_inputs"), _lo(lo_ref), _rf(rf_ref) {}

void reorder_inputs::run(program_impl& p) { run(p, _lo, _rf); }

namespace {

std::map<program_node*, format::type> get_preferred_formats(program_impl& p, layout_optimizer& lo) {
    std::map<program_node*, format::type> fmt_map;
    for (auto n : p.get_processing_order()) {
        if (!n->is_in_data_flow())
            continue;

        auto ex = lo.get_preferred_format(*n);
        auto impl = lo.get_preferred_impl_type(*n);
        fmt_map[n] = ex;

        n->set_preferred_impl_type(impl);
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

void propagate_formats(program_impl& p, std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo) {
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
            size += travel_direction_wrapper<dir>::first(node, next)->get_output_layout().count();
        }
    }

    return { cnt, size };
}

reorder_cnt count_reorders(const std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo, program_node* node) {
    auto fwd = count_reorders_in_dir<direction_e::forwards>(fmt_map, lo, node);
    auto bwd = count_reorders_in_dir<direction_e::backwards>(fmt_map, lo, node);

    return { fwd.number + bwd.number, fwd.total_sizes + bwd.total_sizes };
}

void minimize_local_reorders(program_impl& p, std::map<program_node*, format::type>& fmt_map, layout_optimizer& lo) {
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
                (reorders_cnt.number == best_reorder_cnt.number && reorders_cnt.total_sizes < best_reorder_cnt.total_sizes) ) {
                best_reorder_cnt = reorders_cnt;
                best_format = new_fmt;
            }
        }

        fmt_map.at(node) = best_format;
    }
}

template <direction_e dir>
void insert_reorders_in_dir(program_impl& p, const std::map<program_node*, format::type>& fmt_map, reorder_factory& rf, program_node* node) {
    auto fmt = fmt_map.at(node);

    auto next_cpy = travel_direction_wrapper<dir>::next_nodes(node);
    for (auto next : next_cpy) {
        if (!next->is_in_data_flow())
            continue;

        if (fmt_map.count(next) > 0 && fmt_map.at(next) == fmt)
            continue;

        auto next_layout = next->get_output_layout();
        auto current_layout = node->get_output_layout();

        auto first_layout = travel_direction_wrapper<dir>::first(current_layout, next_layout);
        auto in_layout = first_layout;
        auto out_layout = first_layout;

        travel_direction_wrapper<dir>::first(in_layout, out_layout).format = fmt;

        auto reorder_pair = rf.get_reorder(travel_direction_wrapper<dir>::first(node, next)->id(),
                                           in_layout,
                                           out_layout);
        auto reorder = reorder_pair.first;

        if (reorder) {
            auto& reorder_node = p.get_or_create(reorder);
            p.add_intermediate(reorder_node,
                               *travel_direction_wrapper<dir>::second(node, next),
                               *travel_direction_wrapper<dir>::first(node, next),
                               !reorder_pair.second);
        }
    }
}

void insert_reorders(program_impl& p, const std::map<program_node*, format::type>& fmt_map, reorder_factory& rf) {
    auto fwd_it = p.get_processing_order().begin();
    while (fwd_it != p.get_processing_order().end()) {
        auto node = *(fwd_it++);

        if (fmt_map.count(node) != 1)
            continue;

        auto fmt = fmt_map.at(node);
        if (fmt == format::any || format::is_image(fmt))
            continue;

        insert_reorders_in_dir<direction_e::forwards>(p, fmt_map, rf, node);
    }

    auto bwd_it = p.get_processing_order().rbegin();
    while (bwd_it != p.get_processing_order().rend()) {
        auto node = *(bwd_it++);

        if (fmt_map.count(node) != 1)
            continue;

        auto fmt = fmt_map.at(node);
        if (fmt == format::any || format::is_image(fmt))
            continue;

        insert_reorders_in_dir<direction_e::backwards>(p, fmt_map, rf, node);
    }
}

}  // namespace

void reorder_inputs::run(program_impl& p, layout_optimizer& lo, reorder_factory& rf) {
    auto fmt_map = get_preferred_formats(p, lo);
#if CLDNN_REORDER_INPUTS_VERBOSE_PREFERRED
    {
        CLDNN_REORDER_INPUTS_LOG("Preferred formats:");
        for (auto& node_fmt : fmt_map) {
            if (node_fmt.second != format::any) {
                CLDNN_REORDER_INPUTS_LOG("  " << node_fmt.first->id() << " " << fmt_to_str(node_fmt.second));
            }
        }
    }
#endif
    propagate_formats(p, fmt_map, lo);
    minimize_local_reorders(p, fmt_map, lo);

#if CLDNN_REORDER_INPUTS_VERBOSE_FORMATS
    {
        CLDNN_REORDER_INPUTS_LOG("Selected formats:");
        for (auto node_ptr : p.get_processing_order()) {
            if (fmt_map.count(node_ptr) == 0)
                continue;

            auto fmt = fmt_map.at(node_ptr);
            CLDNN_REORDER_INPUTS_LOG("  " << node_ptr->id() << " " << fmt_to_str(fmt));
        }
    }
#endif
#if CLDNN_REORDER_INPUTS_VERBOSE_STATISTICS
    {
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
        CLDNN_REORDER_INPUTS_LOG("Total number of reorders: " << total_reorder_count.number / 2);
        CLDNN_REORDER_INPUTS_LOG("Total elements count of all reorders: " << total_reorder_count.total_sizes / 2);

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
        CLDNN_REORDER_INPUTS_LOG("Number of nodes with fused reorders: " << nodes_with_fusing);
    }
#endif

    insert_reorders(p, fmt_map, rf);

    for (auto n : p.get_processing_order()) {
        n->recalc_output_layout(true);
    }

    const auto reorder_input_detection_output = [&p, &rf](typed_program_node<detection_output>& detection_output_node) {
        auto detection_output_prim = detection_output_node.get_primitive();

        for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++) {
            auto& input = detection_output_node.get_dependency(i);
            auto new_input = rf.get_reorder(input.id(),
                                            input.get_output_layout(),
                                            layout{ data_types::f32, format::bfyx, input.get_output_layout().size });

            if (new_input.first) {
                p.add_intermediate(new_input.first, detection_output_node, i, !new_input.second);
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

    const auto reorder_input_deconvolution = [&p, &lo, &rf](typed_program_node<deconvolution>& deconv_node) {
        auto& input = deconv_node.input();
        auto input_layout = input.get_output_layout();
        auto new_format = lo.get_preferred_format(deconv_node);
        if (new_format == format::b_fs_zyx_fsv16 || new_format == format::bs_fs_zyx_bsv16_fsv16) {
            auto reorder = rf.get_reorder(input.id(), input_layout,
                layout{ input_layout.data_type, new_format, input_layout.size });
            if (reorder.first) {
                p.add_intermediate(reorder.first, deconv_node, 0, !reorder.second);
            }
        }
    };

    for (auto& prim : p.get_processing_order()) {
        program_helpers::do_for_types<detection_output, binary_convolution, deconvolution>(
            *prim,
            reorder_input_detection_output,
            reorder_input_binary_convolution,
            reorder_input_deconvolution);
    }
}
