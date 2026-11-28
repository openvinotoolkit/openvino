// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "assign_inst.h"
#include "condition_inst.h"
#include "convert_color_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/primitives/input_layout.hpp"
#include "mvn_inst.h"
#include "pass_manager.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;

namespace {
void eliminate_pad_for_onednn_impl(program& p, program_node& node) {
    // Padded offsets aren't supported by onednn kernels
    bool has_paddings = false;
    bool use_onednn = false;
    for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
        const auto& input = node.get_dependency(idx);
        if (!input.is_in_data_flow() || input.is_constant()) {
            continue;
        }
        if (input.get_output_layout().data_padding) {
            has_paddings = true;
            break;
        }
    }

    if (has_paddings) {
        // oneDNN doesn't support padded memory, so we check that onednn impl can be used with dropped paddings
        use_onednn = test_no_input_pad<bool>(node, [](const program_node& node) {
            return node.type()->has_impl_for(node, impl_types::onednn);
        });
    }

    if (use_onednn) {
        // In case of implicit onednn concat, padding should not be removed.
        if (node.can_be_optimized()) {
            return;
        }
        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            auto node_and_port = node.get_dependency_with_port(idx);
            auto& input = *node_and_port.first;
            auto port = node_and_port.second;
            if (!input.is_in_data_flow() || input.is_constant()) {
                continue;
            }

            const auto& in_layout = input.get_output_layout(false, port);
            const auto& in_padding = in_layout.data_padding;
            if (static_cast<bool>(in_padding)) {
                bool spatial_padding = false;
                for (size_t i = 0; i < in_layout.get_spatial_rank(); ++i) {
                    spatial_padding |= (in_padding._lower_size[2 + i] != 0);
                }
                for (size_t i = 0; i < in_layout.get_spatial_rank(); ++i) {
                    spatial_padding |= (in_padding._upper_size[2 + i] != 0);
                }

                bool feature_padding = false;
                feature_padding |= (in_padding._lower_size[1] != 0);
                feature_padding |= (in_padding._upper_size[1] != 0);

                bool batch_padding = false;
                batch_padding |= (in_padding._lower_size[0] != 0);
                batch_padding |= (in_padding._upper_size[0] != 0);

                if (batch_padding && !feature_padding && !spatial_padding) {
                    batch_padding = false;
                }

                if (spatial_padding || batch_padding) {
                    cldnn::layout layout_wo_padding = in_layout;
                    layout_wo_padding.data_padding = cldnn::padding{};
                    layout_wo_padding.data_padding._lower_size[1] = in_layout.data_padding._lower_size[1];
                    layout_wo_padding.data_padding._upper_size[1] = in_layout.data_padding._upper_size[1];
                    if (input.is_type<reorder>()) {
                        input.set_output_padding(padding());
                        input.set_output_layout(layout_wo_padding, false, port);
                    } else {
                        auto new_reorder = std::make_shared<reorder>(input.id() + "_padding_reorder_" + node.id(), input.id(), layout_wo_padding);
                        auto& new_reorder_node = p.get_or_create(new_reorder);
                        p.add_intermediate(new_reorder_node, node, idx);
                        new_reorder_node.recalc_output_layouts(false);
                    }
                } else {
                    return;
                }
            }
        }

        return;
    }
}

// condition passes its input buffers as they are to the parameters of the branch programs, and input_layout_inst::set_data()
// requires them to be in the layout of the parameter. Reorder each input to the format and padding of the branch parameter it
// is mapped to (the default format without padding for bodies built by the plugin), e.g. an output in a blocked format or
// padded for another user.
void reorder_condition_inputs(program& p, condition_node& node) {
    for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
        const auto [input, port] = node.get_dependency_with_port(idx);
        const auto& in_layout = input->get_output_layout(false, port);
        if (in_layout.is_dynamic()) {
            continue;
        }

        // Layout of the static branch parameters the input is mapped to: set_data() doesn't check dynamic ones, and the
        // predicate is not mapped.
        std::optional<layout> param_layout;
        for (const auto* branch : {&node.get_branch_true(), &node.get_branch_false()}) {
            const auto it = branch->input_map.find(input->id());
            if (it == branch->input_map.end() || !branch->inner_program->has_node(it->second)) {
                continue;
            }
            const auto& branch_layout = branch->inner_program->get_node(it->second).get_output_layout();
            if (branch_layout.is_dynamic()) {
                continue;
            }
            if (param_layout) {
                OPENVINO_ASSERT(param_layout->format == branch_layout.format && param_layout->data_padding == branch_layout.data_padding,
                                "[GPU] Input ",
                                input->id(),
                                " of condition ",
                                node.id(),
                                " is mapped to branch parameters with different layouts: ",
                                param_layout->to_short_string(),
                                " and ",
                                branch_layout.to_short_string());
            }
            param_layout = branch_layout;
        }
        if (!param_layout) {
            continue;
        }

        auto expected_layout = in_layout;
        expected_layout.format = param_layout->format;
        expected_layout.data_padding = param_layout->data_padding;
        if (in_layout == expected_layout) {
            continue;
        }

        auto new_reorder =
            std::make_shared<reorder>(input->id() + "_reorder_" + node.id() + "_p" + std::to_string(idx), input_info(input->id(), port), expected_layout);
        auto& new_reorder_node = p.get_or_create(new_reorder);
        p.add_intermediate(new_reorder_node, node, idx);
        new_reorder_node.recalc_output_layouts(false);
    }
}
}  // namespace

/*
This pass checks if data formats (layouts) of output/input in hidden layers match.
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr
*/
void add_required_reorders::add_reorder(program& p, program_node* node, program_node* usr, bool keep_original_dt) {
    layout reorder_layout = node->get_output_layout();
    reorder_layout.format = usr->get_output_layout().format;
    reorder_layout.data_type = usr->get_output_layout().data_type;

    if (keep_original_dt) {
        reorder_layout.data_type = node->get_output_layout().data_type;
    }

    // dep index in reorder id distinguishes multi-port edges; otherwise add_intermediate fails on the second pass.
    auto it = std::find_if(usr->get_dependencies().begin(), usr->get_dependencies().end(), [&](const std::pair<program_node*, int32_t>& dep) {
        return node == dep.first;
    });
    if (it == usr->get_dependencies().end()) {
        throw std::runtime_error("Inconcistency in topology description: user of a node is not present among its dependecies.");
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw std::runtime_error("Internal Error: container index out of range exception.");
    }

    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id() + "_p" + std::to_string(idx), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);
    new_reorder_node.set_output_layout(reorder_layout, false);

    p.add_intermediate(new_reorder_node, *usr, idx);
    new_reorder_node.recalc_output_layouts(false);
}

bool add_required_reorders::test_format(cldnn::program_node& node, format requested_format) {
    for (size_t i = 0; i < node.get_outputs_count(); i++) {
        auto out_layout = node.get_output_layout(false, i);
        out_layout.format = requested_format;
        node.set_output_layout(out_layout, false, i);
    }

    for (size_t i = 0; i < node.get_dependencies().size(); i++) {
        const auto& dep_with_port = node.get_dependency_with_port(i);
        const auto& dep = dep_with_port.first;

        auto current_format = dep->get_output_layout(false, dep_with_port.second).format;

        if (dep->is_constant() || format::is_weights_format(current_format)) {
            continue;
        }

        if (dep->is_type<reorder>()) {
            const auto& port = dep_with_port.second;
            auto new_layout = dep->get_output_layout(false, port);
            new_layout.format = requested_format;
            dep->set_output_layout(new_layout, false, port);
        } else if (current_format != requested_format) {
            add_reorder(node.get_program(), dep_with_port.first, &node, true);
        }
    }

    return node.type()->has_impl_for(node, impl_types::any, shape_types::any);
}

namespace {
const std::set<std::string>& bf16_fallback_excluded_types() {
    static const std::set<std::string> excluded_types = {
        "input_layout", "data", "reorder", "assign", "read_value", "loop", "condition", "kv_cache",
        "lstm", "lstm_dynamic_input", "lstm_dynamic_timeloop", "paged_attention", "scaled_dot_product_attention"
    };
    return excluded_types;
}

bool has_bf16_data_type(program_node& node) {
    const auto& input_layouts = node.get_input_layouts();
    return node.get_output_layout().data_type == data_types::bf16 ||
           std::any_of(input_layouts.begin(), input_layouts.end(), [](const layout& value) {
               return value.data_type == data_types::bf16;
           });
}
}  // namespace

bool add_required_reorders::needs_bf16_fallback(program_node& node) {
    if (bf16_fallback_excluded_types().count(node.get_primitive()->type_string()) || node.get_outputs_count() != 1 ||
        !node.is_in_data_flow() || node.is_in_shape_of_subgraph() || node.can_be_optimized() ||
        node.is_runtime_skippable() || !has_bf16_data_type(node) ||
        (node.get_preferred_impl_type() != impl_types::any && node.get_preferred_impl_type() != impl_types::ocl) ||
        (node.get_forced_impl_type() != impl_types::any && node.get_forced_impl_type() != impl_types::ocl) ||
        node.get_program().get_config().get_force_implementations().count(node.id())) {
        return false;
    }

    const auto params = node.get_kernel_impl_params();
    if (params->weights_layout || params->bias_layout || !params->state_layouts.empty() ||
        !params->inner_progs.empty() || !params->inner_nets.empty()) {
        return false;
    }
    bool found_ocl_manager = false;
    const auto shape_type = ImplementationManager::get_shape_type(*params);
    for (const auto& manager : node.type()->get_all_implementations()) {
        if ((manager->get_shape_type() & shape_type) != shape_type)
            continue;
        if (manager->get_impl_type() != impl_types::ocl) {
            if (manager->validate(node))
                return false;
            continue;
        }
        const auto native_support = manager->query_data_types(*params);
        if (native_support != implementation_data_type_support::unsupported)
            return false;
        found_ocl_manager = true;
    }
    return found_ocl_manager;
}

bool add_required_reorders::try_bf16_fallback(program& p, program_node& node) {
    if (!needs_bf16_fallback(node) || !node.get_fused_primitives().empty()) {
        return false;
    }

    auto native_params = node.get_kernel_impl_params();
    auto fp32_params = *native_params;
    bool needs_fallback = false;
    for (size_t idx = 0; idx < fp32_params.input_layouts.size(); ++idx) {
        if (fp32_params.input_layouts[idx].data_type == data_types::bf16) {
            if (node.get_dependency(idx).is_constant())
                return false;
            fp32_params.input_layouts[idx].data_type = data_types::f32;
            needs_fallback = true;
        }
    }
    auto original_output_layout = node.get_output_layout();
    if (fp32_params.output_layouts[0].data_type == data_types::bf16) {
        fp32_params.output_layouts[0].data_type = data_types::f32;
        needs_fallback = true;
    }
    if (!needs_fallback)
        return false;

    auto fp32_desc = node.get_primitive()->clone();
    fp32_desc->output_data_types[0] = fp32_params.output_layouts[0].data_type;
    fp32_params.desc = fp32_desc;
    auto proposed_node = node.type()->create_node(p, fp32_desc);
    proposed_node->set_output_layout(fp32_params.output_layouts[0], false);
    proposed_node->data_flow = node.data_flow;
    proposed_node->impl_type = node.impl_type;
    proposed_node->forced_impl_type = node.forced_impl_type;
    proposed_node->preferred_input_fmts = node.preferred_input_fmts;
    proposed_node->preferred_output_fmts = node.preferred_output_fmts;
    std::vector<std::shared_ptr<program_node>> input_proxies;
    for (size_t idx = 0; idx < node.get_dependencies().size(); ++idx) {
        auto dependency = node.get_dependency_with_port(idx);
        if (native_params->input_layouts[idx].data_type == data_types::bf16) {
            auto proxy_desc = std::make_shared<input_layout>(node.id() + "_bf16_probe_" + std::to_string(idx),
                                                            fp32_params.input_layouts[idx]);
            auto proxy = proxy_desc->type->create_node(p, proxy_desc);
            proxy->set_output_layout(fp32_params.input_layouts[idx], false);
            dependency = {proxy.get(), 0};
            input_proxies.push_back(std::move(proxy));
        }
        proposed_node->dependencies.push_back(dependency);
    }

    const auto shape_type = ImplementationManager::get_shape_type(*native_params);
    const auto selected_manager = node.type()->choose_impl(*proposed_node, shape_type);
    if (!selected_manager || selected_manager->get_impl_type() != impl_types::ocl ||
        (selected_manager->get_shape_type() & shape_type) != shape_type ||
        selected_manager->query_data_types(fp32_params) != implementation_data_type_support::supported ||
        !selected_manager->validate(*proposed_node) || !selected_manager->support_shapes(fp32_params)) {
        return false;
    }
    const auto implementation = selected_manager->create_impl(*proposed_node, fp32_params);
    if (!implementation)
        return false;

    auto unique_id = [&p](const primitive_id& base) {
        auto result = base;
        size_t suffix = 0;
        while (p.nodes_map.count(result))
            result = base + "_" + std::to_string(++suffix);
        return result;
    };
    const auto original_id = node.id();
    const bool restore_bf16 = original_output_layout.data_type == data_types::bf16;
    const bool was_output = node.is_output();
    const auto original_users = node.get_users();
    node.desc = fp32_desc;
    if (restore_bf16) {
        node.set_output(false);
        p.rename(node, unique_id(original_id + "_bf16_fallback_fp32"));
    }
    node.set_output_layout(fp32_params.output_layouts[0], false);
    for (size_t idx = 0; idx < native_params->input_layouts.size(); ++idx) {
        if (native_params->input_layouts[idx].data_type != data_types::bf16)
            continue;
        const auto dependency = node.get_dependency_with_port(idx);
        auto conversion = std::make_shared<reorder>(unique_id(node.id() + "_input_fp32_" + std::to_string(idx)),
                                                    input_info(dependency.first->id(), dependency.second),
                                                    fp32_params.input_layouts[idx]);
        auto& conversion_node = p.get_or_create(conversion);
        p.add_intermediate(conversion_node, node, idx);
        conversion_node.set_output_layout(fp32_params.input_layouts[idx], false);
    }
    if (restore_bf16) {
        auto conversion = std::make_shared<reorder>(original_id, input_info(node.id()), original_output_layout);
        auto& conversion_node = p.get_or_create(conversion);
        for (auto* user : original_users) {
            for (size_t idx = 0; idx < user->get_dependencies().size(); ++idx) {
                if (user->get_dependency_with_port(idx).first == &node)
                    user->replace_dependency(idx, {&conversion_node, 0}, false);
            }
        }
        p.add_connection(node, conversion_node, 0);
        p.processing_order.insert_next(&node, &conversion_node);
        conversion_node.set_output_layout(original_output_layout, false);
        conversion_node.set_output(was_output);
        for (auto& output : p.outputs) {
            if (output == &node)
                output = &conversion_node;
        }
    }
    return true;
}

void add_required_reorders::run(program& p) {
    bool optimize_data = p.get_config().get_optimize_data();
    auto usr_itr = p.get_processing_order().begin();
    while (usr_itr != p.get_processing_order().end()) {
        const auto& usr = *usr_itr++;
        if (usr->get_dependencies().empty()) {
            continue;  // only nodes with dependencies
        }
        if (usr->is_type<data>()) {
            continue;
        }

        if (!usr->is_all_valid_output_layouts()) {
            usr->recalc_output_layouts(false);
        }

        // If usr is assign and input and output data types are different
        // add reorder with usr's output data type between dep and usr
        if (usr->is_type<assign>()) {
            // The assigned value can be any output of a multi-output producer (e.g. a QSA indexer state).
            const auto [dep, port] = usr->get_dependency_with_port(0);
            auto dep_layout = dep->get_output_layout(true, port);
            auto out_layout = usr->get_output_layout();
            bool required_reorder = out_layout.data_type != dep_layout.data_type;
            if (required_reorder) {
                auto new_reorder =
                    std::make_shared<reorder>(dep->id() + "_reorder_" + usr->id(), input_info(dep->id(), port), out_layout.format, out_layout.data_type);
                auto& new_reorder_node = p.get_or_create(new_reorder);
                p.add_intermediate(new_reorder_node, *usr, 0);
                new_reorder_node.recalc_output_layouts(false);
            }
        }

        if (usr->is_type<eltwise>()) {
            for (size_t i = 0; i < usr->get_dependencies().size(); i++) {
                auto& eltwise_node = usr->as<eltwise>();
                auto& dep = usr->get_dependency(i);
                if (!dep.is_in_data_flow() || dep.is_constant()) {
                    continue;
                }
                auto dep_layout = dep.get_output_layout();
                auto out_layout = usr->get_output_layout();
                bool required_reorder = (format::dimension(out_layout.format) != format::dimension(dep_layout.format)) ||
                                        (usr->is_in_shape_of_subgraph() && (out_layout.data_type != dep_layout.data_type));
                required_reorder &= !eltwise_node.need_align_for_numpy_broadcast(dep_layout);
                if (required_reorder) {
                    auto new_reorder = std::make_shared<reorder>(dep.id() + "_reorder_" + usr->id(), dep.id(), out_layout.format, out_layout.data_type);
                    auto& new_reorder_node = p.get_or_create(new_reorder);
                    p.add_intermediate(new_reorder_node, *usr, dep);
                    new_reorder_node.recalc_output_layouts(false);
                }
            }
        }

        if (optimize_data) {
            auto fused_ops = usr->get_fused_primitives();
            auto out_layout = usr->get_output_layout();
            // If there is a fused reorder at the end, then we use input layout of reorder
            // as target one for fused ops, as code generator in many kernels is expecting that, not final output layout
            // However, the condition below may need some adjustment in the future, if codegen of some primitives behave differently
            if (!fused_ops.empty() && fused_ops.back().is_type<reorder>()) {
                out_layout = fused_ops.back().input_layout;
            }
            for (auto& fused_op : fused_ops) {
                // Some kernels use blocked aligned subgroup reads for a vector of elements from dependency tensor
                // In that case jitter checks that layout of input tensor from fused op is same as output layout or broadcast is possible
                // The code below is intended to insert additional reorder node for const eltwise dependency to ensure jitter can process such fusion
                if (!fused_op.is_type<eltwise>() && (!fused_op.is_type<activation>() || fused_op.total_num_deps != 2)) {
                    continue;
                }

                if (!fused_op.has_outer_dep()) {
                    continue;
                }
                auto dep_id = fused_op.outer_dep_start_idx;
                auto& dep = usr->get_dependency(dep_id);
                if (!dep.is_type<data>()) {
                    continue;
                }

                auto dep_layout = dep.get_output_layout();

                bool valid_broadcast_case = out_layout.is_static() && dep_layout.is_static() &&
                                            (static_cast<size_t>(out_layout.feature()) == dep_layout.count() || dep_layout.count() == 1);

                bool requires_reorder = out_layout.format != dep_layout.format && !valid_broadcast_case;
                if (requires_reorder) {
                    auto new_reorder = std::make_shared<reorder>(dep.id() + "_reorder_" + usr->id(), dep.id(), out_layout.format, dep_layout.data_type);
                    auto& new_reorder_node = p.get_or_create(new_reorder);
                    p.add_intermediate(new_reorder_node, *usr, dep);
                    new_reorder_node.recalc_output_layout(false);
                }
            }
        }

        if (usr->is_type<condition>()) {
            reorder_condition_inputs(p, usr->as<condition>());
        }

        // Remove padded-inputs in spatial axes not to use ref kernel which causes huge perf drop
        if (usr->is_type<mvn>()) {
            if (usr->as<mvn>().input().is_padded_spatial()) {
                auto out_layout = usr->get_output_layout();
                // Check formats of implemented opt kernels without a spatial padding support
                if (out_layout.format == format::b_fs_yx_fsv16 || out_layout.format == format::b_fs_zyx_fsv16 ||
                    out_layout.format == format::bs_fs_yx_bsv32_fsv16 || out_layout.format == format::bs_fs_yx_bsv32_fsv32) {
                    auto& dep = usr->as<mvn>().input();
                    cldnn::layout layout_wo_padding = dep.get_output_layout();
                    layout_wo_padding.data_padding = cldnn::padding{};
                    auto new_reorder = std::make_shared<reorder>(dep.id() + "_no_pad_reorder", dep.id(), layout_wo_padding);
                    auto& new_reorder_node = p.get_or_create(new_reorder);
                    p.add_intermediate(new_reorder_node, *usr, dep);
                    new_reorder_node.recalc_output_layout(false);
                }
            }
        }

        eliminate_pad_for_onednn_impl(p, *usr);

        if (try_bf16_fallback(p, *usr)) {
            continue;
        }

        if (usr->type()->has_impl_for(*usr)) {
            continue;
        }

        bool correct_layout_selected = false;
        bool weights_data = (usr->is_type<convolution>() || usr->is_type<deconvolution>() || usr->is_type<fully_connected>());

        layout original_layout = usr->get_output_layout();

        for (const auto& node : usr->get_dependencies()) {
            if (!node.first->is_in_data_flow() && !weights_data) {
                if (cldnn::format::dimension(original_layout.format) == cldnn::format::dimension(node.first->get_output_layout().format)) {
                    /*
                        ToDo: Here we should handle also the situation where primitive usr has data inputs in different
                       formats
                    */
                    layout current_layout(original_layout.get_partial_shape(), original_layout.data_type, node.first->get_output_layout().format);
                    usr->set_output_layout(current_layout, false);
                    if (usr->type()->has_impl_for(*usr)) {
                        correct_layout_selected = true;
                        break;
                    }
                }
            }
        }

        if (!correct_layout_selected) {
            std::vector<cldnn::format> preferred_layout_formats;
            size_t max_in_dims = std::max(cldnn::format::dimension(original_layout.format), static_cast<size_t>(4));
            for (const auto& node : usr->get_dependencies()) {
                if (format::is_weights_format(node.first->get_output_layout().format)) {
                    continue;
                }
                max_in_dims = std::max(cldnn::format::dimension(node.first->get_output_layout().format), max_in_dims);
            }
            // This list of preferred layouts has been selected arbitrarily due to developers' experience
            preferred_layout_formats = {cldnn::format::get_default_format(max_in_dims)};
            if (max_in_dims == 8) {
                preferred_layout_formats.push_back(cldnn::format::bfvuwzyx);
            } else if (max_in_dims == 7) {
                preferred_layout_formats.push_back(cldnn::format::bfuwzyx);
            } else if (max_in_dims == 6) {
                preferred_layout_formats.push_back(cldnn::format::bfwzyx);
            } else if (max_in_dims == 5) {
                preferred_layout_formats.push_back(cldnn::format::bzyxf);
            } else if (max_in_dims == 4) {
                preferred_layout_formats.push_back(cldnn::format::yxfb);
                preferred_layout_formats.push_back(cldnn::format::byxf);
            }

            if (original_layout.is_dynamic() && usr->type()->has_impl_for(*usr, shape_types::dynamic_shape)) {
                correct_layout_selected = true;
            }

            if (!correct_layout_selected) {
                for (auto new_layout_format : preferred_layout_formats) {
                    if (test_format(*usr, new_layout_format)) {
                        correct_layout_selected = true;
                        break;
                    }
                }
            }
        }

        OPENVINO_ASSERT(correct_layout_selected,
                        "[GPU] No layout format available for ",
                        usr->id(),
                        ", impl_type: ",
                        usr->get_preferred_impl_type(),
                        " (format: ",
                        original_layout.format.to_string(),
                        ", data_type: ",
                        ov::element::Type(original_layout.data_type),
                        ") ",
                        original_layout.to_string(),
                        ", ",
                        correct_layout_selected);
    }
}
