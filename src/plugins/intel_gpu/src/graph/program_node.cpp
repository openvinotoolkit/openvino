// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"
#include "loop_inst.h"
#include "intel_gpu/runtime/debug_configuration.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "convolution_inst.h"
#include "gemm_inst.h"
#include "fully_connected_inst.h"
#include "deconvolution_inst.h"
#include "quantize_inst.h"
#include "reorder_inst.h"
#include "pooling_inst.h"
#include "reduce_inst.h"
#include <impls/onednn/utils.hpp>
#endif // ENABLE_ONEDNN_FOR_GPU

#include "to_string_utils.h"
#include "json_object.h"
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <set>

using namespace cldnn;

thread_local size_t program_node::cur_id = 0;

program_node::program_node(std::shared_ptr<primitive> prim, program& prog)
    : desc(prim), myprog(prog), preferred_input_fmts({}), preferred_output_fmts({}), org_id(prim ? (prim->id) : 0) {
    if (prim) {
        num_outputs = prim->num_outputs;
        for (size_t i = 0 ; i < num_outputs; ++i) {
            layout output_layout = layout{ov::PartialShape{}, data_types::f32, format::bfyx};
            output_layout.data_padding = prim->output_paddings[i];
            output_layouts.push_back(output_layout);
            valid_output_layouts.push_back(false);
        }
    }
}

void program_node::replace_dependency(size_t idx, program_node& new_dep, bool remove_if_dangling) {
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx].first == &new_dep)
        return;

    if (is_type<loop>()) {
        loop_node& loop = *this;
        loop.update_primitive_map(dependencies[idx].first->id(), new_dep.id(), true);
    }

    auto it = std::find(dependencies[idx].first->users.begin(), dependencies[idx].first->users.end(), this);
    if (it != dependencies[idx].first->users.end()) {
        dependencies[idx].first->users.erase(it);
    }

    if (remove_if_dangling)
        myprog.remove_if_dangling(*dependencies[idx].first);

    dependencies[idx].first = &new_dep;
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep, bool remove_if_dangling) {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i].first == &old_dep)
            return replace_dependency(i, new_dep, remove_if_dangling);
}

std::vector<primitive_id> program_node::get_dependencies_ids() const {
    std::vector<primitive_id> dep_ids;
    for (auto& dependency : dependencies) dep_ids.push_back(dependency.first->get_primitive()->id);
    return dep_ids;
}

void program_node::remove_dependency(size_t idx) {
    if (idx >= dependencies.size())
        return;

    dependencies[idx].first->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx].first);
    dependencies.erase(dependencies.begin() + idx);
}

std::set<primitive_id> program_node::get_memory_dependencies() const { return memory_dependencies; }

void program_node::add_memory_dependency(primitive_id prim) { memory_dependencies.insert(prim); }

void program_node::add_memory_dependency(std::vector<primitive_id> prim_list) {
    memory_dependencies.insert(prim_list.begin(), prim_list.end());
}

std::unique_ptr<json_composite> program_node::desc_to_json() const {
    std::unique_ptr<json_composite> node_info = std::unique_ptr<json_composite>(new json_composite());
    node_info->add("ptr", "node_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    node_info->add("id", id());
    node_info->add("type", desc->type_string());
    node_info->add("valid output layout", bool_to_str(valid_output_layouts[0]));
    std::stringstream s;
    s << get_preferred_impl_type();
    node_info->add("preferred impl", s.str());

    node_info->add("output layout", output_layouts[0].to_short_string());

    node_info->add("constant", bool_to_str(constant));
    node_info->add("in data flow", bool_to_str(data_flow));
    node_info->add("output", bool_to_str(output));
    node_info->add("optimized", bool_to_str(optimized));

    json_composite fused_nodes_info;
    size_t index = 0;
    for (auto& fused_desc : get_fused_primitives()) {
        json_composite fused_node_info;
        fused_node_info.add("id", fused_desc.desc->id);
        std::vector<primitive_id> dep_ids;
        for (auto dep : fused_desc.deps) {
            dep_ids.push_back(dep.first);
        }
        fused_node_info.add("dependencies", dep_ids);
        fused_node_info.add("dep start_idx", fused_desc.dep_start_idx);
        json_composite info;
        info.add("data type", dt_to_str(fused_desc.output_layout.data_type));
        info.add("format", fmt_to_str(output_layouts[0].format));
        info.add("size", output_layouts[0].to_short_string());
        fused_node_info.add("output layout", info);
        fused_nodes_info.add("fused primitive idx " + std::to_string(index++), fused_node_info);
    }
    node_info->add("fused primitives", fused_nodes_info);

#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& onednn_post_ops = get_fused_primitives_onednn();
    if (onednn_post_ops.size()) {
        size_t post_op_index = 0;
        json_composite post_ops_info;
        for (auto& fused_prim_desc : onednn_post_ops) {
            json_composite post_op_info;
            post_op_info.add("post op", onednn_post_op_type_to_str(fused_prim_desc.op_type));
            post_op_info.add("memory dependency", fused_prim_desc.mem_dep);
            post_op_info.add("memory offset", fused_prim_desc.mem_offset);
            post_ops_info.add("post ops idx " + std::to_string(post_op_index++), post_op_info);
        }
        node_info->add("onednn post ops", post_ops_info);
    }
#endif

    std::vector<std::string> deps_ptrs;
    {
        bool empty = true;
        auto itr = dependencies.begin();
        while (itr != dependencies.end()) {
            if (empty) {
                empty = false;
            }
            deps_ptrs.push_back(std::to_string(reinterpret_cast<uintptr_t>(itr->first)));
            itr++;
        }
        if (deps_ptrs.empty()) {
            deps_ptrs.push_back("null");
        }
    }
    node_info->add("dependencies", deps_ptrs);

    std::vector<std::string> users_ptrs;
    {
        bool empty = true;
        auto itr = users.begin();
        while (itr != users.end()) {
            if (empty) {
                empty = false;
            }
            users_ptrs.push_back(std::to_string(reinterpret_cast<uintptr_t>(*itr++)));
        }
        if (users_ptrs.empty()) {
            users_ptrs.push_back("null");
        }
    }
    node_info->add("users", users_ptrs);
    std::vector<std::string> impls;
    if (!selected_impl) {
        impls.push_back("null");
    } else {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
#endif
        impls.push_back(selected_impl->get_kernel_name());
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    }
    node_info->add("implementation", impls);
    return node_info;
}

void program_node::remove_dependency(program_node& node) {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i].first == &node)
            remove_dependency(i);
}

size_t program_node::get_user_index(program_node& node) const {
    size_t idx = 0;
    for (auto& user : users) {
        if (user == &node)
            return idx;
        else
            idx++;
    }

    OPENVINO_ASSERT(false, "Search invalid user node" + node.id() + " node");
}

size_t program_node::get_dependency_index(program_node& node) const {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i].first == &node)
            return i;

    OPENVINO_ASSERT(false, "Search invalid dependency node" + node.id() + " node");
}

bool program_node::is_detached(bool whole_branch) {
    if (!users.empty())
        return false;
    if (!whole_branch && !dependencies.empty())
        return false;
    return true;
}

layout program_node::calc_output_layout() const {
    bool allow_new_shape_infer = get_program().get_config().get_property(ov::intel_gpu::allow_new_shape_infer);
    if (allow_new_shape_infer) {
        auto out_layouts = type()->calc_output_layouts(*this, *get_kernel_impl_params());
        if (!out_layouts.empty()) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": calc_output_layout(new):" << out_layouts[0] << std::endl;
            return out_layouts[0];
        }
    }

    auto res = type()->calc_output_layout(*this, *get_kernel_impl_params());
    GPU_DEBUG_TRACE_DETAIL << id() << ": calc_output_layout:" << res << std::endl;

    return res;
}

std::vector<layout> program_node::calc_output_layouts() const {
    bool allow_new_shape_infer = get_program().get_config().get_property(ov::intel_gpu::allow_new_shape_infer);
    if (allow_new_shape_infer) {
        auto out_layouts = type()->calc_output_layouts(*this, *get_kernel_impl_params());
        if (!out_layouts.empty())
            return out_layouts;
    }

    return {type()->calc_output_layout(*this, *get_kernel_impl_params())};
}

layout program_node::get_output_layout(bool invalidate_users_if_changed, size_t idx) {
    if (valid_output_layouts[idx])
        return output_layouts[idx];

    auto new_layout = calc_output_layout();
    set_output_layout(new_layout, invalidate_users_if_changed, idx);
    return output_layouts[idx];
}

layout program_node::get_output_layout(size_t idx) const {
    if (!valid_output_layouts[idx])
        throw std::runtime_error("Output layout not calculated for " + id() + " node");

    return output_layouts[idx];
}

std::vector<layout> program_node::get_output_layouts(bool invalidate_users_if_changed) {
    if (is_all_valid_output_layouts())
        return output_layouts;

    auto new_layouts = calc_output_layouts();
    set_output_layouts(new_layouts, invalidate_users_if_changed);
    return output_layouts;
}

std::vector<layout> program_node::get_output_layouts() const {
    if (!is_all_valid_output_layouts()) {
        throw std::runtime_error("Output layouts not calculated for " + id() + " node");
    }

    return output_layouts;
}

layout program_node::get_non_padded_output_layout(bool invalidate_users_if_changed, size_t idx) {
    auto out_layout = get_output_layout(invalidate_users_if_changed, idx);
    auto result = layout({out_layout.data_type, out_layout.format, out_layout.get_tensor()});
    return result;
}

bool program_node::set_output_layout(layout& new_layout, bool invalidate_users_if_changed, size_t idx) {
    merge_output_padding(new_layout.data_padding, idx);
    new_layout.data_padding = output_layouts[idx].data_padding;
    bool changed = (new_layout != output_layouts[idx]);
    if (changed && invalidate_users_if_changed)  // output_layout has changed! invalidate users
        invalidate_users();

    output_layouts[idx] = new_layout;
    valid_output_layouts[idx] = true;
    return changed;
}

bool program_node::set_output_layouts(std::vector<layout>& new_layouts, bool invalidate_users_if_changed) {
    bool changed = false;
    for (size_t i = 0; i < new_layouts.size(); ++i) {
        auto new_layout = new_layouts[i];
        changed |= set_output_layout(new_layout, invalidate_users_if_changed, i);
    }
    for (auto v : valid_output_layouts) {
        v = true;
    }
    return changed;
}

bool program_node::recalc_output_layout(bool invalidate_users_if_changed) {
    auto new_layout = calc_output_layout();
    return set_output_layout(new_layout, invalidate_users_if_changed);
}

bool program_node::recalc_output_layouts(bool invalidate_users_if_changed) {
    auto new_layouts = calc_output_layouts();
    return set_output_layouts(new_layouts, invalidate_users_if_changed);
}

bool program_node::is_dynamic() const {
    for (const auto& input : get_dependencies()) {
        if (input.first->is_dynamic_output_layout())
            return true;
    }

    for (size_t i = 0; i < output_layouts.size(); ++i) {
        if (output_layouts[i].is_dynamic())
            return true;
    }
    return false;
}

bool program_node::is_dynamic() {
    for (auto& input : get_dependencies()) {
        if (input.first->is_dynamic_output_layout())
            return true;
    }

    for (size_t i = 0; i < output_layouts.size(); ++i) {
        if (output_layouts[i].is_dynamic())
            return true;
    }
    return false;
}

bool program_node::is_dynamic_output_layout(size_t idx) const {
    return (output_layouts[idx].is_dynamic()) ||  (output_layouts[idx].get_partial_shape().size() == 0);
}

bool program_node::is_dynamic_output_layout(size_t idx) {
    return (output_layouts[idx].is_dynamic()) ||  (output_layouts[idx].get_partial_shape().size() == 0);
}

bool program_node::has_padded_dependency() {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const std::pair<program_node*, int32_t>& dep) {
        return dep.first->is_padded();
    });
}

bool program_node::has_padded_dependency() const {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const std::pair<program_node*, int32_t>& dep) {
        return dep.first->is_padded();
    });
}

std::map<size_t, memory::ptr> program_node::get_const_memory_deps() const {
    std::map<size_t, memory::ptr> mem_deps;
    for (auto& i : get_shape_infer_dependencies()) {
        // Some primitives may have flexible count of deps (e.g. reshape), thus allow skipping some deps
        if (i >= get_dependencies().size())
            continue;

        auto& dep = get_dependency(i);
        if (dep.is_type<data>()) {
            mem_deps.insert({i, dep.as<data>().get_attached_memory_ptr()});
        }
    }
    return mem_deps;
}

void program_node::invalidate_users() const {
    for (auto& user : users) {
        for (size_t i = 0; i < user->valid_output_layouts.size(); ++i) {
            if (user->valid_output_layouts[i]) {
                if (user->get_preferred_output_fmt() != format::any)
                    continue;
                user->valid_output_layouts[i] = false;
                user->invalidate_users();
            }
        }
    }
}

void program_node::support_padding_all(bool support) {
    std::fill(_support_padding_in_axis.begin(), _support_padding_in_axis.end(), support);
}

bool program_node::is_padding_supported(int axis, int padding) const {
    if (!support_padding(axis))
        return false;

    auto fmt = output_layouts[0].format;

    // WA for known cases of padding not supported in implementations
    if (fmt == format::b_fs_yx_fsv16) {
        if (axis == 0 || (axis == 1 && padding % 16 != 0))
            return false;
    }

    if (fmt == format::fs_b_yx_fsv32 && (axis == 0))
        return false;

    auto block_sizes_dims = format::per_axis_block_size(fmt);
    for (const auto& block : block_sizes_dims) {
        size_t block_axis = block.first;
        int block_size = block.second;

        if (axis != static_cast<int>(block_axis))
            continue;

        if (padding % block_size != 0)
            return false;
    }

    return true;
}

 void program_node::set_selected_impl(std::unique_ptr<primitive_impl> impl) {
    selected_impl = std::move(impl);
}

bool program_node::need_lockable_memory() const {
    bool need_lockable_mem = get_users().empty() || std::any_of(get_users().begin(), get_users().end(), [](const program_node* n) {
        auto impl = n->get_selected_impl();
        return impl ? impl->is_cpu() : n->get_preferred_impl_type() == impl_types::cpu;
    });

    return need_lockable_mem;
}

void program_node::init_preferred_fmt(size_t dep_node, size_t user_node) {
    preferred_input_fmts.resize(dep_node, format::any);
    preferred_output_fmts.resize(user_node, format::any);
}

void program_node::set_preferred_input_fmt(size_t idx, format::type type) {
    if (idx >= preferred_input_fmts.size())
        preferred_input_fmts.resize(idx+1, format::any);

    preferred_input_fmts.at(idx) = type;
}

void program_node::set_preferred_output_fmt(size_t idx, format::type type) {
    if (idx >= preferred_output_fmts.size())
        preferred_output_fmts.resize(idx+1, format::any);

    preferred_output_fmts.at(idx) = type;
}

    /* ----------------------------------------- */
    /* Onednn fused operations integration logic */
    /* ----------------------------------------- */

#ifdef ENABLE_ONEDNN_FOR_GPU

dnnl::post_ops program_node::try_optimize_post_ops(dnnl::post_ops& p_ops, const std::shared_ptr<dnnl::primitive_attr>& attr,
                                                   bool& optimization_is_completed) {
    // Create new dnnl::post_ops object which will be filled inside the optimization process
    dnnl::post_ops optimized_p_ops;

    // Add new post-op into optimized_p_ops structure
    auto add_post_op = [&](onednn_post_op_type type, const dnnl::post_ops& cur_p_ops, dnnl::post_ops& new_p_ops, int idx) {
        switch (type) {
            case onednn_post_op_type::eltwise_act:
            case onednn_post_op_type::eltwise_clip:
            case onednn_post_op_type::eltwise_linear:
            case onednn_post_op_type::eltwise_round:
            case onednn_post_op_type::eltwise_hardsigmoid:
            {
                dnnl::algorithm alg;
                float alpha, beta;
                cur_p_ops.get_params_eltwise(idx, alg, alpha, beta);
                new_p_ops.append_eltwise(alg, alpha, beta);
                break;
            }

            case onednn_post_op_type::binary_add:
            case onednn_post_op_type::binary_sub:
            case onednn_post_op_type::binary_mul:
            case onednn_post_op_type::binary_max:
            case onednn_post_op_type::binary_min:
            {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                cur_p_ops.get_params_binary(idx, alg, desc);
                new_p_ops.append_binary(alg, desc);
                break;
            }

            case onednn_post_op_type::binary_relu:
            {
                int mask;
                cur_p_ops.get_params_prelu(idx, mask);
                new_p_ops.append_prelu(mask);
                break;
            }

            case onednn_post_op_type::scale:
            {
                break;
            }

            case onednn_post_op_type::sum:
            {
                float scale;
                dnnl::memory::data_type data_type;
                cur_p_ops.get_params_sum(idx, scale, data_type);
                // Only conv supports data type specification in append_sum. Other primitives(deconv, fc) do not support it.
                if (is_type<convolution>()) {
                    new_p_ops.append_sum(scale, 0/*zero_point*/, data_type);
                } else {
                    new_p_ops.append_sum(scale);
                }
                break;
            }

            case onednn_post_op_type::optimized:
            case onednn_post_op_type::optimized_sum:
            case onednn_post_op_type::optimized_eltwise_act:
            case onednn_post_op_type::optimized_eltwise_linear:
            case onednn_post_op_type::optimized_eltwise_clip:
            case onednn_post_op_type::optimized_eltwise_round:
            {
                // Current operation already has been optimized => don't need extra actions
                break;
            }

            default:
                throw std::runtime_error("Unsupported onednn post-operation type");
        }
    };

    // Check that post-op type is any optimized
    auto type_is_any_optimized = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::optimized ||
               type == onednn_post_op_type::optimized_sum ||
               type == onednn_post_op_type::optimized_eltwise_act ||
               type == onednn_post_op_type::optimized_eltwise_linear ||
               type == onednn_post_op_type::optimized_eltwise_clip ||
               type == onednn_post_op_type::optimized_eltwise_round;
    };

    // Check that post-op type is eltwise
    auto type_is_eltwise = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::eltwise_round || type == onednn_post_op_type::eltwise_linear ||
               type == onednn_post_op_type::eltwise_clip  || type == onednn_post_op_type::eltwise_act;
    };

    // Check that post-op type is binary_add or binary_mul
    auto type_is_binary_add_or_mul = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::binary_add || type == onednn_post_op_type::binary_mul;
    };

    // Simple post-op type checks
    auto type_is_optimized         = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::optimized; };
    auto type_is_eltwise_linear    = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::eltwise_linear; };
    auto type_is_optimized_eltwise = [](onednn_post_op_type type) -> bool {
         return type == onednn_post_op_type::optimized_eltwise_act || type == onednn_post_op_type::optimized_eltwise_linear ||
                type == onednn_post_op_type::optimized_eltwise_round || type == onednn_post_op_type::optimized_eltwise_clip;
    };
    auto type_is_binary_add        = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::binary_add; };
    auto type_is_binary_mul        = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::binary_mul; };
    auto type_is_optimized_sum     = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::optimized_sum; };
    auto type_is_scale             = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::scale; };

    auto get_eltwise_type = [](onednn_post_op_type type) {
        switch (type) {
            case onednn_post_op_type::optimized_eltwise_act: return onednn_post_op_type::eltwise_act;
            case onednn_post_op_type::optimized_eltwise_clip: return onednn_post_op_type::eltwise_clip;
            case onednn_post_op_type::optimized_eltwise_linear: return onednn_post_op_type::eltwise_linear;
            case onednn_post_op_type::optimized_eltwise_round: return onednn_post_op_type::eltwise_round;
            default:
                throw std::runtime_error("Unsupported optimized eltwise post-operation type");
                break;
        }
    };

    auto remove_optimized_prefix = [&](std::vector<fused_primitive_desc_onednn>& post_ops) {
        // Check and update post-op map if we already optimized something
        auto iter = post_ops.begin();
        while (iter != post_ops.end()) {
            if (type_is_optimized_sum(iter->op_type)) {
                iter->op_type = onednn_post_op_type::sum;
                ++iter;
            } else if (type_is_optimized_eltwise(iter->op_type)) {
                iter->op_type = get_eltwise_type(iter->op_type);
                ++iter;
            } else if (type_is_optimized(iter->op_type)) {
                iter = post_ops.erase(iter);
            } else {
                ++iter;
            }
        }
    };

    auto& cur_post_ops = get_fused_primitives_onednn();

    int64_t cur_post_op_idx = 1;
    int64_t prev_post_op_idx = 0;
    bool optimization_done = false;

    GPU_DEBUG_TRACE << "================================================" << std::endl;
    GPU_DEBUG_TRACE << " " << id() << ", num of post_ops " << p_ops.len() << std::endl;
    for (size_t i = 0; i < cur_post_ops.size(); i++)
        GPU_DEBUG_TRACE << "    " << i << ": " << cur_post_ops[i].op_type << std::endl;

    remove_optimized_prefix(cur_post_ops);

    GPU_DEBUG_TRACE << "remove optimized prefix ------------------------" << std::endl;
    GPU_DEBUG_TRACE << " " << id() << ", num of post_ops " << p_ops.len() << std::endl;
    for (size_t i = 0; i < cur_post_ops.size(); i++)
        GPU_DEBUG_TRACE << "    " << i << ": " << cur_post_ops[i].op_type << std::endl;
    GPU_DEBUG_TRACE << "----------------------------------->>>>>>>>>>>>>" << std::endl;

    // Get post-ops size for current node
    int64_t post_ops_size = cur_post_ops.size();

    auto get_optimized_eltwise_type = [](onednn_post_op_type type) {
        switch (type) {
            case onednn_post_op_type::eltwise_linear: return onednn_post_op_type::optimized_eltwise_linear;
            case onednn_post_op_type::eltwise_act: return onednn_post_op_type::optimized_eltwise_act;
            case onednn_post_op_type::eltwise_round: return onednn_post_op_type::optimized_eltwise_round;
            case onednn_post_op_type::eltwise_clip: return onednn_post_op_type::optimized_eltwise_clip;
            default:
                throw std::runtime_error("Unsupported optimized eltwise post-operation type");
                break;
        }
    };

    // Try to combine pairs of arithmetic post-ops (adds and muls) into one operation inside this cycle
    while (!optimization_done) {
        auto cur_type = cur_post_ops[cur_post_op_idx].op_type;
        auto prev_type = cur_post_ops[prev_post_op_idx].op_type;

        GPU_DEBUG_TRACE << "before prev_post_op_idx: " << prev_post_op_idx << ", cur_post_op_idx: " << cur_post_op_idx << std::endl;

        // Ignore optimized operations for "previous" operation in our operation pair
        while (type_is_any_optimized(prev_type) && prev_post_op_idx < post_ops_size - 1) {
            prev_post_op_idx++;
            if (prev_post_op_idx == cur_post_op_idx && cur_post_op_idx < post_ops_size - 1)
                cur_post_op_idx++;
            prev_type = cur_post_ops[prev_post_op_idx].op_type;
            cur_type = cur_post_ops[cur_post_op_idx].op_type;
        }

        // Ignore optimized operations for "current" operation in our operation pair
        while (type_is_any_optimized(cur_type) && cur_post_op_idx < post_ops_size - 1) {
            cur_post_op_idx++;
            cur_type = cur_post_ops[cur_post_op_idx].op_type;
        }

        GPU_DEBUG_TRACE << "after prev_post_op_idx: " << prev_post_op_idx << ", cur_post_op_idx: " << cur_post_op_idx << std::endl;

        // If prev_post_op_idx and cur_idx are same, add the last post-op to dnnl::post_ops
        if (prev_post_op_idx == post_ops_size - 1 && prev_post_op_idx == cur_post_op_idx && !type_is_any_optimized(prev_type)) {
            add_post_op(prev_type, p_ops, optimized_p_ops, prev_post_op_idx);
            break;
        }

        // If this is the last pair and it's optimized - add the last post-op and go out from the cycle
        if (cur_post_op_idx == post_ops_size - 1 && (type_is_any_optimized(cur_type) || type_is_any_optimized(prev_type))) {
            if (!type_is_any_optimized(prev_type)) {
                add_post_op(prev_type, p_ops, optimized_p_ops, prev_post_op_idx);
            }
            if (!type_is_any_optimized(cur_type)) {
                add_post_op(cur_type, p_ops, optimized_p_ops, cur_post_op_idx);
            }
            break;
        }

        // Post-ops combinations which can be simplified
        auto eltw_and_eltw  = type_is_eltwise(cur_type) && type_is_eltwise(prev_type);
        auto bin_and_eltw   = type_is_binary_add_or_mul(cur_type) && type_is_eltwise_linear(prev_type);
        auto eltw_and_bin   = type_is_eltwise_linear(cur_type) && type_is_binary_add_or_mul(prev_type);
        auto eltw_and_scale = type_is_eltwise_linear(cur_type) && type_is_scale(prev_type);

        auto can_try_optimize = eltw_and_eltw ||
                                bin_and_eltw ||
                                eltw_and_bin ||
                                eltw_and_scale;

        bool cur_ops_pair_is_optimized = false;

        if (can_try_optimize) {
            if (eltw_and_eltw) {
                dnnl::algorithm cur_alg, prev_alg;
                float cur_alpha, prev_alpha, cur_beta, prev_beta;

                p_ops.get_params_eltwise(prev_post_op_idx, prev_alg, prev_alpha, prev_beta);
                p_ops.get_params_eltwise(cur_post_op_idx, cur_alg, cur_alpha, cur_beta);

                auto eltw_linear_and_eltw_linear = type_is_eltwise_linear(cur_type) && type_is_eltwise_linear(prev_type);

                // eltwise_linear + eltwise_linear combination can be optimized always
                if (eltw_linear_and_eltw_linear) {
                    dnnl::post_ops eltw_p_op;
                    float optimized_alpha = cur_alpha * prev_alpha;
                    float optimized_beta = cur_alpha * prev_beta + cur_beta;
                    eltw_p_op.append_eltwise(cur_alg, optimized_alpha, optimized_beta);

                    // Combine 2 eltwises into one
                    add_post_op(cur_type, eltw_p_op, optimized_p_ops, 0);

                    // Marked current and previous eltwise operations as 'optimized' (they will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;
                    cur_post_ops[prev_post_op_idx].op_type = get_optimized_eltwise_type(prev_type);

                    // Set the flag if extra optimizations checking is needed
                    if (cur_post_op_idx < post_ops_size - 1) {
                        if (type_is_eltwise_linear(cur_post_ops[cur_post_op_idx + 1].op_type) ||
                            type_is_binary_add_or_mul(cur_post_ops[cur_post_op_idx + 1].op_type) ||
                            type_is_optimized_eltwise(cur_post_ops[cur_post_op_idx + 1].op_type)) {
                            optimization_is_completed = true;
                        }
                    }

                    cur_ops_pair_is_optimized = true;
                }
            } else if (bin_and_eltw) {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                float alpha, beta;

                cldnn::program_node& cur_node = get_dependency(cur_post_ops[cur_post_op_idx].mem_dep);

                p_ops.get_params_binary(cur_post_op_idx, alg, desc);
                p_ops.get_params_eltwise(prev_post_op_idx, alg, alpha, beta);

                // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                auto bin_ops_can_be_optimized = cur_node.is_type<data>() && cur_node.is_constant() &&
                                                cur_node.get_users().size() == 1 && desc.get_data_type() == dnnl_f32;

                auto bin_add_and_eltw = alpha == 1.0f && type_is_binary_add(cur_type) && bin_ops_can_be_optimized;
                auto bin_mul_and_eltw = beta == 0.f && type_is_binary_mul(cur_type) && bin_ops_can_be_optimized;

                if (bin_add_and_eltw || bin_mul_and_eltw) {
                    memory::ptr cur_bin_mem_ptr = cur_node.as<data>().get_attached_memory_ptr();
                    if (cur_bin_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for bin + eltw");
                    auto& stream = cur_bin_mem_ptr->get_engine()->get_service_stream();
                    mem_lock<float, mem_lock_type::read_write> bin_and_eltw_lock(cur_bin_mem_ptr, stream);

                    size_t cur_bin_mem_size = cur_node.get_output_layout().count();

                    // Update all binary coefficients
                    if (bin_add_and_eltw) {
                        for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                            bin_and_eltw_lock[data_idx] += beta;
                        }
                    } else {
                        for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                            bin_and_eltw_lock[data_idx] *= alpha;
                        }
                    }

                    // Marked previous eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[prev_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            } else if (eltw_and_bin) {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                float alpha, beta;

                cldnn::program_node& prev_node = get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                p_ops.get_params_eltwise(cur_post_op_idx, alg, alpha, beta);
                p_ops.get_params_binary(prev_post_op_idx, alg, desc);

                // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                auto bin_ops_can_be_optimized = prev_node.is_type<data>() && prev_node.is_constant() &&
                                                prev_node.get_users().size() == 1 && desc.get_data_type() == dnnl_f32;

                auto eltw_and_bin_add = alpha == 1.0f && type_is_binary_add(prev_type) && bin_ops_can_be_optimized;
                auto eltw_and_bin_mul = beta == 0.f && type_is_binary_mul(prev_type) && bin_ops_can_be_optimized;

                if (eltw_and_bin_add || eltw_and_bin_mul) {
                    memory::ptr prev_bin_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                    if (prev_bin_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for eltw + bin");
                    auto& stream = prev_bin_mem_ptr->get_engine()->get_service_stream();
                    mem_lock<float, mem_lock_type::read_write> eltw_and_bin_lock(prev_bin_mem_ptr, stream);

                    size_t prev_bin_mem_size = prev_node.get_output_layout().count();

                    // Update all binary coefficients
                    if (eltw_and_bin_add) {
                        for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                            eltw_and_bin_lock[data_idx] += beta;
                        }
                    } else {
                        for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                            eltw_and_bin_lock[data_idx] *= alpha;
                        }
                    }

                    // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            } else if (eltw_and_scale) {
                dnnl::algorithm alg;
                float alpha, beta;

                cldnn::program_node& prev_node = get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                p_ops.get_params_eltwise(cur_post_op_idx, alg, alpha, beta);

                // Eltwise can be inserted into the output_scale if cur_beta is equal to 0.f
                if (beta == 0.f && prev_node.get_output_layout().data_type == data_types::f32) {
                    memory::ptr prev_scale_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                    if (prev_scale_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for eltw + scale");
                    auto& stream = prev_scale_mem_ptr->get_engine()->get_service_stream();
                    mem_lock<float, mem_lock_type::read_write> eltw_and_scale_lock(prev_scale_mem_ptr, stream);

                    size_t prev_scale_mem_size = prev_node.get_output_layout().count();

                    // Update all scale coefficients
                    for (size_t data_idx = 0; data_idx < prev_scale_mem_size; data_idx++) {
                        eltw_and_scale_lock[data_idx] *= alpha;
                    }

                    // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            }
        }

        // If no optimizations have been applied then copy post-op info into the new optimized_p_ops structure
        if (!cur_ops_pair_is_optimized) {
            add_post_op(prev_type, p_ops, optimized_p_ops, prev_post_op_idx);
        }

        if (cur_post_op_idx == post_ops_size - 1 && !cur_ops_pair_is_optimized) {
            add_post_op(cur_type, p_ops, optimized_p_ops, cur_post_op_idx);
            optimization_done = true;
        } else if (cur_post_ops[cur_post_op_idx].op_type != onednn_post_op_type::optimized && cur_post_op_idx < post_ops_size - 1) {
            cur_post_op_idx++;
            prev_post_op_idx++;
        }
    }

    // if optimization_is_completed is true, try to optimize again.
    optimization_is_completed = !optimization_is_completed;
    if (optimization_is_completed) {
        remove_optimized_prefix(cur_post_ops);
    }

    GPU_DEBUG_TRACE << ">>>>>>>>>>>>>-----------------------------------" << std::endl;
    for (size_t i = 0; i < cur_post_ops.size(); i++)
        GPU_DEBUG_TRACE << "    " << i << ": " << cur_post_ops[i].op_type << std::endl;
    GPU_DEBUG_TRACE << "------------------------------------------------" << std::endl;

    add_onednn_fused_primitives(cur_post_ops);

    return optimized_p_ops;
}


void program_node::init_onednn_primitive_attributes() {
    const std::vector<fused_primitive_desc>& cldnn_post_ops = get_fused_primitives();
    auto attrs = std::make_shared<dnnl::primitive_attr>();
    dnnl::post_ops post_ops;
    size_t memory_offset = 0;

    // Create onednn post-ops list related to the current node
    std::vector<fused_primitive_desc_onednn> fused_ops;

    // Added this for debug purposes only
    size_t empty_mem = 0xff;

    // Add information about post-operation into the list, update indices
    auto update_onednn_post_op_list = [&](onednn_post_op_type type, size_t m_dep,
                                          dnnl::memory::format_tag tag = dnnl::memory::format_tag::undef,
                                          bool flatten = false) {
        fused_primitive_desc_onednn cur_op_desc = { type, memory_offset, m_dep, tag, flatten };
        fused_ops.push_back(cur_op_desc);

        auto has_memory_buffers = type == onednn_post_op_type::binary_add ||
                                  type == onednn_post_op_type::binary_sub ||
                                  type == onednn_post_op_type::binary_mul ||
                                  type == onednn_post_op_type::binary_max ||
                                  type == onednn_post_op_type::binary_min ||
                                  type == onednn_post_op_type::binary_relu ||
                                  type == onednn_post_op_type::scale ||
                                  type == onednn_post_op_type::sum;

        if (has_memory_buffers)
            memory_offset++;
    };

    int32_t num_sum_post_ops = 0;
    for (size_t idx = 0; idx < cldnn_post_ops.size(); idx++) {
        auto& desc = cldnn_post_ops[idx];
        if (desc.is_type<activation>()) {
            auto fused_desc = desc.typed_desc<activation>();
            if (fused_desc->activation_function == cldnn::activation_func::relu_negative_slope
                && !fused_desc->additional_params_input.empty()) {
                auto dep_idx = cldnn_post_ops[idx].dep_start_idx;
                int oc_dim = desc.output_layout.get_tensor().feature.size();
                post_ops.append_prelu(1 << oc_dim);
                update_onednn_post_op_list(onednn_post_op_type::binary_relu, dep_idx);
            } else if (fused_desc->activation_function == cldnn::activation_func::hard_sigmoid) {
                post_ops.append_eltwise(dnnl::algorithm::eltwise_hardsigmoid, fused_desc->additional_params.a, fused_desc->additional_params.b);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_hardsigmoid, empty_mem);
            } else if (fused_desc->activation_function == cldnn::activation_func::hsigmoid) {
                // hard_sigmoid(x,a,b) = clamp(ax+b, 0, 1)
                // hsigmoid(x) = clamp(val+3, 0, 6) / 6 = clamp(val/6+0.5, 0, 1) = hard_sigmoid(val, 1/6, 1/2)
                post_ops.append_eltwise(dnnl::algorithm::eltwise_hardsigmoid, 1./6, 1./2);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_hardsigmoid, empty_mem);
            } else if (fused_desc->activation_function == cldnn::activation_func::negative) {
                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, -1, 0);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
            } else {
                dnnl::algorithm alg = onednn::convert_activation_func(fused_desc->activation_function);
                if (alg == dnnl::algorithm::undef)
                    IE_THROW() << "Activations that are undef algorithms must be converted to other activations before "
                                  "pushing to post-op.";
                // Usage of alpha and beta between cldnn::pow and dnnl::eltwise::pow is different : d = pow(src, a) / d = a * pow(src, b)
                if (alg == dnnl::algorithm::eltwise_pow)
                    post_ops.append_eltwise(alg, 1.0f, fused_desc->additional_params.a);
                else if (alg == dnnl::algorithm::eltwise_hardswish) // mapping val * min(max(0, val + 3), 6) / 6 to val * max(min(1, alpha * val + beta), 0)
                    post_ops.append_eltwise(alg, 1/6.f, 0.5f);
                else
                    post_ops.append_eltwise(alg, fused_desc->additional_params.a, fused_desc->additional_params.b);

                update_onednn_post_op_list(onednn_post_op_type::eltwise_act, empty_mem);
            }
        } else if (desc.is_type<eltwise>()) {
            auto dep_idx = desc.dep_start_idx;
            auto in = get_dependency(dep_idx).get_output_layout();

            auto set_binary_op = [&](dnnl::algorithm alg, onednn_post_op_type op_type) {
                if (is_type<fully_connected>()) {
                    std::unique_ptr<const kernel_impl_params> impl_params = get_kernel_impl_params();
                    auto prim = impl_params->typed_desc<fully_connected>();
                    if (prim->input_size == 3) {
                        cldnn::onednn::combine_bf_with_first_spatial_dim(in);
                    }
                    post_ops.append_binary(alg, onednn::layout_to_memory_desc(in, dnnl::memory::format_tag::ab));
                    update_onednn_post_op_list(op_type, dep_idx);
                } else if (is_type<gemm>()) {
                    size_t rank = cldnn::format::dimension(in.format);
                    dnnl::memory::dims dims = onednn::convert_gemm_tensor(in.get_tensor(), rank, in.batch() > 1);
                    dnnl::memory::data_type dt = onednn::convert_data_type(in.data_type);
                    dnnl::memory::format_tag fmt = onednn::convert_gemm_data_format(dims);
                    post_ops.append_binary(alg, dnnl::memory::desc(dims, dt, fmt));
                    update_onednn_post_op_list(op_type, dep_idx);
                } else {
                    post_ops.append_binary(alg, onednn::layout_to_memory_desc(in));
                    update_onednn_post_op_list(op_type, dep_idx);
                }
            };

            if (desc.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
                auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(*this, cldnn_post_ops[idx]);
                if (fusing_type == add_fusing_type::sum && num_sum_post_ops == 0) {
                    if (is_type<convolution>()) {
                        post_ops.append_sum(1.0f, 0/*zero-point*/, onednn::convert_data_type(in.data_type));
                    } else {
                        post_ops.append_sum(1.0f);
                    }
                    update_onednn_post_op_list(onednn_post_op_type::sum, dep_idx);
                    num_sum_post_ops++;
                } else {
                    set_binary_op(dnnl::algorithm::binary_add, onednn_post_op_type::binary_add);
                }
            } else if (desc.typed_desc<eltwise>()->mode == eltwise_mode::sub) {
                set_binary_op(dnnl::algorithm::binary_sub, onednn_post_op_type::binary_sub);
            } else if (desc.typed_desc<eltwise>()->mode == eltwise_mode::prod) {
                set_binary_op(dnnl::algorithm::binary_mul, onednn_post_op_type::binary_mul);
            } else {
                std::stringstream error_msg;
                error_msg << "Unsupported eltwise mode: " << static_cast<int>(desc.typed_desc<eltwise>()->mode) << ". ";
                error_msg << desc.desc->id << " is fused node of " + this->id() + ".";
                OPENVINO_ASSERT(false, error_msg.str());
            }
        } else if (desc.is_type<quantize>()) {
            auto dep_idx = desc.dep_start_idx;

            // ********************************* Common case with output range usage ********************************* //
            const auto& q_param = desc.get_typed_fuse_params<kernel_selector::quantize_fuse_params>();
            if (q_param->per_tensor_output_range && q_param->out_lo < q_param->out_hi) {
                // 1. pre-scale & pre-shift
                {
                    if (q_param->per_tensor_input_scale && q_param->per_tensor_input_shift) {
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->in_scale, q_param->in_shift);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_param->per_tensor_input_scale) {
                            post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->in_scale, 0.0f);
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            auto in_scale = get_dependency(dep_idx++).get_output_layout();
                            dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                            post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                            update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                        }

                        if (q_param->has_pre_shift) {
                            if (q_param->per_tensor_input_shift) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, q_param->in_shift);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }
                    }
                }

                // 2. round
                auto out_dt = desc.output_layout.data_type;
                {
                    bool output_type_is_int8 = out_dt == data_types::u8 || out_dt == data_types::i8;
                    if (!output_type_is_int8) {
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                    }
                }

                // 3. post-scale & post-shift
                {
                    if (q_param->has_post_scale && q_param->has_post_shift &&
                        q_param->per_tensor_output_scale && q_param->per_tensor_output_shift) {
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->out_scale, q_param->out_shift);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_param->has_post_scale) {
                            if (q_param->per_tensor_output_scale) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->out_scale, 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }

                        if (q_param->has_post_shift) {
                            if (q_param->per_tensor_output_shift) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, q_param->out_shift);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }
                    }
                }

                // 4. clamp
                {
                    if (q_param->has_clamp || idx < cldnn_post_ops.size() - 1) {
                        float out_lo = q_param->has_min_clamp ? q_param->out_lo : data_type_traits::min<float>(out_dt);
                        float out_hi = q_param->has_max_clamp ? q_param->out_hi : data_type_traits::max<float>(out_dt);
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_clip, out_lo, out_hi);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_clip, empty_mem);
                    }
                }
            // ********************************* Rare case with input range usage ********************************* //
            } else {
                // 1. clamp
                {
                    if (q_param->has_clamp) {
                        auto in_lo = get_dependency(dep_idx++).get_output_layout();
                        auto in_hi = get_dependency(dep_idx++).get_output_layout();
                        dnnl::algorithm clamp_max = dnnl::algorithm::binary_max;
                        dnnl::algorithm clamp_min = dnnl::algorithm::binary_min;
                        dnnl::memory::desc in_lo_desc = onednn::layout_to_memory_desc(in_lo, dnnl::memory::format_tag::ab, true);
                        dnnl::memory::desc in_hi_desc = onednn::layout_to_memory_desc(in_hi, dnnl::memory::format_tag::ab, true);

                        post_ops.append_binary(clamp_max, in_lo_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_max, dep_idx - 2, dnnl::memory::format_tag::ab, true);
                        post_ops.append_binary(clamp_min, in_hi_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_min, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                    }
                }

                // 2. pre-scale & pre-shift
                {
                    if (q_param->per_tensor_input_scale && q_param->per_tensor_input_shift) {
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->in_scale, q_param->in_shift);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_param->per_tensor_input_scale) {
                            post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->in_scale, 0.0f);
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            auto in_scale = get_dependency(dep_idx++).get_output_layout();
                            dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                            post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                            update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                        }

                        if (q_param->has_pre_shift) {
                            if (q_param->per_tensor_input_shift) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, q_param->in_shift);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }
                    }
                }

                // 3. round
                {
                    post_ops.append_eltwise(dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                    update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                }

                // 4. post-scale & post-shift
                {
                    if (q_param->has_post_scale && q_param->has_post_shift &&
                        q_param->per_tensor_output_scale && q_param->per_tensor_output_shift) {
                        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->out_scale, q_param->out_shift);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_param->has_post_scale) {
                            if (q_param->per_tensor_output_scale) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, q_param->out_scale, 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }

                        if (q_param->has_post_shift) {
                            if (q_param->per_tensor_output_shift) {
                                post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, q_param->out_shift);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1, dnnl::memory::format_tag::ab, true);
                            }
                        }
                    }
                }
            }
        } else if (desc.is_type<reorder>()) {
            continue;
        } else {
            throw std::runtime_error("Unsupported fused op of " + desc.desc->type_string() + " type for oneDNN primitive");
        }
    }

    // Trying to optimize more than 1 post-ops
    if (fused_ops.size() > 1) {
        dnnl::post_ops optimized_post_ops = post_ops;
        bool optimization_is_finished = false;

        add_onednn_fused_primitives(fused_ops);

        // Trying to combine multiplications and additions which are placed one after another.
        // We do it in the cycle because some optimization cases can be simplified again from time to time
        do {
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->disable_onednn_opt_post_ops)
                break;
            optimized_post_ops = try_optimize_post_ops(optimized_post_ops, attrs, optimization_is_finished);
        } while (!optimization_is_finished);

        attrs->set_post_ops(optimized_post_ops);
    } else {
        // Set post-ops without any optimizations
        add_onednn_fused_primitives(fused_ops);
        attrs->set_post_ops(post_ops);
    }

    add_onednn_attrs(attrs);
}


#endif // ENABLE_ONEDNN_FOR_GPU
