// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_node.h"
#include "program_impl.h"
#include "primitive_inst.h"
#include "to_string_utils.h"
#include "json_object.h"
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <set>

using namespace cldnn;

program_node::program_node(std::shared_ptr<primitive> prim, program_impl& prog)
    : desc(prim), myprog(prog), org_id(prim->id) {
    if (prim)
        output_layout.data_padding = prim->output_padding;
}

void program_node::replace_dependency(size_t idx, program_node& new_dep) {
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx] == &new_dep)
        return;

    auto it = std::find(dependencies[idx]->users.begin(), dependencies[idx]->users.end(), this);
    if (it != dependencies[idx]->users.end()) {
        dependencies[idx]->users.erase(it);
    }
    myprog.remove_if_dangling(*dependencies[idx]);

    dependencies[idx] = &new_dep;
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep) {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &old_dep)
            return replace_dependency(i, new_dep);
}

std::vector<primitive_id> program_node::get_dependencies_ids() const {
    std::vector<primitive_id> dep_ids;
    for (auto& dependency : dependencies) dep_ids.push_back(dependency->get_primitive()->id);
    return dep_ids;
}

void program_node::remove_dependency(size_t idx) {
    if (idx >= dependencies.size())
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx]);
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
    node_info->add("valid output layout", bool_to_str(valid_output_layout));
    std::stringstream s;
    s << get_preferred_impl_type();
    node_info->add("preferred impl", s.str());

    json_composite output_layout_info;
    output_layout_info.add("data type", dt_to_str(output_layout.data_type));
    output_layout_info.add("format", fmt_to_str(output_layout.format));
    output_layout_info.add("size", output_layout.size.to_string());

    json_composite padding_info;
    padding_info.add("lower size", output_layout.data_padding.lower_size().to_string());
    padding_info.add("upper size", output_layout.data_padding.upper_size().to_string());
    output_layout_info.add("padding info", padding_info);

    node_info->add("output layout", output_layout_info);

    node_info->add("in data flow", bool_to_str(data_flow));
    node_info->add("constant", bool_to_str(constant));
    node_info->add("in data flow", bool_to_str(data_flow));
    node_info->add("output", bool_to_str(output));


    json_composite fused_nodes_info;
    size_t index = 0;
    for (auto& fused_desc : get_fused_primitives()) {
        json_composite fused_node_info;
        fused_node_info.add("id", fused_desc.node->id());
        fused_node_info.add("dependencies", fused_desc.deps);
        fused_node_info.add("dep start_idx", fused_desc.dep_start_idx);
        json_composite info;
        info.add("data type", dt_to_str(fused_desc.output_layout.data_type));
        info.add("format", fmt_to_str(output_layout.format));
        info.add("size", output_layout.size.to_string());
        fused_node_info.add("output layout", info);
        fused_nodes_info.add("fused primitive idx " + std::to_string(index++), fused_node_info);
    }
    node_info->add("fused primitives", fused_nodes_info);

    std::vector<std::string> deps_ptrs;
    {
        bool empty = true;
        auto itr = dependencies.begin();
        while (itr != dependencies.end()) {
            if (empty) {
                empty = false;
            }
            deps_ptrs.push_back(std::to_string(reinterpret_cast<uintptr_t>(*itr++)));
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
        if (dependencies[i] == &node)
            remove_dependency(i);
}

bool program_node::is_detached(bool whole_branch) {
    if (!users.empty())
        return false;
    if (!whole_branch && !dependencies.empty())
        return false;
    return true;
}

layout program_node::calc_output_layout() const {
    return type()->calc_output_layout(*this);
}

layout program_node::get_output_layout(bool invalidate_users_if_changed) {
    if (valid_output_layout)
        return output_layout;

    auto new_layout = calc_output_layout();
    set_output_layout(new_layout, invalidate_users_if_changed);
    return output_layout;
}

layout program_node::get_output_layout() const {
    if (!valid_output_layout)
        throw std::runtime_error("Output layout not calculated");

    return output_layout;
}

layout program_node::get_non_padded_output_layout(bool invalidate_users_if_changed) {
    auto out_layout = get_output_layout(invalidate_users_if_changed);
    auto result = layout({out_layout.data_type, out_layout.format, out_layout.size});
    return result;
}

bool program_node::set_output_layout(layout& new_layout, bool invalidate_users_if_changed) {
    merge_output_padding(new_layout.data_padding);
    new_layout.data_padding = output_layout.data_padding;
    bool changed = (new_layout != output_layout);
    if (changed && invalidate_users_if_changed)  // output_layout has changed! invalidate users
        invalidate_users();

    output_layout = new_layout;
    valid_output_layout = true;
    return changed;
}

bool program_node::recalc_output_layout(bool invalidate_users_if_changed) {
    auto new_layout = calc_output_layout();
    return set_output_layout(new_layout, invalidate_users_if_changed);
}

bool program_node::has_padded_dependency() {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](program_node* node) {
        return node->is_padded();
    });
}

bool program_node::has_padded_dependency() const {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const program_node* node) {
        return node->is_padded();
    });
}

void program_node::invalidate_users() const {
    for (auto& user : users) {
        if (user->valid_output_layout) {
            user->valid_output_layout = false;
            user->invalidate_users();
        }
    }
}

void program_node::support_padding_all(bool support) {
    std::fill(_support_padding_in_axis.begin(), _support_padding_in_axis.end(), support);
}

bool program_node::is_padding_supported(int axis, int padding) const {
    if (!support_padding(axis))
        return false;

    auto fmt = output_layout.format;

    // WA for known cases of padding not supported in implementations
    if (fmt == format::b_fs_yx_fsv16) {
        if (axis == 0 || (axis == 1 && padding % 16 != 0))
            return false;
    }

    if (fmt == format::fs_b_yx_fsv32 && (axis == 0))
        return false;

    for (const auto& block : fmt.block_sizes()) {
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
        return n->get_selected_impl()->is_cpu();
    });

    return need_lockable_mem;
}
