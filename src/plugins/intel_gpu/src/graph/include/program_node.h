// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/program.hpp"

#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include <set>
#include <array>
#include <vector>
#include <memory>
#include <list>
#include <algorithm>
#include <thread>

namespace cldnn {

struct program;
struct primitive_impl;
class reorder_inputs;
class graph_initializations;
class prepare_quantization;
class pre_replace_deconv;

template <class T>
struct typed_program_node;

class json_composite;
class xml_composite;

/*
    Base class for all primitives which wraps API class and extends it to be used
    in graph context.

    Besides primitive description provided by user, this class includes functionality to
    ask for direct predecessors and succesors as well as takes care of changes to primitive
    which would affect other graph's nodes (the most commont case is probably calculating output layout).

    At graph level, all connections between nodes are directly stored inside program_nodes - in oposite
    to API level where all primitives store only ids of related ones.
*/
struct program_node {
    friend struct program;                          // to be removed when possible
    friend class compile_graph;                     // to be removed when possible
    friend class graph_initializations;             // to be removed when possible
    friend class pre_replace_deconv;                // to be removed when possible
    friend class prepare_primitive_fusing;          // to be removed when possible
    friend class prepare_quantization;              // to be removed when possible
    friend class propagate_constants;               // to be removed when possible

    template <class PType>
    friend struct typed_program_node;

    program_node(std::shared_ptr<primitive> prim, program& prog);

    program_node(program_node const&) = delete;

    virtual ~program_node() = default;

public:
    virtual const primitive_id& id() const { return desc->id; }
    virtual primitive_type_id type() const { return desc->type; }
    virtual std::shared_ptr<NodeFuseParams> get_fuse_params() const { return nullptr; }

    virtual std::vector<size_t> get_shape_infer_dependencies() const {
        // Default impl will request all deps for shape infer
        // It means that update_shape impl will wait for all memory deps
        // TODO: Return empty vector once all impls have proper overloaded function
        std::vector<size_t> res(get_dependencies().size());
        std::iota(std::begin(res), std::end(res), 0);
        return res;
    }

    bool is_shape_infer_dep(void) const {
        if (!myprog.is_new_shape_infer())
            return false;
        for (auto u : users) {
            for (auto dep_idx : u->get_shape_infer_dependencies()) {
                if (u->get_dependencies().size() <= dep_idx) {
                    continue;
                }
                if (u->is_fused_dep(dep_idx)) {
                    continue;
                }
                if (u->get_dependencies().at(dep_idx).first == this) {
                    return true;
                }
            }
        }
        return false;
    }

    bool is_fused_dep(size_t dep_idx) const;

    bool has_fused_dep() const {
        for (auto& fused : get_fused_primitives()) {
            if (fused.has_outer_dep())
                return true;
        }
        return false;
    }

    int32_t get_first_fused_dep_idx() const {
        if (!has_fused_dep())
            return -1;
        for (auto& fused : get_fused_primitives()) {
            if (fused.has_outer_dep())
                return fused.outer_dep_start_idx;
        }
        return -1;
    }

    std::map<size_t, memory::ptr> get_const_memory_deps() const;

    virtual std::unique_ptr<kernel_impl_params> get_kernel_impl_params() const {
        return get_kernel_impl_params(get_input_layouts(), output_layouts);
    }

    virtual std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const {
        auto params = std::unique_ptr<kernel_impl_params>(new kernel_impl_params(get_program(), get_program().get_engine().get_device_info().dev_type,
                                                                                 get_program().get_stream_ptr(), get_primitive(),
                                                                                 get_unique_id(), in_layouts, out_layouts, get_fused_primitives()));
        params->memory_deps = get_const_memory_deps();
        params->_can_be_optimized = this->optimized;
        params->in_port_to_shape_info_offset = get_input_port_to_shape_info_offset_map();
        params->out_port_to_shape_info_offset = get_output_port_to_shape_info_offset_map();
        auto deps = get_dependencies();
        for (size_t i = 0; i < deps.size(); i++) {
            if (!deps[i].first->is_constant()) {
                params->primary_input_idx = i;
                break;
            }
        }
#ifdef ENABLE_ONEDNN_FOR_GPU
        params->fused_desc_onednn   = get_fused_primitives_onednn();
        params->attrs_onednn        = get_onednn_primitive_attributes();
#endif // ENABLE_ONEDNN_FOR_GPU
        return params;
    }

    template <class PType>
    bool is_type() const {
        static_assert(
            meta::is_primitive<PType>::value,
            "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return type() == PType::type_id();
    }

    program& get_program() { return myprog; }
    program& get_program() const { return myprog; }

    primitive_impl* get_selected_impl() const { return selected_impl.get(); }
    void set_selected_impl(std::unique_ptr<primitive_impl> impl);

    void set_preferred_impl_type(impl_types impl) { impl_type = impl; }
    impl_types get_preferred_impl_type() const { return impl_type; }

    std::vector<std::pair<program_node*, int32_t>> const& get_dependencies() const { return dependencies; }
    program_node& get_dependency(size_t idx) const { return *dependencies.at(idx).first; }
    std::pair<program_node*, int32_t> get_dependency_with_port(size_t idx) const { return dependencies.at(idx); }

    // Count of original primitive inputs, i.e. it doesn't include fused dependencies
    size_t get_inputs_count() const { return desc->input_size(); }
    // Count of original primitive outputs
    size_t get_outputs_count() const { return desc->output_size(); }

    std::vector<layout> const get_input_layouts() const;
    const layout& get_input_layout(size_t idx = 0) const;
    const ov::PartialShape& get_input_pshape(size_t idx = 0) const;
    ov::PartialShape get_output_pshape(size_t idx = 0) const;

    virtual std::vector<layout> get_shape_info_input_layouts() const;
    std::map<size_t, size_t> get_input_port_to_shape_info_offset_map() const;
    std::map<size_t, size_t> get_output_port_to_shape_info_offset_map() const;
    size_t get_total_shape_info_input_size() const;
    size_t get_total_shape_info_output_size() const;
    size_t get_total_shape_info_size() const;

    // replaces idx-th dependency of 'this' with 'new_dep', calls program::remove_if_dangling(old_dep)
    void replace_dependency(size_t idx, program_node& new_dep, bool remove_if_dangling = true);
    void replace_dependency(size_t idx, std::pair<program_node*, int32_t> new_dep, bool remove_if_dangling = true);
    // searches for 'old_dep' in dependencies list of 'this' and replaces it with 'new_dep', calls
    // program::remove_if_dangling(old_dep)
    void replace_dependency(program_node const& old_dep, program_node& new_dep, bool remove_if_dangling = true);
    void replace_dependency(program_node const& old_dep, std::pair<program_node*, int32_t> new_dep, bool remove_if_dangling = true);

    std::vector<primitive_id> get_dependencies_ids() const;

    void remove_dependency(size_t idx);
    void remove_dependency(program_node& node);

    int32_t get_dependency_output_port(const program_node& node) const;
    size_t get_dependency_index(const program_node& node) const;
    size_t get_user_index(const program_node& node) const;

    std::unordered_set<size_t> get_memory_dependencies() const;
    void add_memory_dependency(size_t);
    void add_memory_dependency(std::vector<size_t>);

    template <class PType>
    bool have_user_with_type() const {
        for (auto const& usr : users) {
            if (usr->is_type<PType>())
                return true;
        }
        return false;
    }

    bool is_detached(bool whole_branch = false);

    std::list<program_node*> const& get_users() { return users; }
    // for const method, add const to stored successors/predecessors
    std::list<const program_node*> const& get_users() const {
        return reinterpret_cast<const std::list<const program_node*>&>(users);
    }

    std::unique_ptr<json_composite> desc_to_json() const;
    // do not modify primitive directly to keep synchronisation with graph
    std::shared_ptr<const primitive> get_primitive() const { return desc; }
    // primitive modification functions
    void set_output_padding(padding const& padd, size_t idx = 0) {
        // changing output padding shouldn't cause any changes to other primitives
        // so just change it
        output_layouts[idx].data_padding = padd;
    }

    void merge_output_padding(padding const& padd, size_t idx = 0) {
        set_output_padding(padding::max(padd, output_layouts[idx].data_padding), idx);
    }

    // only calculated output layout (for external usage), does not modify/use cached output layout nor invalidate users
    layout calc_output_layout() const;
    std::vector<layout> calc_output_layouts() const;

    // uses cached output layout if valid, if not calls 'calc_output_layout' and stores its result + invalidate all
    // users if layout has changed and @p invalidate_users_if_changed is set to true
    const layout& get_output_layout(bool invalidate_users_if_changed = true, size_t idx = 0);
    // returns cached output layout if valid, otherwise throws an exception
    const layout& get_output_layout(size_t idx = 0) const;
    const std::vector<layout>& get_output_layouts(bool invalidate_users_if_changed = true);
    const std::vector<layout>& get_output_layouts() const;
    // returns result of get_output_layout without padding
    layout get_non_padded_output_layout(bool invalidate_users_if_changed = true, size_t idx = 0);

    // sets cached output layout to an arbitrary value, invalidates users if new layout differs from previous one and @p
    // invalidate_users_if_changed is set to true returns whether output layout has changed
    bool set_output_layout(layout& new_layout, bool invalidate_users_if_changed = true, size_t idx = 0);
    bool set_output_layouts(std::vector<layout>& new_layout, bool invalidate_users_if_changed = true);

    // forces recalculation of cached output layout, invalidates users if new layout is different than previous one and
    // @p invalidate_users_if_changed is set to true returns whether output layout has changed
    bool recalc_output_layout(bool invalidate_users_if_changed = true);
    bool recalc_output_layouts(bool invalidate_users_if_changed = true);

    bool is_dynamic() const;
    bool is_dynamic();

    bool is_dynamic_output_layout(size_t idx = 0) const;
    bool is_dynamic_output_layout(size_t idx = 0);

    bool is_padded() { return static_cast<bool>(get_output_layout().data_padding); }
    bool is_padded() const { return static_cast<bool>(get_output_layout().data_padding); }

    bool has_padded_dependency();
    bool has_padded_dependency() const;

    bool is_input() const { return dependencies.empty(); }
    bool is_endpoint() const { return users.empty(); }
    void set_output(bool out) { output = out; }
    bool is_output() const { return output; }

    bool is_valid_output_layout(size_t idx = 0) const { return valid_output_layouts[idx]; }
    bool is_all_valid_output_layouts() const {
        for (auto l : valid_output_layouts) {
            if (l == false) return false;
        }
        return true;
    }

    uint8_t mark(uint8_t val = 1) {
        uint8_t ret = user_mark;
        user_mark = val;
        return ret;
    }
    void unmark() { user_mark = 0; }
    bool is_marked() const { return user_mark != 0; }

    void set_in_shape_of_subgraph(bool val = true) { in_shape_of_subgraph = val; }
    bool is_in_shape_of_subgraph() const { return in_shape_of_subgraph; }

    // check/set if the node can be optimized out (removed from the network)
    bool can_be_optimized() const { return optimized; }
    void can_be_optimized(bool opt) { optimized = opt; }

    // check/set if the node is runtime skippable
    bool is_runtime_skippable() const { return runtime_skippable; }
    void set_runtime_skippable(bool skippable) { runtime_skippable = skippable; }

    // check/set if the node's buffer can be shared during the memory pool optimization
    bool can_share_buffer() const { return share_buffer; }
    void can_share_buffer(bool share) { share_buffer = share; }

    // Sets padding support for all axis
    void support_padding_all(bool support);
    // Sets padding support for specified axis
    void support_padding(int axis, bool support) { _support_padding_in_axis[axis] = support; }

    // Checks if primitive supports any padding in specified axis
    bool support_padding(int axis) const { return _support_padding_in_axis[axis]; }
    // Checks whether with current format specified padding is supported;
    bool is_padding_supported(int axis, int padding) const;
    // Check if layout has padding in any spatial axis
    bool is_padded_spatial(size_t idx = 0) const;

    primitive_id get_org_primitive_id() const { return org_id; }

    bool is_constant() const { return constant; }

    // returns true if this node is within main data flow of the network (i.e. it does not describe helper data like
    // convolution's weights etc.)
    bool is_in_data_flow() const { return data_flow; }

    // conversion from generic to specific
    template <class To, class..., class = typename std::enable_if<!std::is_same<To, primitive>::value>::type>
    typed_program_node<To>& as() {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To>&>(*this);
    }

    template <class To, class..., class = typename std::enable_if<!std::is_same<To, primitive>::value>::type>
    typed_program_node<To> const& as() const {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To> const&>(*this);
    }

    template <class To>
    operator typed_program_node<To>&() {
        return as<To>();
    }

    template <class To>
    operator typed_program_node<To> const&() const {
        return as<To>();
    }

    virtual std::set<size_t> get_lockable_input_ids() const;

    void add_dependant_shape_of_node(const program_node* node);

    const std::set<const program_node*>& get_dependant_shape_of_nodes() const {
        return dependant_shape_of_nodes;
    }

    void set_reused_memory_color(uint32_t color) const {
        has_reused_memory = true;
        reused_memory_color = color;
    }

    bool is_reusing_memory() { return has_reused_memory; }
    uint32_t get_reused_memory_color() {
        return reused_memory_color;
    }

    void add_fused_primitive(fused_primitive_desc& d) {
        fused_prims.push_back(d);
    }

    void add_fused_primitives(std::vector<fused_primitive_desc> descs) {
        fused_prims.insert(fused_prims.end(), descs.begin(), descs.end());
    }

    const std::vector<fused_primitive_desc>& get_fused_primitives() const { return fused_prims; }
    std::vector<fused_primitive_desc>& get_fused_primitives() { return fused_prims; }

    void save(cldnn::BinaryOutputBuffer& ob) const;
    void load(cldnn::BinaryInputBuffer& ib);

#ifdef ENABLE_ONEDNN_FOR_GPU
    const std::shared_ptr<dnnl::primitive_attr>& get_onednn_primitive_attributes() const {
        if (onednn_attrs == nullptr)
            const_cast<program_node*>(this)->init_onednn_primitive_attributes();
        return onednn_attrs;
    }
    std::shared_ptr<dnnl::primitive_attr>& get_onednn_primitive_attributes() {
        if (onednn_attrs == nullptr)
            init_onednn_primitive_attributes();
        return onednn_attrs;
    }

    const std::vector<fused_primitive_desc_onednn>& get_fused_primitives_onednn() const { return fused_prims_onednn; }
    std::vector<fused_primitive_desc_onednn>& get_fused_primitives_onednn() { return fused_prims_onednn; }

    void init_onednn_primitive_attributes();
    void create_onednn_primitive_attributes(
                                const std::vector<fused_primitive_desc>& cldnn_post_ops,
                                std::shared_ptr<dnnl::primitive_attr>& attrs,
                                std::vector<fused_primitive_desc_onednn>& fused_ops,
                                kernel_impl_params* impl_params) const;
#endif // ENABLE_ONEDNN_FOR_GPU

    size_t get_fused_inputs_count() const {
        size_t count = 0;
        for (auto& fp : get_fused_primitives()) {
            count += fp.deps.size();
        }
        return count;
    }

    bool has_fused_primitives() const { return !get_fused_primitives().empty(); }

    layout get_fused_output_layout() const {
        auto fp = get_fused_primitives();
        if (fp.empty())
            return layout(data_types::f32, format::bfyx, tensor());
        return fp.back().output_layout;
    }

    bool need_lockable_memory() const;

    size_t get_unique_id() const { return unique_id; }

    void set_unique_id() {
        unique_id = cur_id++;
    }

    void set_unique_id(size_t _id) {
        unique_id = _id;
    }


    static void reset_unique_id() {
        cur_id = 0;
    }

    std::vector<format::type> get_preferred_input_fmts() const { return preferred_input_fmts; }
    std::vector<format::type> get_preferred_output_fmts() const { return preferred_output_fmts; }
    format::type get_preferred_input_fmt(size_t idx = 0) const {
        return (idx < preferred_input_fmts.size()) ? preferred_input_fmts.at(idx) : format::any;
    }
    format::type get_preferred_output_fmt(size_t idx = 0) const {
        return (idx < preferred_output_fmts.size()) ? preferred_output_fmts.at(idx) : format::any;
    }

    void init_preferred_fmt(size_t dep_size, size_t user_size);
    void set_preferred_input_fmt(size_t idx, format::type type);
    void set_preferred_output_fmt(size_t idx, format::type type);

    int32_t get_port_from_deps(primitive_id target_id) const {
        auto deps = get_primitive()->dependencies();
        auto iter = std::find_if(deps.begin(), deps.end(), [&](input_info& info) {
            return target_id == info.pid;
        });
        if (iter != deps.end()) {
            return iter->idx;
        } else {
            return 0;
        }
    }

protected:
    size_t unique_id = 0;
    static thread_local size_t cur_id;

    std::shared_ptr<primitive> desc;
    program& myprog;

    std::unique_ptr<primitive_impl> selected_impl;

    std::vector<bool> valid_output_layouts;
    std::vector<layout> output_layouts;

    std::vector<format::type> preferred_input_fmts;
    std::vector<format::type> preferred_output_fmts;

    std::vector<std::pair<program_node*, int32_t>> dependencies;
    std::list<program_node*> users;

    // list of primitives that can reuse same memory buffers due to execution order conflicts
    std::unordered_set<size_t> memory_dependencies;

    impl_types impl_type = impl_types::any;
    bool constant = false;
    bool data_flow = false;
    bool in_shape_of_subgraph = false;
    bool runtime_skippable = false;

    std::set<const program_node*> dependant_shape_of_nodes;

    bool output = false;
    uint8_t user_mark = 0;
    bool optimized = false;
    bool share_buffer = true;
    std::array<bool, tensor_dim_max> _support_padding_in_axis;

    mutable bool has_reused_memory = false;
    mutable uint32_t reused_memory_color = 0;

    const primitive_id org_id;

    std::vector<fused_primitive_desc> fused_prims;

    void invalidate_users() const;

private:
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::vector<fused_primitive_desc_onednn> fused_prims_onednn;
    std::shared_ptr<dnnl::primitive_attr> onednn_attrs;

    void add_onednn_fused_primitives(std::vector<fused_primitive_desc_onednn> descs) {
        fused_prims_onednn.erase(fused_prims_onednn.begin(), fused_prims_onednn.end());
        fused_prims_onednn.insert(fused_prims_onednn.end(), descs.begin(), descs.end());
    }

    void add_onednn_attrs(std::shared_ptr<dnnl::primitive_attr> attrs) {
        onednn_attrs = attrs;
    }

    dnnl::post_ops try_optimize_post_ops(std::vector<fused_primitive_desc_onednn>& cur_post_ops,
                                                    dnnl::post_ops& p_ops, const std::shared_ptr<dnnl::primitive_attr>& attr,
                                                    bool& optimization_is_completed) const;

#endif // ENABLE_ONEDNN_FOR_GPU
    size_t num_outputs = 1;
};

/*
Template class used to indicate that usage context requires 'program_node' to wrap primitive
of type 'PType'. Successful conversion from 'program_node' to 'typed_program_node<PType>' means
that this restriction in fact holds and functions/method/etc. may saftly use uderlaying primitive.

This class shadows 'get_primitive' method from base class which now returns pointer to more specific
type.
*/
template <class PType>
struct typed_program_node_base : public program_node {
    friend class cldnn::graph_initializations;
    friend class cldnn::pre_replace_deconv;
    friend class cldnn::prepare_quantization;
    friend struct cldnn::program;
    friend class cldnn::reorder_inputs;

public:
    using program_node::program_node;

    std::shared_ptr<const PType> get_primitive() const {
        return std::static_pointer_cast<const PType>(program_node::get_primitive());
    }

protected:
    std::shared_ptr<PType> typed_desc() const { return std::static_pointer_cast<PType>(desc); }
};

/*
    Actual template class used in context which requires 'program_node' to wrap
    primitive of type 'PType'. This class is introduced to provide possibility of explicit specialization.
    In most cases such specializations would add accessors to make access to PType-specific fields easier.

    It's not required to specialize this class for new primitives types.
*/
template <class PType>
struct typed_program_node : public typed_program_node_base<PType> {
    using typed_program_node_base<PType>::typed_program_node_base;

    program_node& input(size_t index = 0) const { return program_node::get_dependency(index); }
};

}  // namespace cldnn
