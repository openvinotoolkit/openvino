// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "primitive_type.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/polymorphic_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "impls/ocl/kernels_cache.hpp"

// TODO: add generic interface for weights_reorder_params and get rid of this dependency
#include "impls/ocl/kernel_selector_helper.h"

#include <memory>
#include <vector>
#include <string>

namespace cldnn {

// checks if any user in a list is a cpu primitive
bool is_any_user_cpu(const std::list<const program_node*>& users);

class primitive_inst;

template <class PType>
class typed_primitive_inst;

struct ImplementationManager;

/*
    Base class for all implementations.
*/
struct primitive_impl {
    primitive_impl() = default;
    explicit primitive_impl(const std::shared_ptr<WeightsReorderParams>& params, std::string kernel_name = "", bool is_dynamic = false)
        : _weights_reorder_params(params), _kernel_name(kernel_name), _is_dynamic(is_dynamic) {
    }
    explicit primitive_impl(std::string kernel_name, bool is_dynamic = false) :
        primitive_impl(nullptr, std::move(kernel_name), is_dynamic) {}
    virtual ~primitive_impl() = default;

    virtual std::vector<layout> get_internal_buffer_layouts() const = 0;
    virtual std::set<size_t> get_lockable_internal_buffers() const { return {}; }
    virtual void set_node_params(const program_node&) {}
    virtual const std::string& get_type_info() const = 0;
    virtual void set_arguments(primitive_inst& instance) = 0;
    virtual void set_arguments(primitive_inst& instance, kernel_arguments_data& args) = 0;
    virtual event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) = 0;
    const std::string& get_kernel_name() const { return _kernel_name; }

    // class typed_primitive_gpu_impl override this with return false;
    virtual bool is_cpu() const { return true; }
    virtual bool is_onednn() const { return false; }
    virtual void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) = 0;
    virtual void init_by_cached_kernels(const kernels_cache&, std::vector<std::string>& cached_kernel_ids) {}
    virtual std::vector<std::string> get_cached_kernel_ids(const kernels_cache&) { return {}; }
    virtual std::unique_ptr<primitive_impl> clone() const = 0;
    virtual std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() { return {}; }
    virtual void reset_kernels_source() {}
    virtual std::vector<kernel::ptr> get_kernels() const { return {}; }
    virtual void save(cldnn::BinaryOutputBuffer& ob) const {
        ob << can_reuse_memory;
        ob << _kernel_name;
        ob << _is_dynamic;
        if (_weights_reorder_params == nullptr) {
            ob << false;
        } else {
            ob << true;
            _weights_reorder_params->save(ob);
        }
    }
    virtual void load(cldnn::BinaryInputBuffer& ib) {
        ib >> can_reuse_memory;
        ib >> _kernel_name;
        ib >> _is_dynamic;
        bool has_weights_reorder_params;
        ib >> has_weights_reorder_params;
        if (has_weights_reorder_params) {
            _weights_reorder_params = std::make_shared<WeightsReorderParams>();
            _weights_reorder_params->load(ib);
        }
    }
    // returns a pair of batch program hash and kernel entry of each ocl impl. Returns "" for other impl types.
    virtual std::pair<std::string, std::string> get_kernels_dump_info() const {
        return std::make_pair("", "");
    }

    // If this flag is set as false, the memory allocated for this primitive is not allowed to be reused
    bool can_reuse_memory = true;
    bool can_share_kernels = false;

    void set_dynamic(bool val) { _is_dynamic = val; }
    bool is_dynamic() const { return _is_dynamic; }

    virtual void update(primitive_inst& inst, const kernel_impl_params& impl_params) { }

    virtual bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const {
        OPENVINO_ASSERT(_is_dynamic, "[GPU] requires_update() is called for static shape implementation ", _kernel_name);
        return false;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params);

    virtual kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const {
        return primitive_impl::static_canonicalize_shapes(impl_params);
    }

    virtual void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) {}
    virtual std::vector<kernel::ptr> get_kernels() { return {}; }

    bool need_weights_reorder() const { return _weights_reorder_params != nullptr; }
    std::shared_ptr<WeightsReorderParams> get_weights_reorder_params() const { return _weights_reorder_params; }

    std::shared_ptr<kernel_impl_params> get_weights_reorder_kernel_params() const;

    const ImplementationManager* m_manager = nullptr;

protected:
    std::shared_ptr<WeightsReorderParams> _weights_reorder_params = nullptr;
    std::string _kernel_name;
    bool _is_dynamic = false;
};

struct ImplementationsFactory {
    ImplementationsFactory(const program_node* node);

    const program_node* m_node;
    std::vector<std::shared_ptr<ImplementationManager>> m_available_impls;
    program::ImplementationsCache& m_static_impls_cache;
    std::vector<std::shared_ptr<primitive_impl>> m_dynamic_impls_cache;

    std::shared_ptr<primitive_impl> get_primitive_impl_for_params(primitive_inst& inst, const kernel_impl_params& params, bool use_async_compilation);
    bool has(impl_types impl_type) const;
};

/*
    Base class for all primitive instances.
    It's main responsibility is to allocate memory required to run single, specified in ctor,
    program_node. It also contains informations about it's predecessor in network graph and checks (<<-- TODO)
    if output should be recalculated between network runs.
*/
class primitive_inst {
    template <class PType>
    friend class typed_primitive_inst;

public:
    primitive_inst(network& network);
    virtual ~primitive_inst() = default;

    const std::vector<std::pair<const primitive_inst*, int32_t>>& dependencies() const {
        return reinterpret_cast<std::vector<std::pair<const primitive_inst*, int32_t>> const&>(_deps);
    }

    memory& dep_memory(size_t index) const {
        auto dep = dependencies().at(index);
        return dep.first->output_memory(dep.second);
    }
    memory::ptr dep_memory_ptr(size_t index) const {
        auto dep = dependencies().at(index);
        return dep.first->output_memory_ptr(dep.second);
    }
    memory& output_memory(size_t index = 0) const { return *_outputs[index]; }
    memory::ptr output_memory_ptr(size_t index = 0) const { return _outputs[index]; }
    size_t inputs_memory_count() const { return _inputs_memory_count; }
    size_t outputs_memory_count() const { return _outputs_memory_count; }
    bool outputs_allocated() const {
        if (_outputs.empty()) return false;
        for (const auto& output : _outputs) {
            if (!output) return false;
        }
        return true;
    }
    primitive_type_id type() const { return _type; }
    const primitive_id& id() const { return _id; }
    const primitive_id& org_id() const { return _org_id; }
    bool can_be_optimized() const { return _can_be_optimized; }
    void set_can_be_optimized(bool optimized) {
        // TODO: consolidate to _impl_param in the future
        _impl_params->_can_be_optimized = optimized;
         this->_can_be_optimized = optimized;
    }
    std::shared_ptr<const primitive> desc() const { return _impl_params->desc; }
    program_node const& get_node() const { return *_node; }
    network& get_network() const { return _network; }
    uint32_t get_network_id() const;
    virtual event::ptr set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0);
    void check_memory_to_set(const memory& mem, const layout& layout) const;
    const std::list<const cldnn::program_node *>& get_users() const { return _node->get_users(); }
    const std::vector<primitive_inst*>& get_user_insts() const { return _users; }
    void init_users() {
        std::vector<primitive_id> users;
        for (auto u : get_users()) {
            users.push_back(u->id());
        }
        _users = _network.get_primitives(users);
    }

    const std::unordered_set<size_t>& get_runtime_memory_dependencies() const { return _runtime_memory_dependencies; }

    const kernel_impl_params* get_impl_params() const { return _impl_params.get(); }
    // return pointer to const to prevent arbitrary 'execute' call -> use primitive_inst.execute() instead
    const primitive_impl* get_impl() const { return _impl.get(); }
    primitive_impl* get_impl() { return _impl.get(); }
    void set_impl(std::unique_ptr<primitive_impl> impl) { _impl = std::move(impl); }

    memory& input_memory(size_t index = 0) const {
        if (index >= inputs_memory_count())
            throw std::range_error("input offset too big");
        return dep_memory(index);
    }

    memory::ptr input_memory_ptr(size_t index = 0) const {
        if (index >= inputs_memory_count())
            throw std::range_error("input offset too big");
        return dep_memory_ptr(index);
    }

    memory::ptr shape_info_memory_ptr() const { return _shape_info_memory; }

    event::ptr execute(const std::vector<event::ptr>& events);
    void init_kernels(const kernels_cache& kernels_cache) {
        _impl->init_kernels(kernels_cache, *_impl_params);
    }

    void set_arguments();

    void validate() const {
        OPENVINO_ASSERT(_impl != nullptr || is_dynamic(), "[GPU] Invalid impl object for ", id(), " primitive");
    }
    bool output_changed() const { return _output_changed; }
    void reset_output_change() { _output_changed = false; }

    bool shape_changed() const { return _shape_changed; }
    bool mem_changed() const { return _mem_changed; }
    void reset_shape_change() { _shape_changed = false; }
    void set_shape_change() { _shape_changed = true; }

    void build_deps();

    void update_paddings();
    void do_runtime_skip_reorder();
    void do_runtime_skip_gather();
    void do_runtime_skip_permute();
    void do_runtime_skip_strided_slice();
    void do_runtime_skip_broadcast();
    void do_runtime_in_place_concat();
    void do_runtime_in_place_kv_cache();
    void do_runtime_in_place_crop();
    void configure_shape_of_dependencies();

    memory::ptr fused_memory(size_t dep_id) const {
        return dep_memory_ptr(get_fused_mem_offset() + dep_id);
    }

    bool has_fused_primitives() const { return _impl_params->has_fused_primitives(); }
    size_t get_fused_mem_count() const { return _fused_mem_count; }
    size_t get_fused_mem_offset() const { return _fused_mem_offset; }
    bool has_mutable_input() const { return _has_mutable_input; }
    void set_mutable_input(bool val) { _has_mutable_input = val; }
    bool is_input() const { return _is_input; }
    bool is_output() const { return _is_output; }
    bool mem_allocated() const { return _mem_allocated; }
    bool is_dynamic() const { return _is_dynamic; }
    bool can_share_buffer() const { return _can_share_buffer; }
    bool is_constant() const { return _is_constant; }
    bool needs_completion_event() const { return _needs_completion_event; }
    bool has_unfused_subgraph() const { return (_unfused_subgraph != nullptr); }
    bool has_node() const { return _node != nullptr; }
    bool has_inner_networks() const;
    void allocate_internal_buffers(bool reset = true);

    static memory::ptr allocate_output(engine& engine,
                                       memory_pool& pool,
                                       const program_node& _node,
                                       const kernel_impl_params& impl_params,
                                       const std::unordered_set<size_t>& memory_dependencies,
                                       uint32_t net_id,
                                       bool is_internal,
                                       size_t idx = 0,
                                       bool reset_mem = true,
                                       bool is_output_buffer = false,
                                       memory* curr_memory = nullptr,
                                       bool runtime_alloc = false);

    std::vector<memory::ptr> get_intermediates_memories() const { return _intermediates_memory; }

    std::string get_implementation_name() const;

    void add_profiling_data(instrumentation::pipeline_stage stage, bool cache_hit, std::string memalloc_info, int64_t time, bool per_iter_mode = false);
    const std::unordered_map<size_t, std::tuple<int64_t, size_t>>& get_profiling_data() const { return _profiling_data; }
    const std::unordered_map<size_t, instrumentation::perf_counter_key>& get_profiling_info() const { return _profiling_info; }

    layout get_input_layout(size_t idx = 0) const { return _impl_params->get_input_layout(idx); }
    layout get_output_layout(size_t idx = 0) const { return _impl_params->get_output_layout(idx); }
    layout get_node_output_layout() const { return _node_output_layout; }
    void set_output_layout(const layout& new_out_lay, size_t idx = 0) {
        _impl_params->output_layouts[idx] = new_out_lay;
    }
    void set_inner_networks(const std::vector<network::ptr> inner_nets) {
        _impl_params->inner_nets = inner_nets;
    }
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::vector<cldnn::fused_primitive_desc_onednn>& get_fused_primitives_onednn() const { return _impl_params->fused_desc_onednn; }
#endif // ENABLE_ONEDNN_FOR_GPU
    template <class PType>
    std::shared_ptr<const PType> get_typed_desc() const { return _impl_params->typed_desc<PType>(); }

    virtual void update_output_memory() {}

    virtual int32_t get_prealloc_iter_num() { return -1; }
    virtual void update_shape_info_tensor(const kernel_impl_params& params);
    kernel_impl_params get_fake_aligned_params_if_possible(kernel_impl_params const& orig_impl_param);
    bool all_dependencies_cpu_impl() const;

protected:
    primitive_inst(network& network, program_node const& node, bool allocate_memory);

    network& _network;
    program_node const* _node;
    layout _node_output_layout;

    bool update_shape_done_by_other = false;
    bool allocation_done_by_other = false;
    std::unique_ptr<kernel_impl_params> _impl_params;
    std::shared_ptr<primitive_impl> _impl;
    std::shared_ptr<ImplementationsFactory> _impls_factory = nullptr;

    // this is a set of dependencies in terms of memory, if execution of this primitive requires data from another one,
    // it should be added to this set
    std::vector<std::pair<primitive_inst*, int32_t>> _deps;

    // List of depandant shape_of primitives for shape_of subgraphs
    std::vector<primitive_inst*> dependant_shape_of_insts;

    std::vector<primitive_inst*> _users;
    // this is a set of dependencies in terms of execution
    // execution of all primitives from this set should be enough to guarantee that all memory deps (see _deps)
    // will be valid when executing this primitive. Most of the time this set will be equal to the _deps minus all
    // cldnn::data (which don't need to be execued) -- this is default, but it is also possible to have, for example,
    // only one fused primitive which will calculate multiple outputs (for example device enqueue can work in such
    // manner) in general - this member is introduced to relax logical connection between primitives which have to be
    // executed and memories which are used by this primitive
    std::vector<primitive_inst*> _exec_deps;

    // List of primitive ids that this primitive can't share memory buffers with
    std::unordered_set<size_t> _runtime_memory_dependencies;

    // This is sub-network generated on demand to execute unfused primitives sequence instead of single fused primitive
    // Needed for dynamic path only, as fusion in some cases may be illegal, but it can't be checked on program build phase,
    // thus we do less restrictive fusion with runtime sanity check and unfusion when needed.
    cldnn::network::ptr _unfused_subgraph = nullptr;

    // _output is optional because its initialization might be postponed (reshape_inst may either allocate it's own
    // buffer or attach input as output
    // depending on reshape_node.is_in_place())
    std::vector<memory::ptr> _outputs;

    std::vector<memory::ptr> _intermediates_memory;

    mutable LruCache<layout, memory::ptr, layout::Hasher> _reordered_weights_cache;

    // Buffer to store actual shapes of dynamic tensor which is automatically asigned as 1st argument to shape agnostic kernels
    memory::ptr _shape_info_memory = nullptr;

    bool _output_changed;  // todo: implement output reuse if neither of inputs has changed
    bool _shape_changed = false;
    bool _mem_changed = false;
    bool _has_valid_input =
        true;  // by default all primitives has valid inputs, exception is input_layout (see input_layout_inst)
    bool _has_mutable_input = false;
    bool _mem_allocated = false;
    bool _is_dynamic = false;
    primitive_type_id _type;
    primitive_id _id;
    primitive_id _org_id;
    bool _is_input = false;
    bool _is_output = false;
    size_t _inputs_memory_count = 0;
    size_t _outputs_memory_count = 0;
    size_t _fused_mem_count = 0;
    size_t _fused_mem_offset = 0;
    bool _can_be_optimized = false;
    bool _can_share_buffer = true;
    bool _is_constant = false;
    bool _needs_completion_event = false;

    std::vector<size_t> _max_output_layout_count;
    std::vector<size_t> max_intermediates_memory_sizes;

    std::vector<memory::ptr> allocate_outputs(kernel_impl_params* updated_params = nullptr,
                                              bool reset_mem = true,
                                              bool runtime_alloc = false);
    memory::ptr allocate_internal_buffer(size_t idx, bool reset = true);
    void allocate_shape_info_memory();
    static std::vector<primitive_inst*> build_exec_deps(
        std::vector<std::pair<primitive_inst*, int32_t>> const& mem_deps);
    int32_t get_index_in_deps(memory::cptr arg) const;

    // event function called by primitive_inst::execute after checking if primitive should rerun and before calling
    // _impl->execute() mainly for reshape (to update output memory if reshape_node.is_in_place() == true)
    virtual void on_execute() {}

    virtual void update_shape();
    virtual event::ptr update_weights();

    void fill_shape_info_data(const layout& runtime_layout, const layout& node_layout, int32_t* shape_info_ptr, size_t& offset);
    bool use_async_compilation();
    // if primitive_inst doesn't replace impl to new impl(static impl with opt kerenl or dynamic impl), return false
    bool update_impl(bool use_async_compilation);
    event::ptr realloc_if_needed();

    cldnn::network::ptr get_unfused_subgraph();

    // This method checks if fusion applied to current primitive is valid.
    // Needed for dynamic case only, and basically tracks single problematic case at the moment:
    // eltwise primitive in dynamic case may be expressed as follows:
    // input1 (dynamic_shape_in1)    input2 (dynamic_shape_in2)
    //       \                 /
    //            eltwise (dynamic_shape_out)
    // Consider that eltwise is fused into primitive that produces input1 tensor. Then
    // this pattern may lead to the one of the following cases:
    // 1. dynamic_shape_in1 == dynamic_shape_in2 => supported fusion
    // 2. dynamic_shape_in1 > dynamic_shape_in2 => supported fusion with additional input broadcast
    // 3. dynamic_shape_in1 < dynamic_shape_in2 => illegal fusion pattern
    // If input2 is not constant, then we can't really understand which case it actuall is,
    // thus for performance reasons we allow fusions for dynamic shape regardless the actual case
    // and then using this method in runtime we check if the fusion was valid or not
    bool is_valid_fusion() const;

    static std::string generic_to_string(program_node const& node, const char* type_name);

    template<typename ShapeType>
    static std::vector<layout> forward_input0_shape(const kernel_impl_params& impl_param) {
        auto in_layout = impl_param.get_input_layout(0);
        auto output_type = impl_param.desc->output_data_types[0].value_or(in_layout.data_type);

        if (impl_param.has_fused_primitives()) {
            output_type = impl_param.get_output_element_type();
        }

        return { layout(in_layout.get<ShapeType>(), output_type, in_layout.format) };
    }

    virtual bool need_reset_input_memory(size_t) const {
        return false;
    }

    virtual bool need_reset_output_memory() const {
        for (const auto& user_inst : get_user_insts()) {
            // Check users of optimized_out inst, as the optimized out inst will not be able to
            // reset it's memory
            if (user_inst->can_be_optimized()) {
                if (user_inst->need_reset_output_memory())
                    return true;
                continue;
            }

            if (user_inst->need_reset_input_memory(user_inst->get_node().get_dependency_index(*_node)))
                return true;
        }
        return false;
    }

    // This could be implemented via single map std::unordered_map<instrumentation::perf_counter_key, std::tuple<int64_t, size_t>>
    // but the overhead on using perf_counter_key as map key is too big, thus we use hash as map key
    // and store mapping onto original perf_clounter_key for further data analysis and dumps
    std::unordered_map<size_t, std::tuple<int64_t, size_t>> _profiling_data;
    std::unordered_map<size_t, instrumentation::perf_counter_key> _profiling_info;
};

/*
Base class for all implementation of specified primitive type.
For example, all cpu convolution implementations should derive directly from typed_primitive_impl<convolution>.
GPU implementations should derive from typed_primitive_gpu_impl<convolution>;
*/
template <class PType>
struct typed_primitive_impl : public primitive_impl {
    static_assert(meta::is_primitive<PType>::value,
                  "PType should be a non-const, non-volatile class derived from primitive");

    using primitive_impl::primitive_impl;

private:
    event::ptr execute(const std::vector<event::ptr>& event, primitive_inst& instance) override {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to execute primitive implementation with mismatching primitive instance");

        return execute_impl(event, reinterpret_cast<typed_primitive_inst<PType>&>(instance));
    }

    std::vector<layout> get_internal_buffer_layouts() const override {
        return get_internal_buffer_layouts_impl();
    }

    virtual std::vector<layout> get_internal_buffer_layouts_impl() const {
        return {};
    }

    void set_arguments(primitive_inst& instance) override {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to set_arguments for primitive implementation with mismatching primitive instance");

        return set_arguments_impl(reinterpret_cast<typed_primitive_inst<PType>&>(instance));
    }

    void set_arguments(primitive_inst& instance, kernel_arguments_data& args) override {
        OPENVINO_ASSERT(instance.type() == PType::type_id(), "[GPU] Implementation type ", instance.type(),
                                                             " does not match primitive type ", PType::type_id());
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to set_arguments for primitive implementation with mismatching primitive instance");

        return set_arguments_impl(reinterpret_cast<typed_primitive_inst<PType>&>(instance), args);
    }

    virtual void set_arguments_impl(typed_primitive_inst<PType>& /*instance*/) {}
    virtual void set_arguments_impl(typed_primitive_inst<PType>& /*instance*/, kernel_arguments_data& /*args*/) {}
    virtual event::ptr execute_impl(const std::vector<event::ptr>& event, typed_primitive_inst<PType>& instance) = 0;
};

template <class PType>
class typed_primitive_inst_base : public primitive_inst {
public:
    using typed_node = typed_program_node<PType>;
    using typed_impl = typed_primitive_impl<PType>;

    const typed_node* node;
    std::shared_ptr<const PType> argument;

    template<typename T>
    static std::vector<layout> calc_output_layouts(const typed_node& node, const kernel_impl_params& impl_param) { return {}; }

    static kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param) {
        return std::move(orig_impl_param);
    }

    typed_primitive_inst_base(network& network, typed_node const& node)
        : typed_primitive_inst_base(network, node, do_allocate_memory(node)) {}

    typed_primitive_inst_base(network& network)
        : primitive_inst(network), node(nullptr), argument(nullptr) {}

protected:
    typed_primitive_inst_base(network& network, typed_node const& node, bool allocate_memory)
        : primitive_inst(network, node, allocate_memory), node(&node), argument(node.get_primitive()) {}

    typed_primitive_inst_base(network& network, typed_node const& node, memory::ptr buffer)
        : typed_primitive_inst_base(network, node, false) {
        _outputs[0] = std::move(buffer);
    }

private:
    bool do_allocate_memory(typed_node const& typ_node) {
        if (typ_node.get_output_layout().is_dynamic() && !typ_node.get_output_layout().has_upper_bound()) {
                return false;
        }

        if (typ_node.template have_user_with_type<concatenation>() && typ_node.get_users().size() == 1 &&
            typ_node.get_users().front()->can_be_optimized()) {  // check if the only user is concat
            return false;
        }
        return true;
    }
};

/*
    Template class which represents instance of primitive 'PType'.
    Each new primitive should explicitly specialize this class.
    The pattern is as follows:
        struct new_primitive {}; // C++ API layer
        template <>
        class typed_primitive_inst<new_primitive> : public typed_primitive_inst_base<new_primitive> {}; // network
   instance specialization using new_primitive_inst = typed_primitive_inst<new_primitive>; //to simplify usage

    Using template specialization instead of dedicated classes for each primitive comes in hand
    when writing other template methods/classes which would like to use primitive_inst.
    As alternative to this, one could use some kind of type traits to translate primitive type
    to related primitive_inst implementation but this approach does the same with less code/classes.
*/
template <class PType>
class typed_primitive_inst : public typed_primitive_inst_base<PType> {
    static_assert(meta::always_false<PType>::value, "Missing typed_primitive_inst specialization");
};

}  // namespace cldnn
