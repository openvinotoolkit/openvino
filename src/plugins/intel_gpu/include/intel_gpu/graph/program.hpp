// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

#include <list>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <set>

namespace cldnn {

struct topology;
struct program_node;
class layout_optimizer;
class pass_manager;
class base_pass;
class program_wrapper;
class kernels_cache;
class ICompilationContext;


struct program {
    using ptr = std::shared_ptr<program>;
    using cptr = std::shared_ptr<const program>;
    friend class calculate_prior_boxes;             // to be removed when possible
    friend class graph_initializations;             // to be removed when possible
    friend class prepare_padding;                   // to be removed when possible
    friend class propagate_constants;               // to be removed when possible
    friend class pre_replace_deconv;                // to be removed when possible
    friend class prepare_primitive_fusing;          // to be removed when possible
    friend class prepare_quantization;              // to be removed when possible
    friend class reorder_inputs;                    // to be removed when possible
    friend class remove_redundant_reorders;         // to be removed when possible
    friend class post_optimize_weights;             // to be removed when possible
    friend class prepare_primitive_fusing_through;  // to be removed when possible
    friend class reorder_transfer;                  // to be removed when possible
    friend class fuse_constant_transposes;          // to be removed when possible
    friend class program_wrapper;                   // this class is intended to extend the interface of program for
                                                    // the usage within tests_core_internal project only
public:
    struct nodes_ordering {
    public:
        typedef std::list<program_node*> list_of_nodes;
        typedef list_of_nodes::const_iterator const_iterator;
        typedef list_of_nodes::const_reverse_iterator const_reverse_iterator;
        typedef list_of_nodes::iterator node_iterator;
        typedef list_of_nodes::reverse_iterator node_reverse_iterator;
        const_iterator begin() const { return _processing_order.begin(); }
        const_iterator end() const { return _processing_order.end(); }
        const_reverse_iterator rbegin() const { return _processing_order.rbegin(); }
        const_reverse_iterator rend() const { return _processing_order.rend(); }

        void calc_processing_order_visit(program_node* node);
        void calc_processing_order(program& p);
        int32_t get_processing_number(program_node* node) const {
            return get_processing_number(get_processing_iterator(*node));
        }
        int32_t get_processing_number(node_iterator iter) const {
            return 1 + (int32_t)std::distance(_processing_order.begin(), const_iterator(iter));
        }
        void calculate_BFS_processing_order();
        size_t size() { return _processing_order.size(); }
        bool is_correct(program_node* node);

        node_iterator get_processing_iterator(program_node& node) const { return processing_order_iterators.at(&node); }
        void clear() {
            processing_order_iterators.clear();
            _processing_order.clear();
        }

        void insert(program_node* key_node, program_node* node) {
            node_iterator _where = processing_order_iterators.at(key_node);
            processing_order_iterators[node] = _processing_order.insert(_where, node);
        }

        void insert_next(program_node* key_node, program_node* node) {
            node_iterator _where = std::next(processing_order_iterators.at(key_node));
            processing_order_iterators[node] = _processing_order.insert(_where, node);
        }

        void erase(program_node* key_node) {
            node_iterator i = processing_order_iterators.at(key_node);
            processing_order_iterators.erase(key_node);
            _processing_order.erase(i);
        }

        void save(cldnn::BinaryOutputBuffer& ob) const;
        void load(cldnn::BinaryInputBuffer& ib, program& p);

    private:
        list_of_nodes _processing_order;
        std::map<program_node*, node_iterator> processing_order_iterators;
    };

    template <class T>
    struct single_element_container {
        explicit single_element_container(T& t) : elem(&t) {}
        constexpr size_t size() const { return 1; }
        single_element_container begin() const { return single_element_container(elem); }
        single_element_container end() const { return single_element_container(nullptr); }
        single_element_container& operator++() {
            elem = nullptr;
            return *this;
        }
        bool operator!=(single_element_container const& sec) { return elem != sec.elem; }

        T operator*() { return *elem; }

    private:
        explicit single_element_container(T* t) : elem(t) {}

        T* elem;
    };

    typedef std::vector<primitive_info> primitives_info;
    typedef std::vector<std::pair<std::string, primitives_info>> graph_optimizer_info;
    typedef std::pair<primitive_id, std::vector<primitive_id>> optimized_info;

    program(engine& engine_ref,
            topology const& topology,
            const ExecutionConfig& config,
            std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
            std::shared_ptr<ICompilationContext> compilation_context,
            bool is_internal = false,
            bool no_optimizations = false,
            bool is_body_program = false);

    program(engine& engine_ref,
            std::set<std::shared_ptr<program_node>> const& nodes,
            const ExecutionConfig& config,
            std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
            bool is_internal);

    explicit program(engine& engine,
                const ExecutionConfig& config = {});
    ~program();
    engine& get_engine() const { return _engine; }
    const ExecutionConfig& get_config() const { return _config; }
    std::shared_ptr<ov::threading::IStreamsExecutor> get_task_executor() const { return _task_executor; }
    std::list<program_node*>& get_inputs() {
        return inputs;
    }  // ToDo: redesign trim to ouptut pass to make it const as_well as get_engine and get options
    std::vector<program_node*>& get_outputs() {
        return outputs;
    }  // ToDo: redesign reorder-inputs pass to make it const as_well as get_engine and get options
    bool is_body_program() const { return _is_body_program; }
    bool can_be_optimized() const { return _can_be_optimized; }
    bool is_internal_program() const { return is_internal; }
    const nodes_ordering& get_processing_order() const;
    nodes_ordering& get_processing_order();
    const std::vector<primitive_id>& get_allocating_order(bool forced_update = false);
    uint32_t get_prog_id() { return prog_id; }
    stream& get_stream() { return *_stream; }
    stream::ptr get_stream_ptr() const { return _stream; }
    const stream& get_stream() const { return *_stream; }
    const std::list<primitive_id>& get_optimized_out() const { return optimized_out; }
    const std::list<optimized_info>& get_optimized() const { return optimized; }
    bool has_node(const primitive_id& prim) const { return nodes_map.count(prim) > 0; }
    program_node& get_node(primitive_id const& id);
    program_node const& get_node(primitive_id const& id) const;
    std::shared_ptr<program_node> get_node_ptr(const primitive_id& prim) { return nodes_map.at(prim); }
    std::shared_ptr<program_node> get_node_ptr(const primitive_id& prim) const { return nodes_map.at(prim); }
    void set_state_initializers(const std::string& variable_id, const primitive_id& id);
    bool has_state_initializers(const std::string& variable_id, const primitive_id& id);
    bool contains_state(const std::string& variable_id);
    const std::vector<primitive_id>& get_initializers(const std::string& variable_id) { return state_initializers.at(variable_id); }
    const std::map<std::string, std::vector<primitive_id>>& get_state_initializers() const { return state_initializers; }

    // returns already existing program_node for given primitive 'prim' (lookup in 'nodes_map')
    // if it was previously created, otherwise creates and then returns program_node
    program_node& get_or_create(std::shared_ptr<primitive> prim);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency at 'prev_idx' index.
    void add_intermediate(program_node& node,
                          program_node& next,
                          size_t prev_idx,
                          bool connect_int_node_with_old_dep = true,
                          bool move_usrs_of_prev_to_node = false);

    // Gets or creates program_node for given primitive 'prim' and inserts it as an intermediate
    // node between 'next' and it's dependency at 'prev_idx' index.
    void add_intermediate(std::shared_ptr<primitive> prim,
                          program_node& next,
                          size_t prev_idx,
                          bool connect_int_node_with_old_dep = true,
                          bool move_usrs_of_prev_to_node = false);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency prev
    void add_intermediate(program_node& node,
                          program_node& next,
                          program_node& prev,
                          bool connect_int_node_with_old_dep = true,
                          bool move_usrs_of_prev_to_node = false);

    void add_connection(program_node& prev, program_node& next, int32_t port_idx = -1);

    // removes a node from the graph and deletes it afterwards,
    // prereq: node cannot be marked as output and has to have exactly one dependency
    // returns if 'node' has been extracted and removed successfully
    bool extract_and_remove(program_node& node);

    bool extract(program_node& node);

    bool move_node(program_node& node,
                   program_node& new_prev,
                   program_node& new_next);

    // Fuses two nodes into fused_node and removes peer_node from graph
    void fuse_nodes(program_node& fused_node,
                    program_node& peer_node,
                    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>>* fusing_history);

    // returns if 'node' has been removed
    bool remove_if_dangling(program_node& node);

    void mark_if_constant(program_node& node);
    // mark if the node is in data flow assuming that all dependencies are marked properly
    void mark_if_data_flow(program_node& node);
    // Reverses connection - user becomes dependency.

    void remove_nodes(std::vector<program_node*>& to_remove);
    void dump_program(const char* stage, bool with_full_info) const;

    const primitives_info& get_primitives_info() const;
    data_types get_inference_precision(const program_node& node) const;
    std::string get_implementation_info(const primitive_id& id) const;
    const graph_optimizer_info& get_optimizer_passes_info() const;
    void save_pass_info(std::string pass_name);

    void add_optimized_primitive_info(primitive_id optimized_primitive_id, std::vector<primitive_id> replaced_with_ids = {});

    uint32_t get_id() const { return prog_id; }

    static ptr build_program(engine& engine,
                             const topology& topology,
                             const ExecutionConfig& config,
                             bool is_internal = false,
                             bool no_optimizations = false,
                             bool is_body_program = false);
    static ptr build_program(engine& engine,
                             const topology& topology,
                             const ExecutionConfig& config,
                             std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                             bool is_internal = false,
                             bool no_optimizations = false,
                             bool is_body_program = false);
    static ptr build_program(engine& engine,
                             const topology& topology,
                             const ExecutionConfig& config,
                             std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                             std::shared_ptr<ICompilationContext> compilation_context,
                             bool is_internal = false,
                             bool no_optimizations = false,
                             bool is_body_program = false);
    static ptr build_program(engine& engine,
                             const std::set<std::shared_ptr<program_node>>& nodes,
                             const ExecutionConfig& config,
                             std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                             bool is_internal);
    static void init_primitives();
    kernels_cache& get_kernels_cache() const;

    // returns {-1, -1} if it failed to estimate by allocating given batch size
    std::pair<int64_t/*const alloc*/, int64_t/*general alloc*/> get_estimated_device_mem_usage();

    using ImplementationsCache = cldnn::LruCacheThreadSafe<kernel_impl_params, std::shared_ptr<primitive_impl>, kernel_impl_params::Hasher>;

    ImplementationsCache& get_implementations_cache() const { return *_impls_cache; }
    ICompilationContext& get_compilation_context() const { return *_compilation_context; }
    std::shared_ptr<ICompilationContext> get_compilation_context_ptr() const { return _compilation_context; }
    void cancel_compilation_context();

    static std::shared_ptr<ov::threading::IStreamsExecutor> make_task_executor(const ExecutionConfig& config);
    static std::shared_ptr<ICompilationContext> make_compilation_context(const ExecutionConfig& config);

    void save(cldnn::BinaryOutputBuffer& ob) const;
    void load(cldnn::BinaryInputBuffer& ib);
    bool is_loaded_from_cache() const { return _loaded_from_cache; }

    bool is_new_shape_infer() const { return new_shape_infer; }
    layout_optimizer& get_layout_optimizer() const { return *_layout_optimizer; }

private:
    uint32_t prog_id = 0;
    engine& _engine;
    stream::ptr _stream;
    // TODO: Consider moving it to engine
    std::unique_ptr<kernels_cache> _kernels_cache;
    ExecutionConfig _config;
    std::shared_ptr<ov::threading::IStreamsExecutor> _task_executor = nullptr;
    std::list<program_node*> inputs;
    std::vector<program_node*> outputs;
    nodes_ordering processing_order;
    std::vector<primitive_id> allocating_order;
    std::unique_ptr<pass_manager> pm;
    std::unique_ptr<layout_optimizer> _layout_optimizer;
    bool is_internal = false;
    bool _is_body_program = false;
    // if subgraph can be optimized if it consists of only inputs and corresponding outputs
    bool _can_be_optimized;
    std::unique_ptr<ImplementationsCache> _impls_cache;
    std::shared_ptr<ICompilationContext> _compilation_context;
    bool _loaded_from_cache = false;

    bool new_shape_infer = false;

    std::map<primitive_id, std::shared_ptr<program_node>> nodes_map;
    std::list<primitive_id> optimized_out;

    std::list<optimized_info> optimized;
    primitives_info prim_info;
    graph_optimizer_info optimizer_passes_info;

    std::map<std::string, std::vector<primitive_id>> state_initializers;

    primitives_info get_current_stage_info() const;
    /*
    ** High-level functions, in order of usage
    */
    /* build nodes internal structure based on topology */
    void prepare_nodes(topology const& topology);
    /* build nodes internal structure based on the subset of nodes of other program  (used in propagate_constants) */
    void prepare_nodes(std::set<std::shared_ptr<program_node>> const& nodes);
    void add_node_dependencies(program_node* node_ptr);
    void copy_node_dependencies(program_node* dest, program_node* src);
    void build_program(bool is_internal);
    void init_graph();
    void set_options();
    void set_layout_optimizer_attributes(layout_optimizer& lo);

    void apply_opt_pass(base_pass& pass);

    template <class Pass, typename... Args>
    typename std::enable_if<std::is_base_of<base_pass, Pass>::value &&
                            std::is_constructible<Pass, Args...>::value>::type
    apply_opt_pass(Args&&... args) {
        auto pass = Pass(std::forward<Args>(args)...);
        apply_opt_pass(pass);
    }

    void run_graph_compilation();
    void pre_optimize_graph(bool is_internal);
    void post_optimize_graph(bool is_internal);
    void transfer_memory_to_device();

    /*
    ** Analysis functions
    */
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool analyze_output_size_handling_need();

    /*
    ** Optimization functions
    */
    void apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding);

    /*
    ** Memory pool functions
    */
    void prepare_memory_dependencies();
    std::string get_memory_dependencies_string() const;

    /*
    ** Utilities
    */
    void add_split_outputs();
    // mark if the node is constant assuming that all dependencies are marked properly
    void reverse_connection(program_node& dep_node, program_node& user_node);

    void remove_connection(program_node& prev, program_node& next);

    void remove_all_connections(program_node& node);

    void rename(program_node& node, primitive_id const& new_id);
    void swap_names(program_node& node1, program_node& node2);
    void replace_all_usages(program_node& old_node, program_node& new_node, bool remove_if_dangling = true);
    void replace_all_usages(program_node& old_node, std::pair<program_node*, int32_t> new_node, bool remove_if_dangling = true);

    // old_node - node which will be replaced
    // new_node - node which will replace the old one
    void replace(program_node& old_node, program_node& new_node);

    void init_program();
};

}  // namespace cldnn
