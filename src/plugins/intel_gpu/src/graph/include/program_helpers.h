// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "convolution_inst.h"

#include <string>
#include <vector>
#include <utility>
#include <iostream>

namespace cldnn {
struct program_helpers {
    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // It should be used in generic code when there's a need to force vector usage
    // in foreach loop over variable which can in one context be a vector or a scalar
    // in another.
    // example:
    // T t;
    // for (auto& string : wrap_if_single(t.dump()))
    // depending on type T, t.dump() may return either std::string or std::vector<std::string>,
    // to ensure compatibility between these cases, wrap_if_single will create single-element
    // container in case t.dump() would return plain std::string.
    //
    // T& case -> returns container which holds T&
    template <class T>
    static program::single_element_container<T> wrap_if_single(T& t) {
        return program::single_element_container<T>(t);
    }

    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    static program::single_element_container<T const> wrap_if_single(T const& t) {
        return program::single_element_container<T const>(t);
    }

    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    static program::single_element_container<T> wrap_if_single(T&& t) {
        static_assert(meta::always_false<T>::value,
                      "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return program::single_element_container<T>(t);
    }

    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // std::vector case -> does not wrap
    template <typename T>
    static std::vector<T>& wrap_if_single(std::vector<T>& t) {
        return t;
    }

    template <typename T>
    static const std::vector<T>& wrap_if_single(const std::vector<T>& t) {
        return t;
    }

    // helper function for selecting function basing on the type of the given primitive
    // this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    static void do_for_types(program_node&) {
    }

    // helper function for selecting function basing on the type of the given primitive
    // this function should be explicitly given set of types and implicitly set of functions.
    // both sets should have equal size. First function will be called if type of the given primitive
    // will match first explicitly given type, second will be called if it matches second explicitly given
    // type etc.
    // Functions given as arguments should themselves take std::shared_ptr<const T> as argument
    // where T is the type that should be match if this function should be called
    //
    // example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](typed_program_node<convolution>&){ do something if 'primitive' is a convolution },
    //      [](typed_program_node<pooling>&)    { do something if 'primitive' is a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    static decltype(static_cast<void>(std::declval<Func>()(std::declval<typed_program_node<T>&>())))
    do_for_types(program_node& node, Func const& func, RestOfFuncs const&... rest) {
        if (node.type() == T::type_id()) {
            func(node.as<T>());
        } else {
            do_for_types<RestOfT...>(node, rest...);
        }
    }

    // helper functions for deconvolution optimizations
    static void reshape_deconvolution_weights(const std::vector<float> &deconv_weights,
                                              const int channels,
                                              const int kernel_width,
                                              const int kernel_height,
                                              const int scale_factor,
                                              std::vector<std::vector<std::vector<float> > >& subpixel_weights);
    template <typename T>
    static void set_weights_values(T* mem, std::vector<std::vector<std::vector<float> > > args) {
        for (uint32_t x = 0; x < static_cast<uint32_t>(args.size()); ++x) {
            for (uint32_t y = 0; y < static_cast<uint32_t>(args[x].size()); ++y) {
                for (uint32_t z = 0; z < static_cast<uint32_t>(args[x][y].size()); ++z) {
                    *mem = static_cast<T>(args[x][y][z]);
                    mem++;
                }
            }
        }
    }

    // ---- buffer-ownership resolution (F10/F11; reused by F6/F4) ----------------------------------------------
    // A node whose output is a view of its input 0 (can_be_optimized / runtime_skippable), or a slot of an in-place
    // concat, does not own its buffer. settled_only follows only flags already set on the node;
    // include_post_freeze_candidates additionally follows nodes that become views AFTER the oneDNN attribute freeze
    // (add_onednn_optimization_attributes): static no-op resample / strided_slice and any plain reorder that
    // remove_redundant_reorders may still turn into a view or remove. The resolution is node-type based and therefore
    // stable between the freeze and runtime.
    enum class view_resolution { settled_only, include_post_freeze_candidates };
    struct buffer_owner {
        const program_node* owner = nullptr;          // node whose own allocation backs the queried node's output
        int32_t owner_port = 0;                       // output port of owner that backs it (view step: port of the dependency; slot step: 0)
        std::vector<const program_node*> path;        // queried node .. owner, in walk order (only if collect_path)
        bool path_has_multi_output_slot = false;      // some node on the walk has several outputs and is a slot of an optimized concat on any port
        bool path_has_output = false;                 // some node on the walk (start and owner included) is_or_becomes_output
    };
    static bool is_buffer_view_of_input0(const program_node& n, view_resolution mode);
    // the optimized concat user of n (n is one of its slots) or nullptr
    static const program_node* in_place_concat_holder_of(const program_node& n);
    // the optimized concat user that has (n, port) as a dependency, or nullptr
    static const program_node* in_place_concat_holder_of(const program_node& n, int32_t port);
    // the input index whose buffer output `port` of the view node n aliases, or -1 if that port is not a view
    static int32_t view_source_of_port(const program_node& n, int32_t port, view_resolution mode);
    // n's buffer is user-visible, or becomes so before prepare_memory_dependencies: n is an output, or (recursively) a
    // mutable_data user of n is. A superset of the one-level propagation in basic_memory_dependencies.
    static bool is_or_becomes_output(const program_node& n);
    // n's buffer is held outside the network (input_layout replaced by set_data, data, mutable_data, variable buffers,
    // condition and loop whose outputs are re-pointed on every execution) or is constant.
    static bool is_external_buffer_root(const program_node& n);
    // Exact add-fusing queries can run per inference (static can_be_optimized resample users re-bind their sum dst via
    // rebind_onednn_reuse_optimized_dst_if_needed -> get_reused_eltwmem_idx), so the walk is allocation-free unless
    // collect_path == true. path_has_output is always computed. collect_path is kept for F4/F17 diagnostics.
    // start_port: the output port of n the query is about (the port through which a consumer reaches n); the walk is port-aware.
    static buffer_owner resolve_buffer_owner(const program_node& n, view_resolution mode, bool collect_path = false, int32_t start_port = 0);
};

struct onednn_eltwise_fusing_helpers {
    enum class add_fusing_type {
        sum,
        binary_per_tensor,
        binary_per_oc,
        not_supported,
    };

    static void for_eltwise(const program_node& conv_node, eltwise_mode mode,
                            std::function<void(const program_node&, const fused_primitive_desc&)> func);
    // exact: the authoritative predicate S (the freeze, basic_memory_dependencies, get_reused_eltwmem_idx).
    // guard_superset: the pre-F10/F11 predicate, a superset of exact (exact = guard_superset && !F10 && !F11), used by the
    // early consistency guards that run before the freeze (reorder_inputs, layout_optimizer, prepare_buffer_fusing,
    // remove_redundant_reorders). They must answer "may be sum", so they must not use the narrower exact result.
    enum class add_fusing_query { exact, guard_superset };
    static add_fusing_type get_add_fusing_type(const program_node& node, const fused_primitive_desc& desc,
                                               add_fusing_query q = add_fusing_query::exact);
    // Index of the dependency whose buffer dst may alias, or -1. For a static node the fused op's frozen oneDNN post-op type must agree
    // (M8), so the result is "frozen AND live". Binding points must additionally ask primitive_inst::reused_eltw_alias_allowed().
    static int32_t get_reused_eltwmem_idx(const program_node& node);
    // The frozen oneDNN post-op list (add_onednn_optimization_attributes, possibly downgraded by refreeze_onednn_inplace_post_ops) holds an
    // in-place post-op for the fused op whose outer dependency is outer_dep_idx: sum / optimized_sum for mode == sum, binary_mul_inplace for
    // prod. Always false without oneDNN.
    static bool is_frozen_inplace_post_op(const program_node& node, int32_t outer_dep_idx, eltwise_mode mode);
    // F17 M3b/M3c: the buffer of `cur` is, or is viewed by, a network output, or is a slot of a concat that is or may become in-place.
    static bool forward_view_chain_is_visible_or_concat_slot(const program_node& cur);
    // F5: why a frozen in-place sum must become a binary_add. The first matching condition is returned; none means the frozen sum stands.
    enum class sum_downgrade_reason : int {
        none = 0,
        preferred_not_onednn = 1,
        in_shape_of_subgraph = 2,
        impl_not_onednn = 3,
        dep_port_not_zero = 4,
        live_not_sum = 5,
        writer_is_concat_slot_or_visible = 6,
    };
    static sum_downgrade_reason frozen_sum_downgrade_reason(const program_node& node, int32_t mem_dep);
    // F17: the alias eligibility of binary_mul_inplace (dst bound to the post-op src1). Rules M0-M7 are structural queries on the graph;
    // the frozen/impl side (M8 in get_reused_eltwmem_idx, primitive_inst::reused_eltw_alias_allowed) only ever downgrades.
    static bool can_use_mul_inplace(const program_node& node, const fused_primitive_desc& desc);
};

using add_fusing_type = onednn_eltwise_fusing_helpers::add_fusing_type;

static inline std::ostream& operator<< (std::ostream& os, add_fusing_type& t) {
    switch (t) {
        case add_fusing_type::sum: os << "sum"; break;
        case add_fusing_type::binary_per_tensor: os << "binary_per_tensor"; break;
        case add_fusing_type::binary_per_oc: os << "binary_per_oc"; break;
        default: os << "not_supported"; break;
    }
    return os;
}

// Base class for performing pattern match style optimizations.
// Uses CRTP idiom, implementing class should be passed as template parameter `Impl`,
// and overload match and optimize methods.
template <typename Impl>
struct pattern_match_optimization {
    pattern_match_optimization(program& prog)
        : prog(prog)
    {}

    // Returns whether optimization can be performed for specified node.
    bool match(program_node& node) {
        return static_cast<Impl*>(this)->match(node);
    }
    // Returns whether optimization invalidated the node and no futher optimizations should execute.
    bool optimize(program_node& node) {
        // TODO: Add program optimizer class that would take responsibility of modifying program.
        //       Then use it to provide more complex control over pattern-matches, ie:
        //       new node added - run applicable optimizations on it as well;
        //       node deleted - don't do more optimizations;
        return static_cast<Impl*>(this)->optimize(node);
    }
    // Returns whether optimization invalidated the node and no futher optimizations should execute.
    bool match_and_optimize(program_node& node) {
        if (!match(node)) {
            return false;
        }
        return optimize(node);
    }

    program& get_program() { return prog; }

    program& prog;
};

// Class for pattern-match optimizations that provides support for matching
// single primitive type `Prim`.
// Implementing class `Impl` is expected to overload:
// bool match(typed_program_node<Prim>&)
// bool optimize(typed_program_node<Prim>&)
// Uses CRTP idiom, implementing class should be passed as template parameter `Impl`.
template <typename Impl, typename Prim>
struct pattern_match_optimization_typed : pattern_match_optimization<pattern_match_optimization_typed<Impl, Prim>> {
    using base = pattern_match_optimization<pattern_match_optimization_typed<Impl, Prim>>;

    using base::base;

    // Returns whether optimization can be performed for specified node.
    bool match(program_node& node) {
        if (!node.is_type<Prim>()) {
            return false;
        }
        return static_cast<Impl*>(this)->match(node.as<Prim>());
    }
    // Should be overloaded by implementation class to match specified primitive.
    bool match(typed_program_node<Prim>& node) {
        return false;
    }

    // Returns whether optimization invalidated the node and no futher optimizations should execute.
    bool optimize(program_node& node) {
        return static_cast<Impl*>(this)->optimize(node.as<Prim>());
    }
    // Should be overloaded by implementation class to optimize specified primitive.
    bool optimize(typed_program_node<Prim>& node) {
        return false;
    }
};

// Runs pattern-match optimiations passed as arguments on `node`.
inline bool run_node_optimizations(program_node& /*node*/) {
    return false;
}

template <typename Opt, typename... Rest>
bool run_node_optimizations(program_node& node, Opt&& opt, Rest&&... rest) {
    if (opt.match_and_optimize(node)) {
        return true;
    }
    return run_node_optimizations(node, std::forward<Rest>(rest)...);
}

// Runs pattern-match optimizations `Opts` on `node`.
// Optimizations should have constructor with single argument `program&`.
template <typename... Opts>
bool run_node_optimizations(program& p, program_node& node) {
    return run_node_optimizations<Opts...>(node, Opts(p)...);
}

// Runs specified pattern-match optimizations on whole program, in processing order.
template <typename... Opts>
void run_node_optimizations(program& p, Opts&&... opts) {
    auto it = p.get_processing_order().begin();
    while (it != p.get_processing_order().end()) {
        auto* node = *it++;
        run_node_optimizations(*node, opts...);
    }
}

template <typename... Opts>
void run_node_optimizations(program& p) {
    run_node_optimizations(p, Opts(p)...);
}

}  // namespace cldnn
