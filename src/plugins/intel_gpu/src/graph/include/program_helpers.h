// Copyright (C) 2018-2025 Intel Corporation
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
        return;
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
        if (node.type() == T::type_id())
            func(node.as<T>());
        else
            do_for_types<RestOfT...>(node, rest...);
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
};

struct onednn_add_fusing_helpers {
    enum class add_fusing_type {
        sum,
        binary_per_tensor,
        binary_per_oc,
        not_supported,
    };

    static bool is_full_tensor(const layout& layout);
    static std::vector<fused_primitive_desc> get_fused_eltwise_primitives();
    static void for_eltwise(const program_node& conv_node, eltwise_mode mode,
                            std::function<void(const program_node&, const fused_primitive_desc&)> func);
    static add_fusing_type get_add_fusing_type(const program_node& node, const fused_primitive_desc& desc);
};

using add_fusing_type = onednn_add_fusing_helpers::add_fusing_type;

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
        if (!match(node))
            return false;
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
        if (!node.is_type<Prim>())
            return false;
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
    if (opt.match_and_optimize(node))
        return true;
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
        auto node = *it++;
        run_node_optimizations(*node, opts...);
    }
}

template <typename... Opts>
void run_node_optimizations(program& p) {
    run_node_optimizations(p, Opts(p)...);
}

}  // namespace cldnn
