/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "program_node.h"
#include "engine_impl.h"
#include "program_impl.h"
#include <string>
#include <vector>
#include <utility>

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
    static program_impl::single_element_container<T> wrap_if_single(T& t) {
        return program_impl::single_element_container<T>(t);
    }

    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    static program_impl::single_element_container<T const> wrap_if_single(T const& t) {
        return program_impl::single_element_container<T const>(t);
    }

    // helper function which creates single-element array if it's given anything
    // other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    static program_impl::single_element_container<T> wrap_if_single(T&& t) {
        static_assert(meta::always_false<T>::value,
                      "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return program_impl::single_element_container<T>(t);
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
    static void merge_buffers(engine_impl& engine,
                              program_node& node,
                              const layout& target_layout,
                              size_t begin_offset,
                              size_t end_offset);

    static std::pair<bool, bool> are_layouts_identical(layout const& l1, layout const& l2);

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
    static layout get_weights_layout(typed_program_node<cldnn::data>& data_node, int32_t split);
};
}  // namespace cldnn
