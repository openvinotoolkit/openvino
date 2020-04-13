// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <layer_transform.hpp>

#include <utility>
#include <tuple>


namespace InferenceEngine {

namespace details {

/**
 * @brief checks whether type index as P has a parent among element in range I..N
 * can be used only for P < I
 * */
template <size_t P, size_t I, class Tuple, class Enable = void>
struct is_base_of_any;

// clang-format off
template <size_t IBase, size_t IDerived, class Tuple>
struct is_base_of_any< IBase, IDerived, Tuple,
    typename std::enable_if<IBase < std::tuple_size<Tuple>::value, void>::type > : public std::true_type {
         using base = typename std::remove_pointer<typename std::tuple_element<IBase, Tuple>::type>::type;
         using derived = typename std::remove_pointer<typename std::tuple_element<IDerived, Tuple>::type>::type;

    static_assert(IDerived < IBase, "cannot match parent using incorrect indices");
    static_assert(!std::is_base_of<derived, base>::value, "probing type is a parent of followed type");

    // check that incoming type have parents in range I..N, and any of I..N not a child of derived type
    static_assert((std::is_base_of<base, derived>::value || is_base_of_any<IBase + 1, IDerived, Tuple>::value), "parent matching failed");
};
// clang-format on

// for matches any->after last
template <size_t IBase, size_t IDerived, class Tuple>
struct is_base_of_any<IBase, IDerived, Tuple,
                      typename std::enable_if<IBase >= std::tuple_size<Tuple>::value, void>::type>
    : public std::false_type {};

/**
 * @brief check whether type ordered from child to base within given list
 */
template <size_t P, class Tuple, class Enable = void>
struct is_types_ordered_from_child_to_base {};

template <size_t P, class Tuple>
struct is_types_ordered_from_child_to_base<
    P, Tuple, typename std::enable_if<P != std::tuple_size<Tuple>::value - 2, void>::type> {
    static constexpr bool value =
        is_base_of_any<P + 1, P, Tuple>::value && is_types_ordered_from_child_to_base<P + 1, Tuple>::value;
};

template <size_t P, class Tuple>
struct is_types_ordered_from_child_to_base<
    P, Tuple, typename std::enable_if<P == std::tuple_size<Tuple>::value - 2, void>::type> {
    static constexpr bool value = is_base_of_any<P + 1, P, Tuple>::value;
};

static_assert(
    is_types_ordered_from_child_to_base<0, AllLayers>::value,
    "All layers must be topologically sorted as so for any layer, it's father appeared later in a types list");

}  // namespace details

}  // namespace InferenceEngine
