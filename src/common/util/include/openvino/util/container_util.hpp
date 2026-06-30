// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>

namespace ov::util {

/**
 * @brief Checks if container contains the specific value.
 *
 * @param container  The container of elements to examine.
 * @param value      Value to compare the elements to.
 * @return True if value found in the container, false otherwise.
 */
template <typename R, typename V>
constexpr bool contains(const R& container, const V& value) {
    return std::find(std::begin(container), std::end(container), value) != std::end(container);
}

/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename Container>
auto product(const Container& container) {
    using T = typename Container::value_type;
    return container.empty() ? T{0} : std::accumulate(container.begin(), container.end(), T{1}, std::multiplies<T>());
}

/**
 * @brief Removes elements from the container that satisfy the given predicate.
 *
 * @param data      The container from which to remove elements.
 * @param predicate A callable that check element and return true if the element should be removed, or false otherwise.
 */
template <typename Container, typename PredicateT>
inline void erase_if(Container& data, const PredicateT& predicate) {
    for (auto it = std::begin(data); it != std::end(data);) {
        if (predicate(*it)) {
            it = data.erase(it);
        } else {
            ++it;
        }
    }
}

/**
 * @brief Creates a std::array from a parameter pack.
 * @tparam T  Optional explicit element type; deduced from args if void.
 * @param args  Elements to store in the array.
 * @return std::array holding the provided arguments.
 */
template <class T = void, class... Args>
constexpr std::array<std::conditional_t<std::is_void_v<T>, std::common_type_t<Args...>, T>, sizeof...(Args)> make_array(
    Args&&... args) {
    return {std::forward<Args>(args)...};
}

}  // namespace ov::util
