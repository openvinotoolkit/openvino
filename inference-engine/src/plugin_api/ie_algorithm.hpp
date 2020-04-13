// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file with simple helper functions for STL containters
 * @file ie_algorithm.hpp
 */

#pragma once
#include <algorithm>
#include <functional>
#include <numeric>

namespace InferenceEngine {

/**
 * @brief A namespace with non-public Inference Engine Plugin API
 * @ingroup ie_dev_api
 */
namespace details {

/**
 * @brief Simple helper function to check element presence in container
 * container must provede stl-compliant find member function
 *
 * @param container - Container to check
 * @param element - element to check
 *
 * @return true if element present in container
 */
template <typename C, typename T>
bool contains(const C& container, const T& element) {
    return container.find(element) != container.end();
}

/**
 * @brief Associative containers doesnt work with remove_if algorithm
 * @tparam ContainerT
 * @tparam PredicateT
 * @param data An associative container 
 * @param predicate A predicate to remove values conditionally
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
 * @brief      Multiplies container
 *
 * @param[in]  beg        The `begin` iterator
 * @param[in]  en         The `end` iterator
 *
 * @tparam     TIterator  An iterator type
 *
 * @return     A result of multiplication.
 */
template <typename TIterator>
auto product(TIterator beg, TIterator en) -> typename std::remove_reference<decltype(*beg)>::type {
    return std::accumulate(beg, en, static_cast<typename std::remove_reference<decltype(*beg)>::type>(1),
                           std::multiplies<typename std::remove_reference<decltype(*beg)>::type>());
}

/**
 * @brief      Clips element to be in range `[min, max]`
 *
 * @param      idx   The pointer to element.
 * @param[in]  min   The minimum value
 * @param[in]  max   The maximum value
 */
inline void clipping(int* idx, const int min, const int max) {
    (*idx) = ((*idx) > min) ? (*idx) : min;
    (*idx) = ((*idx) < max) ? (*idx) : (max - 1);
}

}  // namespace details
}  // namespace InferenceEngine
