// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <functional>
#include <algorithm>
#include <numeric>

namespace InferenceEngine {
namespace details {
/**
 * @rationale - associative containers doesnt work with remove_if algorithm
 * @tparam ContainerT
 * @tparam PredicateT
 * @param items
 * @param predicate
 */
template<typename Container, typename PredicateT>
inline void erase_if(Container &data, const PredicateT &predicate) {
    for (auto it = std::begin(data); it != std::end(data);) {
        if (predicate(*it)) {
            it = data.erase(it);
        } else {
            ++it;
        }
    }
}
/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */

template<typename TIterator>
auto product(TIterator beg, TIterator en) -> typename std::remove_reference<decltype(*beg)>::type {
    return std::accumulate(beg, en,
                           static_cast<typename std::remove_reference<decltype(*beg)>::type>(1),
                           std::multiplies<typename std::remove_reference<decltype(*beg)>::type>());
}
}  // namespace details
}  // namespace InferenceEngine
