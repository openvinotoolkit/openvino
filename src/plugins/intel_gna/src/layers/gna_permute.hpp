// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <list>
#include <utility>
#include <vector>

#include "ie_common.h"
#include "log/debug.hpp"

namespace ov {
namespace intel_gna {
namespace permute {

template <class T>
class PermuteSequence {
public:
    using cnt_type = std::vector<std::pair<T, T>>;

private:
    std::vector<T> orderVec;
    cnt_type permutes;

public:
    explicit PermuteSequence(std::vector<T>&& orderVecIn) : orderVec(std::move(orderVecIn)) {
        std::vector<bool> counter(orderVec.size());
        for (auto&& x : this->orderVec) {
            if (x < 0) {
                THROW_GNA_EXCEPTION << "invalid order: element " << x << " should be >= 0";
            }
            if (static_cast<size_t>(x) >= counter.size()) {
                THROW_GNA_EXCEPTION << "invalid order: element " << x << " should be < " << counter.size();
            }
            if (counter[x]) {
                THROW_GNA_EXCEPTION << "invalid order: element " << x << " present more than once";
            }
            counter[x] = true;
        }

        // generating permutation graph
        std::fill(counter.begin(), counter.end(), false);

        // length of current cycle
        std::list<cnt_type> permuteCycles;
        bool newSeq = false;

        for (int i = 0; i != static_cast<int>(orderVec.size());) {
            // we have this permutation on the list already
            if (counter[i]) {
                newSeq = false;
                i++;
                continue;
            }
            counter[i] = true;
            // looks we found a permutation
            if (orderVec[i] != i) {
                if (!newSeq) {
                    newSeq = true;
                    permuteCycles.push_back({});
                }
                permuteCycles.back().push_back({i, orderVec[i]});
                counter[i] = true;
                i = static_cast<int>(orderVec[i]);
                continue;
            }
            // this dims not permuted
            i++;
        }

        for (auto&& cycle : permuteCycles) {
            for (size_t i = 0; i + 1 < cycle.size(); i++) {
                permutes.push_back(cycle[i]);
            }
        }
    }
    const cnt_type& cnt() const noexcept {
        return permutes;
    }
};

/**
 * @brief generates permutations sequence in order to reach given order
 * @tparam Iterator
 * @return
 */
template <class Iterator>
inline typename PermuteSequence<typename std::iterator_traits<Iterator>::value_type>::cnt_type genPermutations(
    Iterator beg,
    Iterator en) {
    static_assert(std::is_same<std::random_access_iterator_tag,
                               typename std::iterator_traits<Iterator>::iterator_category>::value,
                  "The genPermutations() function only accepts random access iterators or raw pointers to an array.\n");
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> v;
    for (; beg != en; beg++) {
        v.push_back(*beg);
    }
    auto permute = PermuteSequence<value_type>(std::move(v));
    return permute.cnt();
}

template <class T>
inline typename PermuteSequence<T>::cnt_type genPermutations(const std::initializer_list<T>& lst) {
    return genPermutations(lst.begin(), lst.end());
}

/**
 * @brief returns dimensions order for permute from input to output layout
 * @param in_layout
 * @param out_layout
 */
inline std::vector<int> GetPermuteOrder(InferenceEngine::Layout in_layout, InferenceEngine::Layout out_layout) {
    if (in_layout == InferenceEngine::Layout::NHWC && out_layout == InferenceEngine::Layout::NCHW) {
        return {0, 3, 1, 2};
    }
    if (in_layout == InferenceEngine::Layout::NCHW && out_layout == InferenceEngine::Layout::NHWC) {
        return {0, 2, 3, 1};
    }
    return {0, 1, 2, 3};
}

inline bool isTrivialPermute(const std::vector<int64_t> order, const std::vector<size_t>& input_shape) {
    // cases when all permutations happened either between 1 and X shape where no other dims in between
    auto transpose_seq = genPermutations(order.begin(), order.end());
    auto input_order_transformed = input_shape;
    for (auto&& transp : transpose_seq) {
        // check dims of transposed
        if (input_order_transformed[transp.first] == 1 && input_order_transformed[transp.second] == 1) {
            return true;
        }
        if (input_order_transformed[transp.first] != 1 && input_order_transformed[transp.second] != 1) {
            return false;
        }
        // check dims in between
        for (int64_t j = std::min(transp.first, transp.second) + 1; j < std::max(transp.first, transp.second); j++) {
            if (input_order_transformed[j] != 1) {
                return false;
            }
        }
        // apply permutation
        std::swap(input_order_transformed[transp.first], input_order_transformed[transp.second]);
    }
    return true;
}

}  // namespace permute
}  // namespace intel_gna
}  // namespace ov
