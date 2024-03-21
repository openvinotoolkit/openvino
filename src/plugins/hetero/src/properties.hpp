// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace hetero {
/**
 * @brief Read-only property to get device caching properties
 */
static constexpr Property<std::string, PropertyMutability::RO> caching_device_properties{"CACHING_DEVICE_PROPERTIES"};

/**
 * @brief Read-only property showing number of compiled submodels
 */
static constexpr Property<size_t, PropertyMutability::RO> number_of_submodels{"HETERO_NUMBER_OF_SUBMODELS"};

/**
 * @brief Enum to define possible parallel policies
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class ParallelPolicy {
    AUTO_SPLIT = 1,
    MEMORY_FIRST = 2,
    MEMORY_RATIO = 3,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ParallelPolicy& parallel_policy) {
    switch (parallel_policy) {
    case ParallelPolicy::AUTO_SPLIT:
        return os << "AUTO_SPLIT";
    case ParallelPolicy::MEMORY_FIRST:
        return os << "MEMORY_FIRST";
    case ParallelPolicy::MEMORY_RATIO:
        return os << "MEMORY_RATIO";
    default:
        OPENVINO_THROW("Unsupported performance mode hint");
    }
}

inline std::istream& operator>>(std::istream& is, ParallelPolicy& parallel_policy) {
    std::string str;
    is >> str;
    if (str == "AUTO_SPLIT") {
        parallel_policy = ParallelPolicy::AUTO_SPLIT;
    } else if (str == "MEMORY_FIRST") {
        parallel_policy = ParallelPolicy::MEMORY_FIRST;
    } else if (str == "MEMORY_RATIO") {
        parallel_policy = ParallelPolicy::MEMORY_RATIO;
    } else {
        OPENVINO_THROW("Unsupported performance mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Read-write property setting the policy of split model
 */
static constexpr Property<ParallelPolicy, PropertyMutability::RW> parallel_policy{"PARALLEL_POLICY"};
}  // namespace hetero
}  // namespace ov
