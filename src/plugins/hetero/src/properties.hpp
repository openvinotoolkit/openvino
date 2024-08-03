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
 * @brief Enum to define possible schedule policy
 */
enum class SchedulePolicy {
    EQUAL_TO_DEVICES = 1,                 //!<  The model will be split by the ratio of device memory
    PRIORITY_FIRST = 2,                   //!<  The model will be split by the sequence of device priority
};


inline std::ostream& operator<<(std::ostream& os, const SchedulePolicy& schedule_policy) {
    switch (schedule_policy) {
    case SchedulePolicy::EQUAL_TO_DEVICES:
        return os << "EQUAL_TO_DEVICES";
    case SchedulePolicy::PRIORITY_FIRST:
        return os << "PRIORITY_FIRST";
    default:
        OPENVINO_THROW("Unsupported schedule policy");
    }
}

inline std::istream& operator>>(std::istream& is, SchedulePolicy& schedule_policy) {
    std::string str;
    is >> str;
    if (str == "EQUAL_TO_DEVICES") {
        schedule_policy = SchedulePolicy::EQUAL_TO_DEVICES;
    } else if (str == "PRIORITY_FIRST") {
        schedule_policy = SchedulePolicy::PRIORITY_FIRST;
    } else {
        OPENVINO_THROW("Unsupported schedule policy: ", str);
    }
    return is;
}
/** @endcond */
/**
 * @brief Property to set the schedule policy of pipeline parallel
 */
static constexpr Property<SchedulePolicy> schedule_policy{"SCHEDULE_POLICY"};
}  // namespace hetero
}  // namespace ov
