// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {

/**
 * @defgroup ov_runtime_myriad_hddl_prop_cpp_api Intel MYRIAD & HDDL specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel MYRIAD & HDDL specific properties.
 *
 * @defgroup ov_runtime_myriad_prop_cpp_api Intel MYRIAD specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel MYRIAD specific properties.
 *
 * @defgroup ov_runtime_hddl_prop_cpp_api Intel HDDL specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel HDDL specific properties.
 */

/**
 * @brief Namespace with Intel MYRIAD specific properties
 */
namespace intel_myriad {

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 * @ingroup ov_runtime_myriad_hddl_prop_cpp_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_hw_acceleration{"MYRIAD_ENABLE_HW_ACCELERATION"};

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * @ingroup ov_runtime_myriad_hddl_prop_cpp_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_receiving_tensor_time{
    "MYRIAD_ENABLE_RECEIVING_TENSOR_TIME"};

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 * @ingroup ov_runtime_myriad_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> custom_layers{"MYRIAD_CUSTOM_LAYERS"};

/**
 * @brief Enum to define possible device protocols
 * @ingroup ov_runtime_myriad_hddl_prop_cpp_api
 */
enum class Protocol {
    PCIE = 0,  //!< Will use a device with PCIE protocol
    USB = 1,   //!< Will use a device with USB protocol
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Protocol& protocol) {
    switch (protocol) {
    case Protocol::PCIE:
        return os << "MYRIAD_PCIE";
    case Protocol::USB:
        return os << "MYRIAD_USB";
    default:
        throw ov::Exception{"Unsupported myriad protocol"};
    }
};
inline std::istream& operator>>(std::istream& is, Protocol& protocol) {
    std::string str;
    is >> str;
    if (str == "MYRIAD_PCIE") {
        protocol = Protocol::PCIE;
    } else if (str == "MYRIAD_USB") {
        protocol = Protocol::USB;
    } else {
        throw ov::Exception{"Unsupported myriad protocol: " + str};
    }
    return is;
};
/** @endcond */

// Myriad specific properties

/**
 * @brief This option allows to specify protocol.
 * @ingroup ov_runtime_myriad_prop_cpp_api
 */
static constexpr Property<Protocol, PropertyMutability::RW> protocol{"MYRIAD_PROTOCOL"};

/**
 * @brief The flag to reset stalled devices.
 * @ingroup ov_runtime_myriad_prop_cpp_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_force_reset{"MYRIAD_ENABLE_FORCE_RESET"};

/**
 * @brief Enum to define possible device mymory types
 * @ingroup ov_runtime_myriad_prop_cpp_api
 */
enum class DDRType {
    MYRIAD_DDR_AUTO = 0,         //!<  Automatic setting of DDR memory type
    MYRIAD_DDR_MICRON_2GB = 1,   //!<  Using a device with MICRON_2GB DDR memory type
    MYRIAD_DDR_SAMSUNG_2GB = 2,  //!<  Using a device with SAMSUNG_2GB DDR memory type
    MYRIAD_DDR_HYNIX_2GB = 3,    //!<  Using a device with HYNIX_2GB DDR memory type
    MYRIAD_DDR_MICRON_1GB = 4,   //!<  Using a device with MICRON_1GB DDR memory type
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const DDRType& ddrType) {
    switch (ddrType) {
    case DDRType::MYRIAD_DDR_AUTO:
        return os << "MYRIAD_DDR_AUTO";
    case DDRType::MYRIAD_DDR_MICRON_2GB:
        return os << "MYRIAD_DDR_MICRON_2GB";
    case DDRType::MYRIAD_DDR_SAMSUNG_2GB:
        return os << "MYRIAD_DDR_SAMSUNG_2GB";
    case DDRType::MYRIAD_DDR_HYNIX_2GB:
        return os << "MYRIAD_DDR_HYNIX_2GB";
    case DDRType::MYRIAD_DDR_MICRON_1GB:
        return os << "MYRIAD_DDR_MICRON_1GB";
    default:
        throw ov::Exception{"Unsupported myriad ddr type"};
    }
};

inline std::istream& operator>>(std::istream& is, DDRType& ddrType) {
    std::string str;
    is >> str;
    if (str == "MYRIAD_DDR_AUTO") {
        ddrType = DDRType::MYRIAD_DDR_AUTO;
    } else if (str == "MYRIAD_DDR_MICRON_2GB") {
        ddrType = DDRType::MYRIAD_DDR_MICRON_2GB;
    } else if (str == "MYRIAD_DDR_SAMSUNG_2GB") {
        ddrType = DDRType::MYRIAD_DDR_SAMSUNG_2GB;
    } else if (str == "MYRIAD_DDR_HYNIX_2GB") {
        ddrType = DDRType::MYRIAD_DDR_HYNIX_2GB;
    } else if (str == "MYRIAD_DDR_MICRON_1GB") {
        ddrType = DDRType::MYRIAD_DDR_MICRON_1GB;
    } else {
        throw ov::Exception{"Unsupported myriad protocol: " + str};
    }
    return is;
};
/** @endcond */

/**
 * @brief This option allows to specify device memory type.
 * @ingroup ov_runtime_myriad_prop_cpp_api
 */
static constexpr Property<DDRType, PropertyMutability::RW> ddr_type{"MYRIAD_DDR_TYPE"};
}  // namespace intel_myriad
}  // namespace ov
