// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {
namespace intel_myriad {
namespace common {

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 */
static constexpr Property<bool, PropertyMutability::RW> enableHwAcceleration{"MYRIAD_ENABLE_HW_ACCELERATION"};

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 */
static constexpr Property<bool, PropertyMutability::RW> enableReceivingTensorTime{
    "MYRIAD_ENABLE_RECEIVING_TENSOR_TIME"};

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
static constexpr Property<std::string, PropertyMutability::RW> customLayers{"MYRIAD_CUSTOM_LAYERS"};

}  // namespace common

enum class Protocol { PCIE = 0, USB };

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
// Myriad specific properties

/**
 * @brief This option allows to specify protocol.
 */
static constexpr Property<Protocol, PropertyMutability::RW> protocol{"MYRIAD_PROTOCOL"};

/**
 * @brief The flag to reset stalled devices.
 */
static constexpr Property<bool, PropertyMutability::RW> enableForceReset{"MYRIAD_ENABLE_FORCE_RESET"};

enum class DDRType {
    MYRIAD_DDR_AUTO = 0,
    MYRIAD_DDR_MICRON_2GB,
    MYRIAD_DDR_SAMSUNG_2GB,
    MYRIAD_DDR_HYNIX_2GB,
    MYRIAD_DDR_MICRON_1GB
};

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

/**
 * @brief This option allows to specify device memory type.
 */

static constexpr Property<DDRType, PropertyMutability::RW> ddrType{"MYRIAD_DDR_TYPE"};

/**
 * @brief Optimize vpu plugin execution to maximize throughput.
 * This option should be used with integer value which is the requested number of streams.
 * The only possible values are:
 *     1
 *     2
 *     3
 */
static constexpr Property<unsigned int, PropertyMutability::RW> myriadThroughputStreams{"MYRIAD_THROUGHPUT_STREAMS"};
/**
 * @brief Default key definition for InferenceEngine::MYRIAD_THROUGHPUT_STREAMS option.
 */
static constexpr Property<unsigned int, PropertyMutability::RW> myriadThroughputStreamsAuto{
    "MYRIAD_THROUGHPUT_STREAMS_AUTO"};

}  // namespace intel_myriad
};  // namespace ov
