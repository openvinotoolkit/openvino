// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits.h>
#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"

namespace intel_npu {

namespace zeroUtils {

static inline void throwOnFail(const std::string& step, const ze_result_t result) {
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 ",
                       step,
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       ze_result_to_description(result));
    }
}

static inline void throwOnFail(const std::string& step, const ze_result_t result, const std::string& hintOnError) {
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 ",
                       step,
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       ze_result_to_description(result),
                       ". ",
                       hintOnError);
    }
}

static inline ze_command_queue_priority_t toZeQueuePriority(const ov::hint::Priority& val) {
    switch (val) {
    case ov::hint::Priority::LOW:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    case ov::hint::Priority::MEDIUM:
        return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    case ov::hint::Priority::HIGH:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    default:
        OPENVINO_THROW("Incorrect queue priority.");
    }
}

static inline std::size_t precisionToSize(const ze_graph_argument_precision_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return 8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return 8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return 16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return 16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return 32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return 32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return 64;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return 64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return 16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return 16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return 32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return 64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return 1;
    default:
        OPENVINO_THROW("precisionToSize switch->default reached");
    }
}

static inline ze_graph_argument_precision_t getZePrecision(const ov::element::Type_t precision) {
    switch (precision) {
    case ov::element::Type_t::i4:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT4;
    case ov::element::Type_t::u4:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT4;
    case ov::element::Type_t::i8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case ov::element::Type_t::u8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case ov::element::Type_t::i16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case ov::element::Type_t::u16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case ov::element::Type_t::i32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case ov::element::Type_t::u32:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT32;
    case ov::element::Type_t::i64:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT64;
    case ov::element::Type_t::u64:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT64;
    case ov::element::Type_t::bf16:
        return ZE_GRAPH_ARGUMENT_PRECISION_BF16;
    case ov::element::Type_t::f16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case ov::element::Type_t::f32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case ov::element::Type_t::f64:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP64;
    case ov::element::Type_t::u1:
        return ZE_GRAPH_ARGUMENT_PRECISION_BIN;
    default:
        return ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
    }
}

static inline std::size_t layoutCount(const ze_graph_argument_layout_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return 1;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return 3;
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_ANY:
        // When input has empty shape, val is ZE_GRAPH_ARGUMENT_LAYOUT_ANY
        // Add this to pass Single Layer Test on Windows
        return 0;
    default:
        OPENVINO_THROW("layoutCount switch->default reached");
    }
}

static inline std::size_t getSizeIOBytes(const ze_graph_argument_properties_3_t& argument) {
    std::size_t num_elements = 1;
    for (std::size_t i = 0; i < layoutCount(argument.deviceLayout); ++i) {
        num_elements *= argument.dims[i];
    }
    const std::size_t size_in_bits = num_elements * precisionToSize(argument.devicePrecision);
    const std::size_t size_in_bytes = (size_in_bits + (CHAR_BIT - 1)) / CHAR_BIT;
    return size_in_bytes;
}

static inline uint32_t findGroupOrdinal(
    const std::vector<ze_command_queue_group_properties_t>& command_group_properties,
    const ze_device_properties_t& properties) {
    auto log = Logger::global().clone("findGroupOrdinal");

    if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
        for (uint32_t index = 0; index < command_group_properties.size(); ++index) {
            const auto& flags = command_group_properties[index].flags;
            if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0 &&
                (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) == 0) {
                return index;
            }
        }

        // if we don't find a group where only the proper flag is enabled then search for a group where that flag is
        // enabled
        for (uint32_t index = 0; index < command_group_properties.size(); ++index) {
            const auto& flags = command_group_properties[index].flags;
            if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                return index;
            }
        }

        // if still don't find compute flag, return a warning
        log.warning("Fail to find a command queue group that contains compute flag, it will be set to 0.");
        return 0;
    }

    for (uint32_t index = 0; index < command_group_properties.size(); ++index) {
        const auto& flags = command_group_properties[index].flags;
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0 &&
            (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0) {
            return index;
        }
    }

    // if still don't find compute and copy flag, return a warning
    log.warning("Fail to find a command queue group that contains compute and copy flags, it will be set to 0.");
    return 0;
}

}  // namespace zeroUtils
}  // namespace intel_npu
