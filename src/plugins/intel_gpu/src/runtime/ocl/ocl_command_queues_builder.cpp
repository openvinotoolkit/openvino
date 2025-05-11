// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_command_queues_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <string>

namespace cldnn {
namespace ocl {

command_queues_builder::command_queues_builder()
    : _profiling(false),
      _out_of_order(false),
      _supports_queue_families(false),
      _priority_mode(),
      _throttle_mode() {}

#if CL_TARGET_OPENCL_VERSION >= 200
std::vector<cl_queue_properties> command_queues_builder::get_properties(const cl::Device& device, uint16_t stream_id) {
    std::vector<cl_queue_properties> properties;

    if (_priority_mode.has_value()) {
        unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;
        switch (_priority_mode.value()) {
            case ov::hint::Priority::HIGH:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
                break;
            case ov::hint::Priority::LOW:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
                break;
            default:
                break;
        }

        properties.insert(properties.end(), {CL_QUEUE_PRIORITY_KHR, cl_queue_priority_value});
    }

    if (_throttle_mode.has_value()) {
        unsigned cl_queue_throttle_value = CL_QUEUE_THROTTLE_MED_KHR;
        switch (_throttle_mode.value()) {
            case ov::intel_gpu::hint::ThrottleLevel::HIGH:
                cl_queue_throttle_value = CL_QUEUE_THROTTLE_HIGH_KHR;
                break;
            case ov::intel_gpu::hint::ThrottleLevel::LOW:
                cl_queue_throttle_value = CL_QUEUE_THROTTLE_LOW_KHR;
                break;
            default:
                break;
        }

        properties.insert(properties.end(), {CL_QUEUE_THROTTLE_KHR, cl_queue_throttle_value});
    }

    if (_supports_queue_families) {
        cl_uint num_queues = 0;
        cl_uint family = 0;

        std::vector<cl_queue_family_properties_intel> qfprops = device.getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>();
        for (cl_uint q = 0; q < qfprops.size(); q++) {
            if (qfprops[q].capabilities == CL_QUEUE_DEFAULT_CAPABILITIES_INTEL && qfprops[q].count > num_queues) {
                family = q;
                num_queues = qfprops[q].count;
            }
        }

        if (num_queues)
            properties.insert(properties.end(), {CL_QUEUE_FAMILY_INTEL, family,
                                                 CL_QUEUE_INDEX_INTEL, stream_id % num_queues});
    }

    bool out_of_order = _out_of_order;
    if (_out_of_order) {
        auto queue_properties = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        using cmp_t = std::common_type<decltype(queue_properties), typename std::underlying_type<cl::QueueProperties>::type>::type;
        if (!(static_cast<cmp_t>(queue_properties) & static_cast<cmp_t>(cl::QueueProperties::OutOfOrder))) {
            out_of_order = false;
            GPU_DEBUG_INFO << "Requested out-of-order queue is not supported by current device. Use in-order instead\n";
        }
    }

    cl_command_queue_properties cl_queue_properties =
        ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));

    properties.insert(properties.end(), {CL_QUEUE_PROPERTIES, cl_queue_properties, 0});

    return properties;
}
#else
cl_command_queue_properties command_queues_builder::get_properties(const cl::Device& device, uint16_t stream_id) {
    cl_command_queue_properties cl_queue_properties =
        ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (_out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));

    return cl_queue_properties;
}
#endif

ocl_queue_type command_queues_builder::build(const cl::Context& context, const cl::Device& device) {
    ocl_queue_type queue;
    cl_int error_code = CL_SUCCESS;
    static std::atomic<uint16_t> stream_id{0};

    auto properties = get_properties(device, stream_id++);
#if CL_TARGET_OPENCL_VERSION >= 200
    queue = clCreateCommandQueueWithProperties(context.get(), device.get(), properties.data(), &error_code);
#else
    queue = clCreateCommandQueue(context.get(), device.get(), properties, &error_code);
#endif
    OPENVINO_ASSERT(error_code == CL_SUCCESS, "[GPU] Command queues builder returned ", error_code, " error code");
    return queue;
}

void command_queues_builder::set_priority_mode(ov::hint::Priority priority, bool extension_support) {
    if (extension_support) {
        _priority_mode = priority;
    }
}

void command_queues_builder::set_throttle_mode(ov::intel_gpu::hint::ThrottleLevel throttle, bool extension_support) {
    if (extension_support) {
        _throttle_mode = throttle;
    }
}

void command_queues_builder::set_supports_queue_families(bool extension_support) {
    _supports_queue_families = extension_support;
}
}  // namespace ocl
}  // namespace cldnn
