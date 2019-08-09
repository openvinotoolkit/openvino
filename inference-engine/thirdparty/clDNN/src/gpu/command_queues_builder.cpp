/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "command_queues_builder.h"
#include "error_handler.h"
#include <string>

namespace cldnn {
namespace gpu {

command_queues_builder::command_queues_builder(const cl::Context& context,
                                               const cl::Device& device,
                                               const cl_platform_id& platform_id)
    : _context(context),
      _device(device),
      _platform_id(platform_id),
      _priority_mode(cldnn_priority_disabled),
      _throttle_mode(cldnn_throttle_disabled) {}

cl_command_queue_properties command_queues_builder::get_properties() {
    cl_command_queue_properties ret =
        ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (_out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));
    return ret;
}

void command_queues_builder::build() {
    auto properties = get_properties();

    if (_priority_mode == cldnn_priority_disabled && _throttle_mode == cldnn_throttle_disabled) {
        _queue = cl::CommandQueue(_context, _device, properties);
        return;
    }

    unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;

    switch (_priority_mode) {
        case cldnn_priority_high:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
            break;
        case cldnn_priority_low:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
            break;
        default:
            break;
    }

    unsigned cl_queue_throttle_value = CL_QUEUE_THROTTLE_MED_KHR;

    switch (_throttle_mode) {
        case cldnn_throttle_high:
            cl_queue_throttle_value = CL_QUEUE_THROTTLE_HIGH_KHR;
            break;
        case cldnn_throttle_low:
            cl_queue_throttle_value = CL_QUEUE_THROTTLE_LOW_KHR;
            break;
        default:
            break;
    }

    cl_int error_code = CL_SUCCESS;

    if (_priority_mode != cldnn_priority_disabled && _throttle_mode != cldnn_throttle_disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_PRIORITY_KHR,
                                                cl_queue_priority_value,
                                                CL_QUEUE_THROTTLE_KHR,
                                                cl_queue_throttle_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        _queue = clCreateCommandQueueWithProperties(_context.get(), _device.get(), properties_low, &error_code);
    } else if (_priority_mode != cldnn_priority_disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_PRIORITY_KHR,
                                                cl_queue_priority_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        _queue = clCreateCommandQueueWithProperties(_context.get(), _device.get(), properties_low, &error_code);
    } else if (_throttle_mode != cldnn_throttle_disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_THROTTLE_KHR,
                                                cl_queue_throttle_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        _queue = clCreateCommandQueueWithProperties(_context.get(), _device.get(), properties_low, &error_code);
    }

    if (error_code != CL_SUCCESS) {
        CLDNN_ERROR_MESSAGE("Command queues builders",
                            "clCreateCommandQueueWithPropertiesINTEL error " + std::to_string(error_code));
    }
}

void command_queues_builder::set_priority_mode(cldnn_priority_mode_type priority, bool extension_support) {
    if (priority != cldnn_priority_disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - priority_mode",
                            std::string("The param priority_mode is set in engine_configuration, ")
                            .append("but cl_khr_priority_hints or cl_khr_create_command_queue ")
                            .append("is not supported by current OpenCL implementation."));
    }
    _priority_mode = priority;
}

void command_queues_builder::set_throttle_mode(cldnn_throttle_mode_type throttle, bool extension_support) {
    if (throttle != cldnn_throttle_disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - throttle_mode",
                            std::string("The param throttle_mode is set in engine_configuration, ")
                            .append("but cl_khr_throttle_hints is not supported by current OpenCL implementation."));
    }
    _throttle_mode = throttle;
}
}  // namespace gpu
}  // namespace cldnn
