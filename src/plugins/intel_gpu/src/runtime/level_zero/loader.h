/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef COMPUTE_SAMPLES_OCL_ENTRYPOINTS_LOADER_HPP
#define COMPUTE_SAMPLES_OCL_ENTRYPOINTS_LOADER_HPP

#include <CL/cl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace compute_samples {
template <typename T>
T load_entrypoint(const cl_platform_id platform, const std::string name) {
  T p = reinterpret_cast<T>(
      clGetExtensionFunctionAddressForPlatform(platform, name.c_str()));
  if (!p) {
    throw std::runtime_error("clGetExtensionFunctionAddressForPlatform(" +
                             name + ") returned NULL.");
  }
  return p;
}

template <typename T>
T load_entrypoint(const cl_device_id device, const std::string name) {
  cl_platform_id platform;
  cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                 &platform, nullptr);
  if (error) {
    throw std::runtime_error("Failed to retrieve CL_DEVICE_PLATFORM: " +
                             std::to_string(error));
  }
  return load_entrypoint<T>(platform, name);
}

template <typename T>
T load_entrypoint(const cl_context context, const std::string name) {
  size_t size;
  cl_int error =
      clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);
  if (error) {
    throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES size: " +
                             std::to_string(error));
  }

  std::vector<cl_device_id> devices(size / sizeof(cl_device_id));

  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices.data(),
                           nullptr);
  if (error) {
    throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES: " +
                             std::to_string(error));
  }

  return load_entrypoint<T>(devices.front(), name);
}

template <typename T>
T load_entrypoint(const cl_kernel kernel, const std::string name) {
  cl_context context;
  cl_int error = clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(context),
                                 &context, nullptr);
  if (error) {
    throw std::runtime_error("Failed to retrieve CL_KERNEL_CONTEXT: " +
                             std::to_string(error));
  }
  return load_entrypoint<T>(context, name);
}

template <typename T>
T load_entrypoint(const cl_command_queue queue, const std::string name) {
  cl_context context;
  cl_int error = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context),
                                       &context, nullptr);
  if (error) {
    throw std::runtime_error("Failed to retrieve CL_QUEUE_CONTEXT: " +
                             std::to_string(error));
  }
  return load_entrypoint<T>(context, name);
}
} // namespace compute_samples

#endif
