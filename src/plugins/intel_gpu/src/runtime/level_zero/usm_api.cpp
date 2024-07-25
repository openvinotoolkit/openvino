/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <CL/cl_ext.h>
#include "loader.h"

namespace cs = compute_samples;

CL_API_ENTRY void *CL_API_CALL clHostMemAllocINTEL(
    cl_context context, const cl_mem_properties_intel *properties, size_t size,
    cl_uint alignment, cl_int *errcode_ret) {
  const auto e = cs::load_entrypoint<clHostMemAllocINTEL_fn>(
      context, "clHostMemAllocINTEL");
  return e(context, properties, size, alignment, errcode_ret);
}

CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      const cl_mem_properties_intel *properties, size_t size,
                      cl_uint alignment, cl_int *errcode_ret) {
  const auto e = cs::load_entrypoint<clDeviceMemAllocINTEL_fn>(
      context, "clDeviceMemAllocINTEL");
  return e(context, device, properties, size, alignment, errcode_ret);
}

CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      const cl_mem_properties_intel *properties, size_t size,
                      cl_uint alignment, cl_int *errcode_ret) {
  const auto e = cs::load_entrypoint<clSharedMemAllocINTEL_fn>(
      context, "clSharedMemAllocINTEL");
  return e(context, device, properties, size, alignment, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(cl_context context, void *ptr) {
  const auto e =
      cs::load_entrypoint<clMemFreeINTEL_fn>(context, "clMemFreeINTEL");
  return e(context, ptr);
}

CL_API_ENTRY cl_int CL_API_CALL clMemBlockingFreeINTEL(cl_context context,
                                                       void *ptr) {
  const auto e =
      cs::load_entrypoint<clMemFreeINTEL_fn>(context, "clMemBlockingFreeINTEL");
  return e(context, ptr);
}

CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  const auto e = cs::load_entrypoint<clGetMemAllocInfoINTEL_fn>(
      context, "clGetMemAllocInfoINTEL");
  return e(context, ptr, param_name, param_value_size, param_value,
           param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
  const auto e = cs::load_entrypoint<clSetKernelArgMemPointerINTEL_fn>(
      kernel, "clSetKernelArgMemPointerINTEL");
  return e(kernel, arg_index, arg_value);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemFillINTEL(
    cl_command_queue command_queue, void *dst_ptr, const void *pattern,
    size_t pattern_size, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  const auto e = cs::load_entrypoint<clEnqueueMemFillINTEL_fn>(
      command_queue, "clEnqueueMemFillINTEL");
  return e(command_queue, dst_ptr, pattern, pattern_size, size,
           num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue command_queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  const auto e = cs::load_entrypoint<clEnqueueMemcpyINTEL_fn>(
      command_queue, "clEnqueueMemcpyINTEL");
  return e(command_queue, blocking, dst_ptr, src_ptr, size,
           num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  const auto e = cs::load_entrypoint<clEnqueueMigrateMemINTEL_fn>(
      command_queue, "clEnqueueMigrateMemINTEL");
  return e(command_queue, ptr, size, flags, num_events_in_wait_list,
           event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  const auto e = cs::load_entrypoint<clEnqueueMemAdviseINTEL_fn>(
      command_queue, "clEnqueueMemAdviseINTEL");
  return e(command_queue, ptr, size, advice, num_events_in_wait_list,
           event_wait_list, event);
}
