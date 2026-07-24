// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_evaluate.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "common/convert_common.hpp"
#include "interface/properties.hpp"

namespace ov::intel_gpu::mlir {

using namespace ::mlir;

static cl_device_id extract_device_from_context(cl_context context) {
    size_t devices_size;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting context info: ", err);
    }
    if (devices_size / sizeof(cl_device_id) != 1) {
        OPENVINO_THROW("Expected exactly one device in the context, got ", devices_size);
    }

    cl_device_id devices;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, &devices, NULL);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting device IDs: ", err);
    }

    return devices;
}

MLIREvaluateGcGPU::MLIREvaluateGcGPU(OwningOpRef<::mlir::ModuleOp> _module,
                                     std::shared_ptr<ov::EvaluationContext> loweringContext) {
    if (::ov::intel_gpu::mlir::is_debug()) {
        OPENVINO_MLIR_DEBUG_PRINT("-------------- Source MLIR --------------");
        _module->dump();
        OPENVINO_MLIR_DEBUG_PRINT("-----------------------------------------");
    }

    gc::gpu::OclModuleBuilderOpts opts;
    gc::gpu::OclModuleBuilder builder(std::move(_module), opts);

    auto it = loweringContext->find(ov::intel_gpu::ocl_context.name());
    if (it == loweringContext->end()) {
        OPENVINO_THROW("No cl_context provided for OpenCL execution");
    }
    auto context = reinterpret_cast<cl_context>(it->second.as<ov::intel_gpu::gpu_handle_param>());
    // assuming there's always one device per context
    auto device = extract_device_from_context(context);

    if (auto mod = builder.build(device, context)) {
        module = *mod;
    } else {
        OPENVINO_THROW("Failed to build gc::gpuOclModule module");
    }
}

bool MLIREvaluateGcGPU::invoke(const ov::TensorVector& inputs,
                               ov::TensorVector& outputs,
                               const ov::EvaluationContext& evaluationContext) {
    std::vector<void*> waitList;
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext, waitList);
    gc::gpu::StaticExecutor exec(module);

    auto it = evaluationContext.find(ov::internal::mlir_meta::is_kernel_arg_usm.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No is_kernel_arg_usm provided for OpenCL execution");
    }
    std::vector<bool> arg_types = it->second.as<std::vector<bool>>();

    for (size_t i = 0; i < inputs.size(); ++i) {
        exec.arg(inputs[i].data(), arg_types[i]);
    }
    for (size_t i = 0, j = inputs.size(); i < outputs.size(); ++i, ++j) {
        exec.arg(outputs[i].data(), arg_types[j]);
    }

    exec(ctx);

    maybe_set_result_events(evaluationContext, ctx);
    return true;
}

bool MLIREvaluateGcGPU::invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) {
    std::vector<void*> waitList;
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext, waitList);
    gc::gpu::DynamicExecutor exec(module);

    // Layout (5 pointers per memref, see MemRefDescriptor::append_to_packed_args
    // in transformations/op/mlir_op.cpp):
    //   [aligned, rank, shape*, strides*, is_usm]
    constexpr size_t kStride = 5;
    OPENVINO_ASSERT(args.size() % kStride == 0,
                    "[GPU] MLIREvaluateGcGPU::invoke_packed: malformed args vector");
    for (size_t i = 0; i < args.size(); i += kStride) {
        exec.arg(
            /*alignedPtr=*/args[i],
            /*rank=*/reinterpret_cast<size_t>(args[i + 1]),
            /*shape=*/reinterpret_cast<int64_t*>(args[i + 2]),
            /*strides=*/reinterpret_cast<int64_t*>(args[i + 3]),
            /*isUsm=*/reinterpret_cast<uintptr_t>(args[i + 4]) != 0
        );
    }
    exec(ctx);
    maybe_set_result_events(evaluationContext, ctx);
    return true;
}

void MLIREvaluateGcGPU::maybe_set_result_events(const ov::EvaluationContext& evaluationContext,
                                                gc::gpu::OclContext& ctx) {
    auto events_it = evaluationContext.find(ov::internal::mlir_meta::result_events.name());
    if (events_it == evaluationContext.end())
        return;

    auto retain_event = [](cl_event event) {
        const auto err = clRetainEvent(event);
        if (err != CL_SUCCESS) {
            OPENVINO_THROW("Failed to retain MLIR result event, error: ", err);
        }
    };

    auto* events = events_it->second.as<std::vector<void*>*>();
    events->reserve(events->size() + ctx.events.size());
    for (auto event : ctx.events) {
        retain_event(event);
        events->push_back(event);
    }
}

gc::gpu::OclContext MLIREvaluateGcGPU::build_ocl_context(const ov::EvaluationContext& evaluationContext,
                                                        std::vector<void*>& waitList) {
    auto it = evaluationContext.find(ov::intel_gpu::ocl_queue.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No queue provided for OpenCL execution");
    }
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(it->second.as<void*>());

    uint32_t waitListLen = 0;

    it = evaluationContext.find(ov::internal::mlir_meta::wait_list.name());
    if (it != evaluationContext.end()) {
        waitList = it->second.as<std::vector<void*>>();
        waitListLen = waitList.size();
    }

    const bool createEvents = evaluationContext.count(ov::internal::mlir_meta::result_events.name()) != 0;
    return gc::gpu::OclContext(module->runtime, queue, createEvents, waitListLen,
                               reinterpret_cast<cl_event*>(waitList.data()));
}

}  // namespace ov::intel_gpu::mlir
