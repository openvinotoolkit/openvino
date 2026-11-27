// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

// Simplified runtime interface for MEGAKERNEL POC.
namespace mk {

// Simple error codes.
using TErrorcode = int;

// Constant parameters (weights) for the runtime.
class IConstantParams {};

// Runtime parameters (inputs/outputs) for the runtime.
class IRuntimeParams {};

// Platform-specific parameters for the runtime(e.g., OpenCL context, device,
// etc.).
class IPlatformParams {};

// [NOTE]: Temporary here. It should be a part of the platform-specific implementation.
struct OCLPlatformParams : IPlatformParams {
  cl_device_id deviceId = nullptr;
  cl_context context = nullptr;
  cl_command_queue stream = nullptr;
};

// Interface for a MegaKernel runtime.
class IMegakernelRuntime {
 public:
  // Initialize the runtime with constant parameters and platform-specific
  // parameters. This is where e.g. megakernel compilation and memory allocation
  // would happen.
  virtual TErrorcode Init(const IConstantParams* weights,
                          const IPlatformParams* platform) = 0;

  // Execute(run inference) the runtime with the given input/output parameters.
  virtual TErrorcode Execute(const IRuntimeParams* io) = 0;

  // Destroy the runtime and release any allocated resources.
  virtual TErrorcode Destroy() = 0;

  virtual ~IMegakernelRuntime() = default;
};

}  // namespace mk

extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();
extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);