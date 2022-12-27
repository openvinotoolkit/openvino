// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory tensors
 *
 * @file openvino/runtime/intel_gpu/ocl/ocl.hpp
 */
#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace intel_auto {

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class MultiContext : public RemoteContext {
protected:
    /**
     * @brief MULTI device name
     */
    static constexpr const char* device_name = "MULTI";

public:
    /**
     * @brief Constructs context object from user-supplied context list
     * @param contexts a vector holding the hardware contexts
     */
    MultiContext(Core& core, std::vector<RemoteContext> contexts) {
        *this = core.create_context(device_name, contexts).as<MultiContext>();
    }
};
}  // namespace intel_auto
}  // namespace ov
