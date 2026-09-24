// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * Level Zero context and Level Zero shared memory tensors
 *
 * @file openvino/runtime/intel_gpu/ze/ze.hpp
 */
#pragma once

#include <string>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @defgroup ov_runtime_ze_gpu_cpp_api Intel GPU Level Zero interoperability
 * @ingroup ov_runtime_cpp_api
 * Set of C++ classes and properties to work with Remote API for Intel GPU Level Zero plugin.
 */

/**
 * @brief Namespace with Intel GPU Level Zero specific remote objects
 */
namespace ze {

using gpu_handle_param = ov::intel_gpu::gpu_handle_param;

using SharedBufferHandle = ov::intel_gpu::SharedBufferHandle;
using VirtualAddressMemory = ov::intel_gpu::VirtualAddressMemory;

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * backed by Level Zero addressable memory.
 * The plugin object derived from this class can be obtained with ZeContext::create_tensor() call.
 * @note User can obtain Level Zero memory pointer from this class.
 * @ingroup ov_runtime_ze_gpu_cpp_api
 */
class ZeRemoteTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::mem_handle.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::CPU_VA).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::USM_USER_BUFFER).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER).as<std::string>()}}});
    }

    /**
     * @brief Returns the underlying Level Zero memory pointer.
     * @return underlying Level Zero memory pointer
     */
    void* get() {
        return static_cast<void*>(get_params().at(ov::intel_gpu::mem_handle.name()).as<gpu_handle_param>());
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is served by the Level Zero runtime backend of the GPU plugin.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::get_default_context() calls.
 * @note On devices where OpenCL<->Level Zero interoperability (LEO) is enabled,
 * the context reported by the plugin can be of ov::intel_gpu::ContextType::OCL
 * type as well - this class accepts both.
 * @ingroup ov_runtime_ze_gpu_cpp_api
 */
class ZeContext : public RemoteContext {
public:
    // Needed to make create_tensor overloads from base class visible for user
    using RemoteContext::create_tensor;

    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_context A remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        RemoteContext::type_check(remote_context,
                                  {{std::string(ov::intel_gpu::ocl_context.name()), {}},
                                   {std::string(ov::intel_gpu::context_type.name()),
                                    {ov::Any(ov::intel_gpu::ContextType::ZE).as<std::string>(),
                                     ov::Any(ov::intel_gpu::ContextType::OCL).as<std::string>()}}});
    }

    /**
     * @brief This function is used to obtain a remote tensor object that wraps user-supplied
     * CPU virtual address memory, using host-buffer creation exposed by the Level Zero backend.
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A description of user-supplied CPU virtual address memory to wrap
     * @return A remote tensor instance
     */
    ZeRemoteTensor create_tensor(const element::Type type, const Shape& shape, VirtualAddressMemory buffer) {
        OPENVINO_ASSERT(buffer.ptr != nullptr, "host buffer must not be nullptr for CPU_VA memory type");

        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::CPU_VA},
                         {ov::intel_gpu::cpu_va.name(), static_cast<gpu_handle_param>(buffer.ptr)},
                         {ov::intel_gpu::cpu_va_size.name(), buffer.size},  // if -1 then use shape to get the size
                         {ov::intel_gpu::cpu_va_access.name(), buffer.access}};
        return create_tensor(type, shape, params).as<ZeRemoteTensor>();
    }
};

}  // namespace ze
}  // namespace intel_gpu
}  // namespace ov
