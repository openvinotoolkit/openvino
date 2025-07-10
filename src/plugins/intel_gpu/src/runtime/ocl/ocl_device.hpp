// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "ocl_common.hpp"

namespace cldnn {
namespace ocl {

struct ocl_device : public device {
public:
    using ptr = std::shared_ptr<ocl_device>;

    /**
     * @brief Constructs an ocl_device from a given OpenCL device, context, and platform.
     *
     * @param dev OpenCL device handle.
     * @param ctx OpenCL context associated with the device.
     * @param platform OpenCL platform for the device.
     * @param initialize If true, the device is initialized immediately.
     *                   If false, initialization is deferred and must be performed explicitly
     *                   before any use of the device.
     *
     * Initializes device info and memory capabilities.
     * If @p initialize is true, performs full device initialization.
     */
    ocl_device(const cl::Device dev, const cl::Context& ctx, const cl::Platform& platform, bool initialize = true);

    /**
     * @brief Constructs an ocl_device by copying properties from another ocl_device instance.
     *
     * @param other Pointer to an existing ocl_device instance.
     * @param initialize If true, performs full device initialization using the @p other ocl_device's
     *                   device and context. The source device must be already initialized.
     *                   If false, initialization is deferred and must be performed explicitly
     *                   before any use of the device.
     *
     * Copies platform, device info, and memory capabilities.
     * If @p initialize is true, initializes the new ocl_device instance accordingly.
     */
    ocl_device(const ocl_device::ptr other, bool initialize = true);

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const cl::Device& get_device() const { return _device; }
    cl::Device& get_device() { return _device; }
    const cl::Context& get_context() const { return _context; }
    const cl::Platform& get_platform() const { return _platform; }
    const cl::UsmHelper& get_usm_helper() const { return *_usm_helper; }

    bool is_same(const device::ptr other) override;

    void set_mem_caps(const memory_capabilities& memory_capabilities) override;
    void set_sub_device_idx(uint32_t idx);

    void initialize() override;
    bool is_initialized() const override { return _is_initialized; };

    ~ocl_device() = default;

protected:
    void initialize_device(const cl::Device dev, const cl::Context& ctx);

    bool _is_initialized = false;

    cl::Context _context;
    cl::Device _device;
    cl::Platform _platform;
    device_info _info;
    memory_capabilities _mem_caps;
    std::unique_ptr<cl::UsmHelper> _usm_helper;
};

}  // namespace ocl
}  // namespace cldnn
