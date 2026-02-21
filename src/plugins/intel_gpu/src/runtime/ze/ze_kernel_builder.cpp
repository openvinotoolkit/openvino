// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_kernel_builder.hpp"

// To be removed once OCL compilation is no longer required
#include "ocl/ocl_kernel_builder.hpp"
#include "ocl/ocl_device_detector.hpp"

#include <unordered_map>

using namespace cldnn;
using namespace ze;

void ze_kernel_builder::init_ocl_builder() const {
    using namespace cldnn::ocl;
    ocl_device_detector ocl_detector;
    auto devices = ocl_detector.get_available_devices(nullptr, nullptr);
    const auto &dev_info = m_device.get_info();
    m_ocl_device = nullptr;
    for (const auto& [key, dev] : devices) {
        auto ocl_dev = std::dynamic_pointer_cast<ocl_device>(dev);
        OPENVINO_ASSERT(ocl_dev != nullptr, "[GPU] Unexepected null device");
        if (ocl_dev->get_info().uuid.uuid == dev_info.uuid.uuid) {
            m_ocl_device = ocl_dev;
        }
    }
    OPENVINO_ASSERT(m_ocl_device != nullptr, "[GPU] L0 kernel builder was not able to find matching OCL device");
    if (!m_ocl_device->is_initialized())
        m_ocl_device->initialize();
    m_ocl_builder = std::make_shared<ocl_kernel_builder>(*m_ocl_device);
}

bool ze_kernel_builder::check_l0_build_support() const {
    static std::unordered_map<ze_device_handle_t, bool> cache;
    const char src[] = R"(__kernel void k(){})";
    auto src_bytes = sizeof(src);
    auto dev_handle = m_device.get_device();
    if (cache.find(dev_handle) != cache.end()) {
        return cache.at(dev_handle);
    }
    try {
        build_module_l0(src, src_bytes, KernelFormat::SOURCE, "");
        cache[dev_handle] = true;
    } catch (std::exception&) {
        GPU_DEBUG_INFO << "[GPU] Device(" << dev_handle << ") does not support kernel compilation from source through L0" << std::endl;
        cache[dev_handle] = false;
    }
    return cache.at(dev_handle);
}

std::shared_ptr<ze_module_holder> ze_kernel_builder::build_module_l0(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options) const {
    ze_module_desc_t module_desc = {
            ZE_STRUCTURE_TYPE_MODULE_DESC,
            nullptr,
            ZE_MODULE_FORMAT_NATIVE,
            src_bytes,
            reinterpret_cast<const uint8_t *>(src),
            options.c_str(),
            nullptr // specialization constants
        };
    switch (src_format) {
        case KernelFormat::SOURCE: {
            module_desc.format = ze_module_format_oclc;
            break;
        }
        case KernelFormat::NATIVE_BIN: {
            module_desc.format = ZE_MODULE_FORMAT_NATIVE;
            break;
        }
        default:
            OPENVINO_THROW("[GPU] Trying to build kernel from unexpected format");
            break;
    }
    ze_module_handle_t module_handle;
    ze_module_build_log_handle_t log_handle;
    ze_result_t build_result = zeModuleCreate(m_device.get_context(), m_device.get_device(), &module_desc, &module_handle, &log_handle);
    if (build_result != ZE_RESULT_SUCCESS) {
        size_t log_size = 0;
        OV_ZE_EXPECT(zeModuleBuildLogGetString(log_handle, &log_size, nullptr));
        std::string log(log_size, ' ');
        OV_ZE_EXPECT(zeModuleBuildLogGetString(log_handle, &log_size, log.data()));
        OV_ZE_EXPECT(zeModuleBuildLogDestroy(log_handle));
        GPU_DEBUG_INFO << "-------- Kernel build error" << std::endl;
        GPU_DEBUG_INFO << log << std::endl;
        GPU_DEBUG_INFO << "-------- End of Kernel build error" << std::endl;
        OPENVINO_THROW("[GPU] Failed to build module");
    }
    return std::make_shared<ze_module_holder>(module_handle, log_handle);
}

std::shared_ptr<ze_module_holder> ze_kernel_builder::build_module_ocl(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options) const {
    OPENVINO_ASSERT(src_format == KernelFormat::SOURCE, "[GPU] L0 kernel builder should only fallback to OCL when building kernels from source");
    OPENVINO_ASSERT(m_ocl_builder != nullptr, "[GPU] L0 kernel builder expected initialized OCL builder");
    std::vector<kernel::ptr> tmp;
    m_ocl_builder->build_kernels(src, src_bytes, src_format, options, tmp);
    OPENVINO_ASSERT(tmp.size() > 0, "[GPU] L0 kernel builder expected non-empty module");
    auto binary = tmp[0]->get_binary();
    return build_module_l0(binary.data(), binary.size(), KernelFormat::NATIVE_BIN, options);
}

void ze_kernel_builder::build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const {
    std::shared_ptr<ze_module_holder> module_holder;
    if (src_format == KernelFormat::SOURCE && !check_l0_build_support()) {
        {
            std::lock_guard lock(this->m_mutex);
            // Prevent ocl builder init call from multiple threads
            if (!m_ocl_builder) {
                init_ocl_builder();
            }
        }
        module_holder = build_module_ocl(src, src_bytes, src_format, options);
    } else {
        module_holder = build_module_l0(src, src_bytes, src_format, options);
    }
    ze_kernel::create_kernels_from_module(module_holder, out);
}
