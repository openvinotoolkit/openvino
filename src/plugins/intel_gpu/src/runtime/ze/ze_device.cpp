// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device.hpp"
#include "ze_common.hpp"

#include <ze_api.h>
#include <ze_intel_gpu.h>
#include <vector>
#include <algorithm>
#include <cassert>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "gpu/intel/jit/generator.hpp"
#endif

namespace cldnn {
namespace ze {

namespace {
#ifdef ENABLE_ONEDNN_FOR_GPU
//TODO merge this with ocl_device
gpu_arch convert_ngen_arch(ngen::HW gpu_arch) {
    switch (gpu_arch) {
        case ngen::HW::Gen9: return gpu_arch::gen9;
        case ngen::HW::Gen11: return gpu_arch::gen11;
        case ngen::HW::XeLP: return gpu_arch::xe_lp;
        case ngen::HW::XeHP: return gpu_arch::xe_hp;
        case ngen::HW::XeHPG: return gpu_arch::xe_hpg;
        case ngen::HW::XeHPC: return gpu_arch::xe_hpc;
        case ngen::HW::Xe2: return gpu_arch::xe2;
        case ngen::HW::Xe3: return gpu_arch::xe3;
        case ngen::HW::Gen10:
        case ngen::HW::Unknown: return gpu_arch::unknown;
    }
    return gpu_arch::unknown;
}
#endif

bool supports_extension(const std::vector<ze_driver_extension_properties_t>& extensions, const std::string& ext_name, uint32_t ext_ver) {
    return std::find_if(extensions.begin(), extensions.end(), [&ext_name, &ext_ver](const ze_driver_extension_properties_t& ep) {
        return std::string(ep.name) == ext_name && ep.version == ext_ver;
    }) != extensions.end();
}

device_info init_device_info(ze_driver_handle_t driver, ze_device_handle_t device) {
    device_info info;

    uint32_t num_ext = 0;
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &num_ext, nullptr));

    std::vector<ze_driver_extension_properties_t> extensions(num_ext);
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &num_ext, &extensions[0]));

    ze_driver_properties_t driver_properties{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
    ZE_CHECK(zeDriverGetProperties(driver, &driver_properties));

    bool supports_luid = supports_extension(extensions, ZE_DEVICE_LUID_EXT_NAME, ZE_DEVICE_LUID_EXT_VERSION_1_0);
    bool supports_ip_version = supports_extension(extensions, ZE_DEVICE_IP_VERSION_EXT_NAME, ZE_DEVICE_IP_VERSION_VERSION_1_0);
    bool supports_mutable_list = supports_extension(extensions, ZE_MUTABLE_COMMAND_LIST_EXP_NAME, ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_0);
    bool supports_pci_properties = supports_extension(extensions, ZE_PCI_PROPERTIES_EXT_NAME, ZE_PCI_PROPERTIES_EXT_VERSION_1_0);
    bool supports_cp_offload =
        supports_extension(extensions, ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME, ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_1_0);
    bool supports_dp_properties =
        supports_extension(extensions, ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME, ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_1_0);

    ze_device_ip_version_ext_t ip_version_properties = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT, nullptr, 0};
    ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, supports_ip_version ? &ip_version_properties : nullptr};
    ZE_CHECK(zeDeviceGetProperties(device, &device_properties));

    ze_device_compute_properties_t device_compute_properties{ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
    ZE_CHECK(zeDeviceGetComputeProperties(device, &device_compute_properties));

    uint32_t queue_properties_count = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &queue_properties_count, nullptr));

    std::vector<ze_command_queue_group_properties_t> queue_properties(queue_properties_count);
    for (auto& mp : queue_properties) {
        mp.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }

    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &queue_properties_count, &queue_properties[0]));

    auto compute_queue_props = std::find_if(queue_properties.begin(), queue_properties.end(), [](const ze_command_queue_group_properties_t& qp) {
        return (qp.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == true;
    });

    OPENVINO_ASSERT(compute_queue_props != queue_properties.end());

    uint32_t memory_properties_count = 0;
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, nullptr));

    std::vector<ze_device_memory_properties_t> device_memory_properties(memory_properties_count);
    for (auto& mp : device_memory_properties) {
        mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    }
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, &device_memory_properties[0]));

    ze_device_memory_access_properties_t device_memory_access_properties{ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
    ZE_CHECK(zeDeviceGetMemoryAccessProperties(device, &device_memory_access_properties));

    auto mem_properties = std::find_if(device_memory_properties.begin(), device_memory_properties.end(), [](const ze_device_memory_properties_t& p) {
        auto name = std::string(p.name);
        return name == "DDR" || name == "HBM";
    });

    ze_device_module_properties_t device_module_properties{ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES};
    ze_intel_device_module_dp_exp_properties_t dp_properties{ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES, nullptr};
    if (supports_dp_properties) {
        device_module_properties.pNext = &dp_properties;
    }
    ZE_CHECK(zeDeviceGetModuleProperties(device, &device_module_properties));

    ze_device_image_properties_t device_image_properties{ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES};
    ZE_CHECK(zeDeviceGetImageProperties(device, &device_image_properties));

    info.vendor_id = device_properties.vendorId;
    info.dev_name = device_properties.name;
    info.driver_version = std::to_string(driver_properties.driverVersion);
    info.dev_type = (device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? device_type::integrated_gpu : device_type::discrete_gpu;

    info.execution_units_count = device_properties.numEUsPerSubslice * device_properties.numSubslicesPerSlice * device_properties.numSlices;

    info.gpu_frequency = device_properties.coreClockRate;

    info.supported_simd_sizes = {};
    info.has_separate_cache = true;

    info.max_work_group_size = device_compute_properties.maxTotalGroupSize;
    info.max_local_mem_size = device_compute_properties.maxSharedLocalMemory;

    if (mem_properties != device_memory_properties.end()) {
        info.max_global_mem_size = mem_properties->totalSize;
        info.device_memory_ordinal = std::distance(device_memory_properties.begin(), mem_properties);
    } else {
        info.max_global_mem_size = 0;
        info.device_memory_ordinal = 0;
    }

    info.max_alloc_mem_size = device_properties.maxMemAllocSize;

    info.supports_image = device_image_properties.maxSamplers > 0;
    info.supports_intel_planar_yuv = false;
    info.max_image2d_width = device_image_properties.maxImageDims2D;
    info.max_image2d_height = device_image_properties.maxImageDims2D;

    info.supports_fp16 = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP16) != 0;
    info.supports_fp64 = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP64) != 0;
    info.supports_fp16_denorms = info.supports_fp16 && (device_module_properties.fp16flags & ZE_DEVICE_FP_FLAG_DENORM) != 0;

    info.supports_khr_subgroups = true;
    info.supports_intel_subgroups = true;
    info.supports_intel_subgroups_short = true;
    info.supports_intel_subgroups_char = true;
    info.supports_intel_required_subgroup_size = true;
    info.supports_cp_offload = supports_cp_offload;

    info.supports_imad = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_DP4A) != 0;
    info.supports_immad = supports_dp_properties && (dp_properties.flags & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS) != 0;

    info.supports_usm = device_memory_access_properties.hostAllocCapabilities && device_memory_access_properties.deviceAllocCapabilities;

    info.gfx_ver = {0, 0, 0}; // could find how to retrieve this from L0 so far
    info.ip_version = ip_version_properties.ipVersion;
    info.sub_device_idx = (std::numeric_limits<uint32_t>::max)();

    info.device_id = device_properties.deviceId;
    info.num_slices = device_properties.numSlices;
    info.num_sub_slices_per_slice = device_properties.numSubslicesPerSlice;
    info.num_eus_per_sub_slice = device_properties.numEUsPerSubslice;
    info.num_threads_per_eu = device_properties.numThreadsPerEU;

    info.num_ccs = compute_queue_props->numQueues;
    info.supports_queue_families = true;

    info.kernel_timestamp_valid_bits  = device_properties.kernelTimestampValidBits;
    info.timer_resolution  = device_properties.timerResolution;
    info.compute_queue_group_ordinal = std::distance(queue_properties.begin(), compute_queue_props);

    static_assert(ZE_MAX_DEVICE_UUID_SIZE == ov::device::UUID::MAX_UUID_SIZE, "");
    static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == ov::device::LUID::MAX_LUID_SIZE, "");
    std::copy_n(&device_properties.uuid.id[0], ZE_MAX_DEVICE_UUID_SIZE, info.uuid.uuid.begin());

    if (supports_luid) {
        ze_device_luid_ext_properties_t luid_props{ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES, nullptr};
        ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &luid_props};
        if (zeDeviceGetProperties(device, &device_properties) == ZE_RESULT_SUCCESS)
            std::copy_n(&luid_props.luid.id[0], ZE_MAX_DEVICE_LUID_SIZE_EXT, info.luid.luid.begin());
    }

    info.supports_mutable_command_list = false;

    if (supports_mutable_list) {
        ze_mutable_command_list_exp_properties_t mutable_list_props = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES,  nullptr, 0, 0 };
        ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &mutable_list_props};
        if (zeDeviceGetProperties(device, &device_properties) == ZE_RESULT_SUCCESS) {
            ze_mutable_command_exp_flags_t required_features = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_SIGNAL_EVENT |
                                                               ZE_MUTABLE_COMMAND_EXP_FLAG_WAIT_EVENTS;

            info.supports_mutable_command_list = (mutable_list_props.mutableCommandFlags & required_features) == required_features;
        }
    }
    if (supports_pci_properties) {
        ze_pci_ext_properties_t pci_properties{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES, nullptr};
        if (zeDevicePciGetPropertiesExt(device, &pci_properties) == ZE_RESULT_SUCCESS) {
            info.pci_info.pci_bus = pci_properties.address.bus;
            info.pci_info.pci_device = pci_properties.address.device;
            info.pci_info.pci_domain = pci_properties.address.domain;
            info.pci_info.pci_function = pci_properties.address.function;
        }
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    using namespace dnnl::impl::gpu::intel::jit;
    // Create temporary context just for OneDNN HW detection
    ze_context_desc_t context_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
    ze_context_handle_t context;
    ZE_CHECK(zeContextCreate(driver, &context_desc, &context));
    ngen::Product product = ngen::LevelZeroCodeGenerator<ngen::HW::Unknown>::detectHWInfo(context, device);
    zeContextDestroy(context);
    info.arch = convert_ngen_arch(ngen::getCore(product.family));

    if (product.family == ngen::ProductFamily::Unknown) {
        info.supports_immad = false;
    }
#else  // ENABLE_ONEDNN_FOR_GPU
    info.arch = gpu_arch::unknown;
#endif  // ENABLE_ONEDNN_FOR_GPU

    return info;
}

memory_capabilities init_memory_caps(ze_device_handle_t device, const device_info& info) {
    std::vector<allocation_type> memory_caps;

    ze_device_memory_access_properties_t device_memory_access_properties{ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
    ZE_CHECK(zeDeviceGetMemoryAccessProperties(device, &device_memory_access_properties));

    if (info.supports_usm) {
        if (device_memory_access_properties.hostAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_host);
        }
        if (device_memory_access_properties.sharedSingleDeviceAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_shared);
        }
        if (device_memory_access_properties.deviceAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_device);
        }
    }

    return memory_capabilities(memory_caps);
}

}  // namespace


ze_device::ze_device(ze_driver_handle_t driver, ze_device_handle_t device, bool initialize)
: _driver(driver)
, _device(device)
, _info(init_device_info(driver, device))
, _mem_caps(init_memory_caps(device, _info)) {
    if (initialize) {
        this->initialize();
    }
}

void ze_device::initialize() {
    if (_is_initialized)
        return;

    ze_context_desc_t context_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
    ZE_CHECK(zeContextCreate(_driver, &context_desc, &_context));
    _is_initialized = true;
}

bool ze_device::is_initialized() const {
    return _is_initialized;
}

bool ze_device::is_same(const device::ptr other) {
    auto casted = downcast<ze_device>(other.get());
    if (!casted)
        return false;

    if (is_initialized() && casted->is_initialized()) {
        // Do not compare contexts as one driver can have many different contexts
        return _device == casted->get_device() && _driver == casted->get_driver();
    }
    return _info.is_same_device(casted->_info);
}

void ze_device::set_mem_caps(const memory_capabilities& memory_capabilities) {
    _mem_caps = memory_capabilities;
}

ze_device::~ze_device() {
    //FIXME segfault
    //if (_is_initialized)
    //    zeContextDestroy(_context);
}

}  // namespace ze
}  // namespace cldnn
