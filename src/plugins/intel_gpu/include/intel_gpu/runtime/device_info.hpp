// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_device GPU Device
/// @{

/// @brief Enumeration of supported device types
enum class device_type {
    integrated_gpu = 0,
    discrete_gpu = 1
};

enum class gpu_arch {
    unknown = 0,
    gen9 = 1,
    gen11 = 2,
    xe_lp = 3,
    xe_hp = 4,
    xe_hpg = 5,
    xe_hpc = 6,
    xe2 = 7,
    xe3 = 8,
};

/// @brief Defines version of GFX IP
struct gfx_version {
    uint16_t major;
    uint8_t minor;
    uint8_t revision;
    friend bool operator < (const gfx_version& l, const gfx_version& r)  {
        return std::tie(l.major, l.minor, l.revision)
               < std::tie(r.major, r.minor, r.revision); // same order
    }
};

/// @brief Information about the device properties and capabilities.
struct device_info {
    uint32_t execution_units_count;             ///< Number of available execution units.
    uint32_t gpu_frequency;                     ///< Clock frequency in MHz.

    uint64_t max_work_group_size;               ///< Maximum number of work-items in a work-group executing a kernel using the data parallel execution model.
    uint64_t max_local_mem_size;                ///< Maximum size of local memory arena in bytes.
    uint64_t max_global_mem_size;               ///< Maximum size of global device memory in bytes.
    uint64_t max_alloc_mem_size;                ///< Maximum size of memory object allocation in bytes.

    uint64_t max_image2d_width;                 ///< Maximum image 2d width supported by the device.
    uint64_t max_image2d_height;                ///< Maximum image 2d height supported by the device.

    bool supports_fp16;                         ///< Does engine support FP16.
    bool supports_fp64;                         ///< Does engine support FP64.
    bool supports_fp16_denorms;                 ///< Does engine support denormalized FP16.
    bool supports_khr_subgroups;                ///< Does engine support cl_khr_subgroups extension.
    bool supports_intel_subgroups;              ///< Does engine support cl_intel_subgroups extension.
    bool supports_intel_subgroups_short;        ///< Does engine support cl_intel_subgroups_short extension.
    bool supports_intel_subgroups_char;         ///< Does engine support cl_intel_subgroups_char extension.
    bool supports_intel_required_subgroup_size; ///< Does engine support cl_intel_required_subgroup_size extension.
    bool supports_local_block_io;               ///< Does engine support cl_intel_subgroup_local_block_io extension.
    bool supports_queue_families;               ///< Does engine support cl_intel_command_queue_families extension.
    bool supports_image;                        ///< Does engine support images (CL_DEVICE_IMAGE_SUPPORT cap).
    bool supports_intel_planar_yuv;             ///< Does engine support cl_intel_planar_yuv extension.

    bool supports_imad;                         ///< Does engine support int8 mad.
    bool supports_immad;                        ///< Does engine support int8 multi mad.

    bool supports_usm;                          ///< Does engine support unified shared memory.
    bool has_separate_cache;                    ///< Does the target hardware has separate cache for usm_device and usm_host

    std::vector<size_t> supported_simd_sizes;   ///< List of SIMD sizes supported by current device and compiler

    uint32_t vendor_id;                         ///< Vendor ID
    std::string dev_name;                       ///< Device ID string
    std::string driver_version;                 ///< Version of OpenCL driver

    device_type dev_type;                       ///< Defines type of current GPU device (integrated or discrete)

    gfx_version gfx_ver;                        ///< Defines GFX IP version
    gpu_arch arch;                              ///< Defines arch human readable name
    uint32_t ip_version;                        ///< Defines raw GFX IP version
    uint32_t device_id;                         ///< ID of current GPU
    uint32_t num_slices;                        ///< Number of slices
    uint32_t num_sub_slices_per_slice;          ///< Number of subslices in a slice
    uint32_t num_eus_per_sub_slice;             ///< Number of execution units per subslice
    uint32_t num_threads_per_eu;                ///< Number of hardware threads per execution unit
    uint32_t num_ccs;                           ///< Number of compute command streamers

    ov::device::UUID uuid;                      ///< UUID of the gpu device
    ov::device::LUID luid;                      ///< LUID of the gpu device
};

/// @}

/// @}

}  // namespace cldnn
