// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_platform: entry-point configuration of the standalone Vulkan core.
// All runtime types now live in the core namespace (vk_types.hpp); nothing is
// re-exported from a platform layer anymore.

#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace ov::core::vulkan {
namespace cross_platform {

// Platform targets of the Vulkan core. The runtime is identical for every
// platform; the differences are confined to instance/device setup:
//   desktop  — plain Vulkan 1.3, any driver (Windows/Linux).
//   moltenvk — Vulkan on Apple platforms: portability enumeration at instance
//              creation, VK_KHR_portability_subset enabled at device creation.
//   vulkan_sc — Vulkan SC 1.2 (embedded/Automotive): offline pipeline
//              artifacts are exported instead of (eventually) imported.
enum class vk_platform {
    desktop,
    moltenvk,
    vulkan_sc,
};

// Entry-point configuration passed to vk_engine::create(). A programmatic
// config overrides the environment; defaults come from the environment
// (OV_GPU_VK_PLATFORM, OV_GPU_VK_OFFLINE_DIR) when the defaults are used.
struct vk_platform_config {
    vk_platform platform = vk_platform::desktop;
    // The device name the engine was created for ("GPU" / "CPU" / "NPU").
    // The cross-platform core serves all of them: the physical Vulkan device
    // is picked by this name (GPU → discrete/integrated, CPU → a CPU-type
    // driver such as lavapipe, NPU → a Vulkan SC-class device, none today on
    // desktop). A ".N" suffix (device id) is ignored.
    std::string device_name = "GPU";
    // Force portability enumeration even on non-MoltenVK drivers (testing).
    bool force_portability_enumeration = false;
    // Vulkan SC: when non-empty, every built pipeline is exported into this
    // directory as <kernel_id>.spv plus a VkPipelineCache blob
    // (vk_pipeline_cache.bin) — the offline artifacts an SC system consumes.
    // Runtime pipeline creation stays enabled (full SC offline creation needs
    // the vksc_core.h SDK and replaces vkCreateComputePipelines entirely).
    std::string offline_pipeline_dir;
    // Reserved: NVIDIA multi-GPU NCCL bridge over VK_KHR_external_memory_* /
    // VK_KHR_external_semaphore_*. Not implemented yet — config slot only.
    bool nccl_bridge = false;
};

// Default entry-point configuration: platform and offline-pipeline directory
// are taken from the environment (OV_GPU_VK_PLATFORM, OV_GPU_VK_OFFLINE_DIR)
// unless a programmatic vk_platform_config overrides them.
inline vk_platform_config platform_config_from_env() {
    vk_platform_config cfg;
    if (const char* p = std::getenv("OV_GPU_VK_PLATFORM")) {
        const std::string_view s(p);
        if (s == "desktop")
            cfg.platform = vk_platform::desktop;
        else if (s == "moltenvk")
            cfg.platform = vk_platform::moltenvk;
        else if (s == "vulkan_sc")
            cfg.platform = vk_platform::vulkan_sc;
    }
    if (const char* d = std::getenv("OV_GPU_VK_OFFLINE_DIR"))
        cfg.offline_pipeline_dir = d;
    return cfg;
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
