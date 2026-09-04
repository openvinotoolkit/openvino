// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "vulkan_pipeline_cache.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_device;

class vulkan_kernel final : public kernel {
public:
    vulkan_kernel(std::shared_ptr<vulkan_device> device, std::vector<uint8_t> spirv, std::string entry_point);

    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override;
    bool is_same(const kernel& other) const override;
    std::string get_id() const override;
    std::vector<uint8_t> get_binary() const override;
    std::string get_build_log() const override;

    std::shared_ptr<const vulkan_pipeline_state> get_or_create_pipeline(uint32_t descriptor_count,
                                                                        uint32_t push_constants_size,
                                                                        const std::array<size_t, 3>& local_size,
                                                                        const vulkan_specialization_constants& specialization_constants = {});

private:
    struct shared_state;
    explicit vulkan_kernel(std::shared_ptr<shared_state> state);

    std::shared_ptr<shared_state> _state;
};

}  // namespace vulkan
}  // namespace cldnn
