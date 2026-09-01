// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "fusion_analysis.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "kernel_selection.hpp"
#include "metadata_builder.hpp"
#include "vulkan/vulkan_pipeline_cache.hpp"

namespace cldnn::vulkan::eltwise_detail {

class EltwiseDispatch final {
public:
    static EltwiseDispatch build(eltwise_inst& instance,
                                 kernel_kind kind,
                                 const std::optional<scalar_constant>& scalar,
                                 const std::optional<fused_eltwise_chain>& fused,
                                 const std::optional<fused_post_op_info>& post_op,
                                 const EltwiseMetadata& metadata,
                                 memory::ptr metadata_memory,
                                 uint32_t elements_per_invocation,
                                 uint32_t local_size,
                                 bool restricted_kernel_available);

    kernel_arguments_desc& descriptor() {
        return _descriptor;
    }

    kernel_arguments_data& arguments() {
        return _arguments;
    }

    kernel_arguments_data&& take_arguments() {
        return std::move(_arguments);
    }

    const vulkan_specialization_constants& specialization_constants() const {
        return _specialization_constants;
    }

    size_t kernel_index() const {
        return _kernel_index;
    }

private:
    kernel_arguments_desc _descriptor;
    kernel_arguments_data _arguments;
    vulkan_specialization_constants _specialization_constants;
    size_t _kernel_index = 0;
};

}  // namespace cldnn::vulkan::eltwise_detail
