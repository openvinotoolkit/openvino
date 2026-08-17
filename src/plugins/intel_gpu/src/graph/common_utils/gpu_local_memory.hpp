// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "intel_gpu/runtime/kernel_args.hpp"

namespace cldnn {

class BinaryInputBuffer;
class BinaryOutputBuffer;

/// Backend mapping selected when a logical work-group memory requirement is prepared.
enum class gpu_local_memory_mapping : uint8_t {
    backend_argument,
    specialization_constant,
    static_shader,
};

/// Backend-neutral description of one logical work-group memory allocation.
struct gpu_local_memory_requirement {
    uint64_t byte_size = 0;
    uint32_t element_size = 1;
    gpu_local_memory_mapping mapping = gpu_local_memory_mapping::backend_argument;
    uint32_t mapping_id = 0;
    bool runtime_resolved = false;
};

/// Compile/update-time contract for work-group-local memory used by a dispatch.
///
/// materialize() translates logical byte requirements into an existing backend
/// argument or a pipeline specialization constant. It is intentionally not
/// called by gpu_execution_plan::execute(), so dynamic allocation and mapping
/// decisions stay out of the inference dispatch path.
class gpu_local_memory_contract final {
public:
    static constexpr uint32_t current_serialization_version = 1;

    void add(gpu_local_memory_requirement requirement);

    bool empty() const noexcept;
    size_t size() const noexcept;
    const gpu_local_memory_requirement& operator[](size_t index) const;

    void materialize(kernel_arguments_desc& descriptor, local_memory_args_desc& backend_arguments, uint64_t max_local_memory_bytes) const;

    void save(BinaryOutputBuffer& output) const;
    void load(BinaryInputBuffer& input);

private:
    static void validate(const gpu_local_memory_requirement& requirement, uint64_t max_local_memory_bytes);

    std::vector<gpu_local_memory_requirement> _requirements;
};

}  // namespace cldnn
