// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kernel.hpp"

namespace cldnn {

/// @brief Defines possible kernel formats
enum class KernelFormat {
    SOURCE,      ///< backend source code
    NATIVE_BIN,  ///< device-native executable binary
    SPIRV,       ///< portable SPIR-V module
};

/// @brief Selects how a source compiler consumes kernel-selector batch headers.
enum class KernelSourceHeaders {
    BATCH_PREAMBLE,   ///< prepend the complete header preamble used by driver compilers
    REFERENCED_ONLY,  ///< prepend the common preamble and inline headers referenced by the translation unit
};

/// @brief Backend-neutral properties of a kernel compiler used by the common cache frontend.
struct kernel_compiler_info {
    KernelFormat source_cache_format = KernelFormat::NATIVE_BIN;
    KernelSourceHeaders source_headers = KernelSourceHeaders::BATCH_PREAMBLE;
    size_t max_source_kernels_per_batch = 0;
    std::string cache_identity;
};

/// @brief Immutable, non-owning description passed to a backend kernel builder.
///
/// Payload ownership remains with the caller for the duration of build_kernels().
/// Metadata is computed when compilation/cache entries are created and is never
/// consulted from the inference dispatch path.
struct kernel_artifact {
    const void* payload = nullptr;
    size_t payload_size = 0;
    KernelFormat format = KernelFormat::SOURCE;
    std::string entry_point;
    std::string build_options;
};

/// @brief Interface for building the GPU kernels. Implementations must be thread-safe to support case where multiple threads use single builder.
class kernel_builder {
public:
    virtual ~kernel_builder() = default;
    virtual void build_kernels(const void* src, size_t src_bytes, KernelFormat src_format, const std::string& options, std::vector<kernel::ptr>& out) const = 0;

    /// @brief Build kernels from a semantically tagged artifact.
    ///
    /// Backends may override this adapter when metadata such as an explicit
    /// entry point is part of their module-creation contract.
    virtual void build_kernels(const kernel_artifact& artifact, std::vector<kernel::ptr>& out) const {
        build_kernels(artifact.payload, artifact.payload_size, artifact.format, artifact.build_options, out);
    }

    /// @brief Describes source compilation without exposing a backend or source compiler.
    virtual kernel_compiler_info get_compiler_info() const {
        return {};
    }
};

}  // namespace cldnn
