// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

/**
 * @brief Decodes VCL profiling output using only the VCL profiling entry points.
 * @note Construction only resolves the VCL function table (library load + symbol lookup); unlike building an
 * IVCLCompiler, it never calls vclCompilerCreate, so it's cheap enough to attempt unconditionally on import.
 */
class VCLProfilingDecoder final {
public:
    /**
     * @param functions A shared pointer to the VCL function table. Must expose the profiling entry points.
     */
    explicit VCLProfilingDecoder(std::shared_ptr<const VCLFunctionTable> functions);

    std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData, const ov::Tensor& network) const;

private:
    std::shared_ptr<const VCLFunctionTable> _functions;

    Logger _logger;
};

/**
 * @brief Resolves the VCL function table via VCLLoader and wraps the decoder together with the VCL library, the
 * same way makeVCLCompiler() does for the compiler-in-plugin. Throws if the VCL compiler library cannot be loaded.
 */
ov::SoPtr<VCLProfilingDecoder> makeVCLProfilingDecoder();

}  // namespace intel_npu
