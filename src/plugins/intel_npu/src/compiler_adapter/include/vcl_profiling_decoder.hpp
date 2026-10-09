// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "zero_profiling_decoder.hpp"

namespace intel_npu {

/**
 * @brief Decodes VCL profiling output using only the VCL profiling entry points.
 */
class VCLProfilingDecoder final : public IProfilingDecoder {
public:
    /**
     * @param functions A shared pointer to the VCL function table. Must expose the profiling entry points.
     */
    explicit VCLProfilingDecoder(std::shared_ptr<const VCLFunctionTable> functions);

    std::vector<ov::ProfilingInfo> decode(const IGraph& graph,
                                          const zeroProfiling::ProfilingQuery& query) const override;

    std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData, const ov::Tensor& network) const;

private:
    std::shared_ptr<const VCLFunctionTable> _functions;

    Logger _logger;
};

/**
 * @brief Resolves the VCL function table via VCLLoader and wraps it in a decoder.
 * @details For callers (e.g. Parser) that have not already resolved a VCLFunctionTable of their own, such as
 * VCLCompilerImpl has (see VCLCompilerImpl::createProfilingDecoder()). Throws if the VCL compiler library cannot
 * be loaded.
 */
std::shared_ptr<IProfilingDecoder> makeVCLProfilingDecoder();

}  // namespace intel_npu
