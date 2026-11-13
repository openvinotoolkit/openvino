// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/common/iprofiling_decoder.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"

namespace intel_npu {

/**
 * @brief Decodes VCL profiling output using only the VCL profiling entry points.
 */
class VCLProfilingDecoder final : public IProfilingDecoder {
public:
    /**
     * @param functions A shared pointer to the VCL function table. Must expose the profiling entry points.
     */
    explicit VCLProfilingDecoder(const std::shared_ptr<const VCLFunctionTable>& functions);

    std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData,
                                          const ov::Tensor& network) const override;

private:
    std::shared_ptr<const VCLFunctionTable> _functions;

    Logger _logger;
};

}  // namespace intel_npu
