// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class IProfilingDecoder;

class ParserFactory final {
public:
    std::unique_ptr<IParser> getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                       bool perfCountEnabled) const;

private:
    /**
     * @brief The only place that resolves the VCL function table for the decoder; Parser only receives the
     * already-built decoder.
     */
    std::shared_ptr<IProfilingDecoder> makeProfilingDecoderIfAvailable(bool perfCountEnabled) const;
};

}  // namespace intel_npu
