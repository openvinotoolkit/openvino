// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include <exception>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "parser.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {

std::unique_ptr<IParser> ParserFactory::getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                                  bool perfCountEnabled,
                                                  const VCLFunctionTableProvider& vclFunctions) const {
    OPENVINO_ASSERT(
        zeroInitStructs != nullptr,
        "Could not find an NPU device. The driver compiler requires a valid device to be present in the system.");

    std::shared_ptr<IProfilingDecoder> profilingDecoder;
    if (perfCountEnabled) {
        try {
            OPENVINO_ASSERT(vclFunctions != nullptr,
                            "Profiling is enabled but no way to obtain the VCL entry points was given");
            profilingDecoder = makeVCLProfilingDecoder(vclFunctions());
        } catch (const std::exception& ex) {
            Logger::global().warning("Failed to create the VCL profiling decoder for model import: %s", ex.what());
        }
    }

    return std::make_unique<Parser>(zeroInitStructs, std::move(profilingDecoder));
}

}  // namespace intel_npu
