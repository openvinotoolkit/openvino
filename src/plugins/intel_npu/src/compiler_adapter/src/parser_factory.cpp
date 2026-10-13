// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include <exception>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "parser.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {

std::shared_ptr<IProfilingDecoder> ParserFactory::makeProfilingDecoderIfAvailable(bool perfCountEnabled) const {
    if (!perfCountEnabled) {
        return nullptr;
    }

    try {
        auto vclLoader = VCLLoader::getInstance();
        OPENVINO_ASSERT(vclLoader != nullptr, "VCL loader is nullptr");
        return makeVCLProfilingDecoder(vclLoader->sharedFunctions());
    } catch (const std::exception& ex) {
        Logger::global().warning("Failed to create the VCL profiling decoder for model import: %s", ex.what());
        return nullptr;
    }
}

std::unique_ptr<IParser> ParserFactory::getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                                  bool perfCountEnabled) const {
    OPENVINO_ASSERT(
        zeroInitStructs != nullptr,
        "Could not find an NPU device. The driver compiler requires a valid device to be present in the system.");

    return std::make_unique<Parser>(zeroInitStructs, makeProfilingDecoderIfAvailable(perfCountEnabled));
}

}  // namespace intel_npu
