// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_profiling_decoder.hpp"

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#include <cstring>
#include <utility>

#include "intel_npu/profiling.hpp"
#include "vcl_error_utils.hpp"

namespace intel_npu {

VCLProfilingDecoder::VCLProfilingDecoder(std::shared_ptr<const VCLFunctionTable> functions)
    : _functions(std::move(functions)),
      _logger("VCLProfilingDecoder", Logger::global().level()) {
    OPENVINO_ASSERT(_functions != nullptr, "VCLProfilingDecoder requires a non-null VCLFunctionTable");
    OPENVINO_ASSERT(_functions->vclGetVersion && _functions->vclLogHandleGetString && _functions->vclProfilingCreate &&
                        _functions->vclProfilingGetProperties && _functions->vclGetDecodedProfilingBuffer &&
                        _functions->vclProfilingDestroy,
                    "VCLProfilingDecoder received a VCLFunctionTable missing a required profiling entry point. "
                    "Was it populated from a VCLLoader?");
}

std::vector<ov::ProfilingInfo> VCLProfilingDecoder::decode(const std::vector<uint8_t>& profData,
                                                           const ov::Tensor& network) const {
    _logger.debug("decode start");

    vcl_profiling_handle_t profilingHandle;
    vcl_profiling_input_t profilingInput = {network.data<const uint8_t>(),
                                            network.get_byte_size(),
                                            profData.data(),
                                            profData.size()};
    vcl_log_handle_t logHandle;
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclProfilingCreate",
                          _functions->vclProfilingCreate(&profilingInput, &profilingHandle, &logHandle),
                          nullptr);

    vcl_profiling_properties_t profProperties;
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclProfilingGetProperties",
                          _functions->vclProfilingGetProperties(profilingHandle, &profProperties),
                          logHandle);

    _logger.info("VCL Profiling Properties: Version: %d.%d",
                 profProperties.version.major,
                 profProperties.version.minor);

    // We only use layer level info
    vcl_profiling_request_type_t request = VCL_PROFILING_LAYER_LEVEL;

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclGetDecodedProfilingBuffer",
                          _functions->vclGetDecodedProfilingBuffer(profilingHandle, request, &profOutput),
                          logHandle);
    if (profOutput.data == NULL) {
        OPENVINO_THROW("Failed to get VCL profiling output");
    }

    std::vector<ze_profiling_layer_info> layerInfo(profOutput.size / sizeof(ze_profiling_layer_info));
    if (profOutput.size > 0) {
        _logger.debug("VCL profiling output size: %d", profOutput.size);
        std::memcpy(layerInfo.data(), profOutput.data, profOutput.size);
    }

    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclProfilingDestroy",
                          _functions->vclProfilingDestroy(profilingHandle),
                          logHandle);

    // Return processed profiling info
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

// SoPtr<decoder, vcllib>
ov::SoPtr<VCLProfilingDecoder> makeVCLProfilingDecoder() {
    auto vclLoader = VCLLoader::getInstance();
    OPENVINO_ASSERT(vclLoader != nullptr, "VCL loader is nullptr");

    auto decoder = std::make_shared<VCLProfilingDecoder>(vclLoader->sharedFunctions());

    // Pairing the decoder with the library keeps the .so alive for as long as the decoder is.
    auto vclLib = vclLoader->getLibrary();
    OPENVINO_ASSERT(vclLib != nullptr, "VCL library is nullptr");

    return ov::SoPtr<VCLProfilingDecoder>(decoder, vclLib);
}

}  // namespace intel_npu
