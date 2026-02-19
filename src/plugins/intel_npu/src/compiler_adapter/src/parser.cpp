// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parser.hpp"

#include "dynamic_graph.hpp"
#include "graph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "weightless_graph.hpp"

namespace intel_npu {

Parser::Parser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("Parser", Logger::global().level()) {
    _logger.info("initialize Parser start");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    OPENVINO_ASSERT(_zeGraphExt != nullptr,
                    "Failed to create ZeGraphExtWrappers in Parser. Please check if the driver is properly installed.");
}

std::shared_ptr<IGraph> Parser::parse(const ov::Tensor& mainBlob,
                                      const FilteredConfig& config,
                                      const std::optional<std::vector<ov::Tensor>>& initBlobs,
                                      std::optional<std::shared_ptr<const ov::Model>>&& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "Parser", "parse");

    const void* data = mainBlob.data();
    size_t size = mainBlob.get_byte_size();
    std::string header;
    if (size >= 20) {
        header.assign(static_cast<const char*>(data), 20);
    } else {
        header.assign(static_cast<const char*>(data), size);
    }
    if (header.find("ELF") == std::string::npos) {
        // no _compiler::parse call is required. networkmetadata will be obtained in DynamicGraph constructor
        _logger.debug("blob is not ELF format, create graph for LLVM IR!");
        return std::make_shared<DynamicGraph>(_zeroInitStruct, std::move(mainBlob), true, config);
    } else {
        _logger.debug("blob is ELF format, create graph for elf blob!");
    }

    GraphDescriptor mainGraphDesc;
    NetworkMetadata mainNetworkMetadata;

    _logger.debug("main schedule parse start");
    OV_ITT_TASK_NEXT(PARSE_BLOB, "getMainGraphDescriptor");
    mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMetaMainGraph");
    mainNetworkMetadata = _zeGraphExt->getNetworkMeta(mainGraphDesc);
    _logger.debug("main schedule parse end");
    if (model) {
        mainNetworkMetadata.name = model.value()->get_friendly_name();
    } else {
        _logger.info("networkMeta name is empty in parse!");
    }

    // exporting the blob when we get it from cache or ov::hint::compiled_blob property
    // shall be available
    const bool blobIsPersistent = config.has<COMPILED_BLOB>()       ? true
                                  : config.has<LOADED_FROM_CACHE>() ? config.get<LOADED_FROM_CACHE>()
                                                                    : false;

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       mainGraphDesc,
                                       std::move(mainNetworkMetadata),
                                       mainBlob,
                                       config,
                                       blobIsPersistent);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initNetworkMetadata;

    _logger.debug("inits schedule parse start");
    for (const auto& initBlob : initBlobs.value()) {
        OV_ITT_TASK_NEXT(PARSE_BLOB, "getInitGraphDescriptor");
        auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());
        OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMetaInitGraph");
        auto initNetworkMeta = _zeGraphExt->getNetworkMeta(initGraphDesc);

        initGraphDescriptors.push_back(initGraphDesc);
        initNetworkMetadata.push_back(std::move(initNetworkMeta));
    }
    _logger.debug("inits schedule parse end");

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphDesc,
                                             std::move(mainNetworkMetadata),
                                             mainBlob,
                                             initGraphDescriptors,
                                             std::move(initNetworkMetadata),
                                             initBlobs,
                                             std::move(model.value()),
                                             config,
                                             blobIsPersistent);
}

}  // namespace intel_npu
