// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>
#include <utility>

#include <ie_icore.hpp>
#include <caseless.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/optional.hpp>
#include <vpu/configuration/plugin_configuration.hpp>

#include "mvnc.h"

namespace vpu {

namespace ie = InferenceEngine;

//
// DataInfo
//

struct DataInfo final {
    std::unordered_map<std::string, int> offset;
    std::unordered_map<std::string, ie::TensorDesc> descFromPlugin;
    int totalSize = 0;
};

//
// CompiledGraph
//

struct CompiledGraph final {
    using Ptr = std::shared_ptr<CompiledGraph>;

    std::vector<char> blob;
    std::pair<char*, size_t> blobHeader;

    std::string networkName;

    int networkBatch = 0;

    GraphMetaInfo graphMeta;
    int numActiveStages = 0;

    DataInfo inputInfo;
    DataInfo outputInfo;

    int inputBufSize = 0;
    int outputBufSize = 0;

    std::uint32_t numShaves = 0;
    std::uint32_t numSlices = 0;
    std::uint32_t numExecutors = 0;
};

//
// compileNetwork
//

CompiledGraph::Ptr compileNetwork(const ie::CNNNetwork& network, const PluginConfiguration& config, const Logger::Ptr& log,
                                  const std::shared_ptr<ie::ICore> core);

CompiledGraph::Ptr compileSubNetwork(const ie::CNNNetwork& network, const PluginConfiguration& subConfig, const std::shared_ptr<ie::ICore> core);

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayers(const ie::CNNNetwork& network, const PluginConfiguration& config, const Logger::Ptr& log,
                                         const std::shared_ptr<ie::ICore> core, const std::set<std::string>& namesToExclude = {});

//
// Blob version and checks
//

const uint32_t BLOB_MAGIC_NUMBER  = 9709;
const uint32_t BLOB_VERSION_MAJOR = 6;
// Must be changed when possible
const uint32_t BLOB_VERSION_MINOR = 0;

}  // namespace vpu
