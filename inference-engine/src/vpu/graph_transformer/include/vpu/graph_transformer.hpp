// Copyright (C) 2018-2019 Intel Corporation
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

#include <ie_icnn_network.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/optional.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// CompilationConfig
//

VPU_DECLARE_ENUM(Platform,
    UNKNOWN = 0,
    MYRIAD_2 = 2450,
    MYRIAD_X = 2480
)

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(ExecutionMode,
    AUTO = -1,
    SINGLE = 0,
    PARALLEL = 1
)

VPU_DECLARE_ENUM(ComputeLayout,
    AUTO,
    NCHW,
    NHWC,
    NCDHW,
    NDHWC
)

struct CompilationConfig final {
    //
    // Main flags
    //

    int numSHAVEs = -1;
    int numCMXSlices = -1;

    bool hwOptimization = true;

    bool hwAdaptiveMode = true;

    bool ignoreIRStatistic = false;

    std::string networkConfig;

    std::string customLayers;

    //
    // Debug flags
    //

    ComputeLayout forceLayout = ComputeLayout::AUTO;

    bool detectBatch = true;

    std::string hwWhiteList;
    std::string hwBlackList;

    std::string noneLayers;

    bool ignoreUnknownLayers = false;

    Optional<bool> copyOptimization;
    Optional<bool> injectSwOps;
    Optional<bool> packDataInCmx;

    bool mergeHwPoolToConv = true;

    //
    // Deprecated flags
    //

    float inputScale = 1.0f;
    float inputBias = 0.0f;

    bool hwDilation = false;
    std::map<std::string, std::vector<int>> ioStrides;
};


//
// DataInfo
//

struct DataInfo final {
    std::unordered_map<std::string, int> offset;
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
};

//
// compileNetwork
//

CompiledGraph::Ptr compileNetwork(
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log);

CompiledGraph::Ptr compileSubNetwork(
        const ie::ICNNNetwork& network,
        const CompilationConfig& subConfig);

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayers(
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log);

//
// Blob version and checks
//

const uint32_t BLOB_MAGIC_NUMBER  = 9709;
const uint32_t BLOB_VERSION_MAJOR = 5;
const uint32_t BLOB_VERSION_MINOR = 0;

}  // namespace vpu
