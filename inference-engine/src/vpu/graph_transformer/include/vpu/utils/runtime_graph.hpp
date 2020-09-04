// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/perf_report.hpp>
#include <ie_icnn_network.hpp>

#include <vector>

namespace vpu {

InferenceEngine::ICNNNetwork::Ptr buildRuntimeGraphAsIeNet(
        GraphMetaInfo& graphMetaInfo,
        const std::vector<float>& perfInfo);
InferenceEngine::ICNNNetwork::Ptr buildRuntimeGraph(
        GraphMetaInfo& graphMetaInfo,
        const std::vector<float>& perfInfo);


}  // namespace vpu
