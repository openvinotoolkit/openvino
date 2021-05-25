// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/perf_report.hpp>
#include <cpp/ie_cnn_network.h>

#include <vector>

namespace vpu {

InferenceEngine::CNNNetwork buildRuntimeGraph(
        GraphMetaInfo& graphMetaInfo,
        const std::vector<float>& perfInfo);


}  // namespace vpu
