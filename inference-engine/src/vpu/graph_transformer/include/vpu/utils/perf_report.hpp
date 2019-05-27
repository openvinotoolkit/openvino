// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>

#include <ie_common.h>

#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

struct StageMetaInfo final {
    ie::InferenceEngineProfileInfo::LayerStatus status = ie::InferenceEngineProfileInfo::LayerStatus::NOT_RUN;

    std::string layerName;
    std::string layerType;

    std::string stageName;
    std::string stageType;
};

VPU_DECLARE_ENUM(PerfReport,
    PerLayer,
    PerStage
)

std::map<std::string, ie::InferenceEngineProfileInfo> parsePerformanceReport(
        const std::vector<StageMetaInfo>& stagesMeta,
        const float* deviceTimings,
        int deviceTimingsCount,
        PerfReport perfReport,
        bool printReceiveTensorTime);

}  // namespace vpu
