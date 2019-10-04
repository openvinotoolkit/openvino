// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/perf_report.hpp>

#include <vector>
#include <string>
#include <map>

namespace vpu {

std::map<std::string, ie::InferenceEngineProfileInfo> parsePerformanceReport(
        const std::vector<StageMetaInfo>& stagesMeta,
        const float* deviceTimings,
        int deviceTimingsCount,
        PerfReport perfReport,
        bool printReceiveTensorTime) {
    IE_ASSERT(deviceTimings != nullptr);
    IE_ASSERT(deviceTimingsCount > 0);

    std::map<std::string, ie::InferenceEngineProfileInfo> outPerfMap;

    int timeIndex = 0;
    int execIndex = 1;

    for (const auto& stageMeta : stagesMeta) {
        float timeMS = 0;
        if (stageMeta.status == ie::InferenceEngineProfileInfo::EXECUTED &&
            timeIndex < deviceTimingsCount) {
            timeMS = deviceTimings[timeIndex];
            timeIndex++;
        }

        if (stageMeta.stageType == "<Receive-Tensor>" &&
            !printReceiveTensorTime) {
            continue;
        }

        ie::InferenceEngineProfileInfo profInfo = {};

        profInfo.status = stageMeta.status;

        profInfo.cpu_uSec = 0;
        profInfo.realTime_uSec = static_cast<long long int>(timeMS * 1000);

        stageMeta.layerType.copy(profInfo.layer_type, sizeof(profInfo.layer_type) / sizeof(profInfo.layer_type[0]), 0);
        stageMeta.stageType.copy(profInfo.exec_type, sizeof(profInfo.exec_type) / sizeof(profInfo.exec_type[0]), 0);

        if (stageMeta.stageType == "<Receive-Tensor>") {
            profInfo.execution_index = 0;
        } else if (stageMeta.status == ie::InferenceEngineProfileInfo::EXECUTED) {
            profInfo.execution_index = execIndex;
            execIndex++;
        }

        if (perfReport == PerfReport::PerStage) {
            outPerfMap[stageMeta.displayStageName] = profInfo;
        } else if (perfReport == PerfReport::PerLayer) {
            auto it = outPerfMap.find(stageMeta.layerName);
            if (it == outPerfMap.end()) {
                outPerfMap[stageMeta.layerName] = profInfo;
            } else {
                auto& prevProfInfo = it->second;

                if (profInfo.status == ie::InferenceEngineProfileInfo::EXECUTED) {
                    prevProfInfo.status = ie::InferenceEngineProfileInfo::EXECUTED;
                }

                prevProfInfo.cpu_uSec += profInfo.cpu_uSec;
                prevProfInfo.realTime_uSec += profInfo.realTime_uSec;
            }
        }
    }

    return outPerfMap;
}

}  // namespace vpu
