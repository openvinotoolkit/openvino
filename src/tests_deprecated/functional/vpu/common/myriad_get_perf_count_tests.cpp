// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <limits>
#include <ngraph_functions/subgraph_builders.hpp>

#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;

#if defined(_WIN32) || defined(__APPLE__) || defined(ANDROID)
typedef std::chrono::time_point<std::chrono::steady_clock> time_point;
#else
typedef std::chrono::time_point<std::chrono::system_clock> time_point;
#endif
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

#define TIMEDIFF(start, end) ((std::chrono::duration_cast<ms>((end) - (start))).count())

using myriadGetPerformanceTests_nightly = myriadLayersTests_nightly;

TEST_F(myriadGetPerformanceTests_nightly, CorrectTimings) {
    std::shared_ptr<ngraph::Function> fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();

    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork,
    {
        {
            CONFIG_KEY(PERF_COUNT),
            CONFIG_VALUE(YES)
        },
        {
            CONFIG_KEY(LOG_LEVEL),
            CONFIG_VALUE(LOG_WARNING)
        }
    }));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());

    time_point start = Time::now();
    ASSERT_NO_THROW(_inferRequest.Infer());
        time_point end = Time::now();
    double inferTime_mSec = (std::chrono::duration_cast<ms>(end - start)).count();

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = _inferRequest.GetPerformanceCounts());
        long long stagesTime_uSec = 0;
    for (const auto &i : perfMap) {
        stagesTime_uSec += i.second.realTime_uSec;
    }
    double stagesTime_mSec = stagesTime_uSec / 1000.0;
    ASSERT_TRUE(stagesTime_mSec > std::numeric_limits<double>::epsilon() && stagesTime_mSec < inferTime_mSec);
}
