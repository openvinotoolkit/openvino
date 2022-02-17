// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"

#ifdef _WIN32
# include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_context_api_ocl.hpp"

#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU")
);

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("GPU")
);

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("GPU")
);
} // namespace
