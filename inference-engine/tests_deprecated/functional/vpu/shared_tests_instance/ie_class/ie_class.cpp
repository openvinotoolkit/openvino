// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_class.hpp"
#include "vpu_tests_config.hpp"
#include "common_test_utils/file_utils.hpp"

using IEClassExecutableNetworkGetMetricTest_nightly = IEClassExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetConfigTest_nightly = IEClassExecutableNetworkGetConfigTest;

using IEClassGetMetricTest_nightly = IEClassGetMetricTest;
using IEClassGetConfigTest_nightly = IEClassGetConfigTest;

std::string devices[] = {
    std::string(vpu::tests::deviceName()),
};

std::pair<std::string, std::string> plugins [] = {
    std::make_pair(std::string(vpu::tests::pluginName()) , std::string(vpu::tests::deviceName()) ),
};

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
    IEClassBasicTestP_smoke, IEClassBasicTestP,
    ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_CASE_P(
    IEClassNetworkTestP_smoke, IEClassNetworkTestP,
    ::testing::ValuesIn(devices));

//
// IEClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

class IEClassNetworkTestP_VPU : public IEClassNetworkTestP {};

TEST_P(IEClassNetworkTestP_VPU, smoke_ImportNetworkNoThrowIfNoDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(strm));
    if (!strm.str().empty() && deviceName.find("FPGA") != std::string::npos) {
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm));
    }
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

TEST_P(IEClassNetworkTestP_VPU, smoke_ImportNetworkNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(strm));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, deviceName));
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

TEST_P(IEClassNetworkTestP_VPU, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(fileName));
    }
    if (CommonTestUtils::fileExists(fileName)) {
        {
            std::ifstream strm(fileName);
            SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

using IEClassNetworkTestP_VPU_GetMetric = IEClassNetworkTestP_VPU;
TEST_P(IEClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    Core ie;
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 1);
    ASSERT_EQ(optimizationCapabilities.front(), METRIC_VALUE(FP16));
}

INSTANTIATE_TEST_CASE_P(
    smoke_IEClassGetMetricP, IEClassNetworkTestP_VPU_GetMetric,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP, IEClassNetworkTestP_VPU,
        ::testing::Values(std::string(vpu::tests::deviceName()), "HETERO:" + std::string(vpu::tests::deviceName())));

#if defined(ENABLE_MKL_DNN) && ENABLE_MKL_DNN
INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP_HETERO_CPU, IEClassNetworkTestP_VPU,
        ::testing::Values("HETERO:" + std::string(vpu::tests::deviceName()) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest_nightly,
    IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest_nightly,
    IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest_nightly,
    IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest_nightly,
    IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest_nightly,
    IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
    ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetConfigTest_nightly,
    IEClassExecutableNetworkGetConfigTest,
    ::testing::ValuesIn(devices));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_ThrowUnsupported,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_AVAILABLE_DEVICES,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_SUPPORTED_METRICS,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetMetricTest_nightly,
    IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
    ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
    IEClassGetConfigTest_nightly,
    IEClassGetConfigTest,
    ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
    IEClassGetConfigTest_nightly,
    IEClassGetConfigTest_ThrowUnsupported,
    ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
    DISABLED_IEClassQueryNetworkTest_smoke,
    IEClassQueryNetworkTest,
    ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
    IEClassLoadNetworkTest_smoke,
    IEClassLoadNetworkTest,
    ::testing::ValuesIn(devices));
