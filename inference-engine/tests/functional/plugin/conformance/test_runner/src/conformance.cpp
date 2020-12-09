// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/generate_xml_paths.hpp"
#include "functional_test_utils/core_config.hpp"
#include "conformance.hpp"

namespace ConformanceTests {

const char* targetDevice = "";
std::vector<std::string> IRFolderPaths = {};

std::string ConformanceTest::getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>>& obj) {
    std::string pathToModel, deviceName;
    std::tie(pathToModel, deviceName) = obj.param;

    std::ostringstream result;
    result << "ModelPath=" << pathToModel << "_";
    result << "TargetDevice=" << deviceName << "_";
    return result.str();
}

void ConformanceTest::LoadNetwork() {
    cnnNetwork = getCore()->ReadNetwork(pathToModel);
    function = cnnNetwork.getFunction();
    CoreConfiguration(this);
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
}

void ConformanceTest::SetUp() {
    std::tie(pathToModel, targetDevice) = this->GetParam();
}

TEST_P(ConformanceTest, ReadIR) {
    Run();
}

namespace {
INSTANTIATE_TEST_CASE_P(conformance,
                        ConformanceTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(Utils::generateXMLpaths(IRFolderPaths)),
                                ::testing::Values(targetDevice)),
                        ConformanceTest::getTestCaseName);
} // namespace

} // namespace ConformanceTests
