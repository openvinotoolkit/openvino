// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

#include <ie_core.hpp>

typedef std::tuple<
    std::vector<size_t>,                // Input Shape
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Export Configuration
    std::map<std::string, std::string>, // Import Configuration
    std::string                         // Application Header
> exportImportNetworkParams;

namespace FuncTestUtils {

class ImportNetworkTestBase : public testing::WithParamInterface<exportImportNetworkParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj);
    void Run() override;
    void TestRun(bool isModelChanged);

protected:
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::string                        applicationHeader;

private:
    virtual void exportImportNetwork();
};

} // namespace FuncTestUtils
