// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ConformanceTests {

extern const char* targetDevice;
extern std::vector<std::string> IRFolderPaths;

class ConformanceTest : public testing::WithParamInterface<std::tuple<std::string, std::string>>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>> &obj);
protected:
    std::string pathToModel;

    void LoadNetwork() override;
    void SetUp() override;
};
} // namespace ConformanceTests
