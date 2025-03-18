// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;
using ConfigParams = std::tuple<std::string,  // Priority devices
                                bool          // if throw exception
                                >;
class IsMetaDeviceInCandidateListTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string priorityDevices;
        bool expectedRet;
        std::tie(priorityDevices, expectedRet) = obj.param;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        if (expectedRet) {
            result << "_expection_true";
        } else {
            result << "_expection_false";
        }
        return result.str();
    }

    void SetUp() override {
        ON_CALL(*plugin, is_meta_device).WillByDefault([this](const std::string& priorityDevices) {
            return plugin->Plugin::is_meta_device(priorityDevices);
        });
        std::tie(priorityDevices, expectedRet) = GetParam();
    }

protected:
    // get Parameter
    std::string priorityDevices;
    bool expectedRet;
};

TEST_P(IsMetaDeviceInCandidateListTest, CheckMetaDeviceWithPriority) {
    EXPECT_CALL(*plugin, is_meta_device(_)).Times(1);
    ASSERT_EQ(plugin->is_meta_device(priorityDevices), expectedRet);
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{"CPU", false},
                                               ConfigParams{"GPU", false},
                                               ConfigParams{"OTHER", false},
                                               ConfigParams{"GPU.0", false},
                                               ConfigParams{"GPU.1", false},
                                               ConfigParams{"GPU.AUTO_DETECT", false},
                                               ConfigParams{"GPU.MULTI_DETECT", false},
                                               ConfigParams{"OTHER.AUTO_DETECT", false},
                                               ConfigParams{"CPU,OTHER.AUTO_DETECT", false},
                                               ConfigParams{"GPU,OTHER.AUTO_DETECT", false},
                                               ConfigParams{"GPU,OTHER.MULTI_DETECT", false},
                                               ConfigParams{"AUTO", true},
                                               ConfigParams{"AUTO,CPU", true},
                                               ConfigParams{"OTHER,AUTO", true},
                                               ConfigParams{"AUTO:CPU", true},
                                               ConfigParams{"AUTO:OTHER", true},
                                               ConfigParams{"AUTO.AUTO_DETECT", true},
                                               ConfigParams{"AUTO:AUTO.AUTO_DETECT", true},
                                               ConfigParams{"CPU,AUTO:AUTO.AUTO_DETECT", true},
                                               ConfigParams{"MULTI", true},
                                               ConfigParams{"CPU,MULTI", true},
                                               ConfigParams{"CPU,MULTI.0", true},
                                               ConfigParams{"MULTI:CPU", true},
                                               ConfigParams{"GPU,MULTI:CPU", true},
                                               ConfigParams{"MULTI:OTHER", true},
                                               ConfigParams{"MULTI.AUTO_DETECT", true},
                                               ConfigParams{"MULTI:AUTO.AUTO_DETECT", true},
                                               ConfigParams{"GPU,MULTI:AUTO.AUTO_DETECT", true},
                                               ConfigParams{"GPU,MULTI:MULTI.AUTO_DETECT", true}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         IsMetaDeviceInCandidateListTest,
                         ::testing::ValuesIn(testConfigs),
                         IsMetaDeviceInCandidateListTest::getTestCaseName);
