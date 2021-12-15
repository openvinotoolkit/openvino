// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

using namespace ::testing;

static std::string getDeviceStringWithMulti(std::vector<std::string> names) {
    std::string allDevices = "MULTI:";
    for (auto && device : names) {
        allDevices += device;
        allDevices += ((device == names[names.size()-1]) ? "" : ",");
    }
    return allDevices;
}
using DeviceName = std::string;
using DevicesNames = std::vector<DeviceName>;
using DevicesNamesAndSupportPair = std::pair<DevicesNames, bool>;

class MultiDevice_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<DevicesNames> {
    void SetUp() override {
        device_names = getDeviceStringWithMulti(this->GetParam());
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNames> &obj) {
        auto s = getDeviceStringWithMulti(obj.param);
        std::replace(s.begin(), s.end(), ',', '_');
        return "device_names_" + s;
    }
protected:
    std::string device_names;
    std::shared_ptr<ngraph::Function> fn_ptr;
};

class MultiDevice_SupportTest : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<DevicesNamesAndSupportPair> {
    void SetUp() override {
        device_names = getDeviceStringWithMulti(this->GetParam().first);
        expected_status = this->GetParam().second;
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNamesAndSupportPair> &obj) {
        auto s = getDeviceStringWithMulti(obj.param.first);
        std::replace(s.begin(), s.end(), ',', '_');
        return "device_names_" + s;
    }
protected:
    std::string device_names;
    bool expected_status;
    std::shared_ptr<ngraph::Function> fn_ptr;
};
#define MULTI  CommonTestUtils::DEVICE_MULTI
#define CPU    CommonTestUtils::DEVICE_CPU
#define GPU    CommonTestUtils::DEVICE_GPU
#define MYRIAD CommonTestUtils::DEVICE_MYRIAD
