// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/util/common_util.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"

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
using DevicesNamesAndSupportTuple = std::tuple<DevicesNames, bool, ov::AnyMap>;
using DevicesNamseAndProperties = std::pair<DevicesNames, ov::AnyMap>;

class MultiDevice_Test : public ov::test::TestsCommon, public testing::WithParamInterface<DevicesNamseAndProperties> {
    void SetUp() override {
        std::vector<DeviceName> deviceNameList;
        std::tie(deviceNameList, _properties) = this->GetParam();
        device_names = getDeviceStringWithMulti(deviceNameList);
        fn_ptr = ov::test::utils::make_split_multi_conv_concat();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNamseAndProperties>& obj) {
        auto s = getDeviceStringWithMulti(obj.param.first);
        ov::AnyMap properties = obj.param.second;
        std::replace(s.begin(), s.end(), ',', '_');
        std::replace(s.begin(), s.end(), ':', '_');
        std::ostringstream result;
        result << "device_names_" << s << "_";
        if (!properties.empty()) {
            result << "properties=" << ov::util::join(ov::util::split(ov::util::to_string(properties), ' '), "_");
        } else {
            result << "no_property";
        }
        return result.str();
    }

protected:
    std::string device_names;
    ov::AnyMap _properties;
    std::shared_ptr<ov::Model> fn_ptr;
};

class MultiDevice_SupportTest : public ov::test::TestsCommon, public testing::WithParamInterface<DevicesNamesAndSupportTuple> {
    void SetUp() override {
        std::vector<DeviceName> deviceNameList;
        std::tie(deviceNameList, expected_status, _properties) = this->GetParam();
        device_names = getDeviceStringWithMulti(deviceNameList);
        fn_ptr = ov::test::utils::make_split_multi_conv_concat();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNamesAndSupportTuple>& obj) {
        auto s = getDeviceStringWithMulti(std::get<0>(obj.param));
        std::string expected_status = std::get<1>(obj.param) == true ? "expect_TRUE" : "expect_FALSE";
        ov::AnyMap properties = std::get<2>(obj.param);
        std::replace(s.begin(), s.end(), ',', '_');
        std::replace(s.begin(), s.end(), ':', '_');
        std::ostringstream result;
        result << "device_names_" << s << "_" << expected_status << "_";
        if (!properties.empty()) {
            result << "properties=" << ov::util::join(ov::util::split(ov::util::to_string(properties), ' '), "_");
        } else {
            result << "no_property";
        }
        return result.str();
    }

protected:
    std::string device_names;
    bool expected_status;
    ov::AnyMap _properties;
    std::shared_ptr<ov::Model> fn_ptr;
};

class MultiDeviceMultipleGPU_Test : public ov::test::TestsCommon, public testing::WithParamInterface<DevicesNames> {
    void SetUp() override {
        device_names = getDeviceStringWithMulti(this->GetParam());
        device_lists = this->GetParam();
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    }
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNames> &obj) {
        auto s = getDeviceStringWithMulti(obj.param);
        std::replace(s.begin(), s.end(), ',', '_');
        std::replace(s.begin(), s.end(), ':', '_');
        return "device_names_" + s;
    }
protected:
    std::string device_names;
    std::vector<std::string> device_lists;
    std::shared_ptr<ov::Model> fn_ptr;
};
#define MULTI  ov::test::utils::DEVICE_MULTI
#define CPU    ov::test::utils::DEVICE_CPU
#define GPU    ov::test::utils::DEVICE_GPU
