// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ::testing;

static std::string getDeviceStringWithMulti(std::vector<std::string> names, bool flag = true) {
    std::string allDevices = flag ? "MULTI:" : "AUTO:";
    for (auto && device : names) {
        allDevices += device;
        allDevices += ((device == names[names.size()-1]) ? "" : ",");
    }
    return allDevices;
}
using DeviceName = std::string;
using DevicesNames = std::vector<DeviceName>;
using DevicesNamesAndSupportPair = std::pair<DevicesNames, bool>;
using DevicesNamesMultiCumuTuple = std::tuple<DevicesNames, bool, bool>;

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

class MultiDeviceMultipleGPU_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<DevicesNamesMultiCumuTuple> {
    void SetUp() override {
        auto param = this->GetParam();
        device_names = getDeviceStringWithMulti(std::get<0>(param), std::get<1>(param));
        device_lists = std::get<0>(param);
        auto fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice("");
        using namespace ov::preprocess;
        auto p = PrePostProcessor(fn_ptr);
        p.input().tensor().set_element_type(ov::element::i8);
        p.input().preprocess().convert_element_type(ov::element::f32);

        function = p.build();
        std::string prioritylist;
        std::vector<std::string>::iterator iter;
        for (iter = device_lists.begin(); iter != device_lists.end();) {
            prioritylist += *iter;
            prioritylist += ((*iter == device_lists[device_lists.size() - 1]) ? "" : ",");
            // remove CPU from context candidate list
            if ((*iter).find("CPU") != std::string::npos)
                device_lists.erase(iter);
            else
                iter++;
        }
        config = {ov::device::priorities(prioritylist)};
        cumu_enabled = std::get<2>(param);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DevicesNamesMultiCumuTuple> &obj) {
        auto param = obj.param;
        auto s = getDeviceStringWithMulti(std::get<0>(param));
        std::replace(s.begin(), s.end(), ',', '_');
        std::ostringstream ss;
        ss << "device_names_" << s << "_" << (std::get<1>(param) ? "MULTI" : "AUTO") << "_";
        if (std::get<2>(param))
            ss << "cumulative_yes";
        else
            ss << "cumulative_no";
        return ss.str();
    }

protected:
    std::string device_names;
    std::vector<std::string> device_lists;
    std::shared_ptr<ov::Model> function;
    ov::AnyMap config;
    bool cumu_enabled;
};

using MultiDeviceCreateContextMultipleGPU_Test = MultiDeviceMultipleGPU_Test;

#define MULTI  CommonTestUtils::DEVICE_MULTI
#define CPU    CommonTestUtils::DEVICE_CPU
#define GPU    CommonTestUtils::DEVICE_GPU
