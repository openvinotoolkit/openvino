// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <openvino/runtime/core.hpp>

#include "common_test_utils/unicode_utils.hpp"

std::string getBackendName(const ov::Core& core);

std::vector<std::string> getAvailableDevices(const ov::Core& core);

std::string modelPriorityToString(const ov::hint::Priority priority);

std::string removeDeviceNameOnlyID(const std::string& device_name_id);

std::vector<ov::AnyMap> getRWMandatoryPropertiesValues(std::vector<ov::AnyMap> props);

std::shared_ptr<ov::Model> createModelWithStates(ov::element::Type type, const ov::Shape& shape);

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
void removeDirFilesRecursive(const std::basic_string<C>& path) {
    if (!ov::util::directory_exists(path)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        ov::test::utils::removeFile(entry.path().generic_string<C>());
    }
    ov::test::utils::removeDir(path);
    // EISW-105043: [Linux] [Bug] Cannot delete loaded shared libraries unicode directories
    // `Directory not empty` throw on linux for code below
    // std::filesystem::remove_all(path);
}

template <typename T>
std::string vectorToString(std::vector<T> v) {
    std::ostringstream res;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            res << ",";
        } else {
            res << "{";
        }
        res << v[i];
    }
    res << "}";
    return res.str();
}

// This utility class comes handy when OpenVino doesn't provide expicit `getTestCaseName`
// functions for declared test classes, but on NPU plugin side, we still need to append
// `_targetPlatform=NPUXXXX` to test name for activation of platform specific tests associated with test classes.
// Stands as a wrapper on `getTestCaseName` functions if declared and implements generic test name functions otherwise
// Programmer should NOT use this explicitly as it is part of `appendPlatformTypeTestName` utility.

struct GenericTestCaseNameClass {
    template <typename, typename = void>
    static constexpr bool hasGetTestCaseName = false;

    template <typename T>
    static std::string getTestCaseName(testing::TestParamInfo<typename T::ParamType>& obj) {
        if constexpr (hasGetTestCaseName<T>) {
            return T::getTestCaseName(obj);
        } else {
            std::ostringstream result;
            ::testing::PrintToStringParamName printToStringParamName;
            result << printToStringParamName(obj);
            return result.str();
        }
    }
};

template <typename T>
constexpr bool
    GenericTestCaseNameClass::hasGetTestCaseName<T,
                                                 std::void_t<decltype(std::declval<T>().getTestCaseName(
                                                     std::declval<testing::TestParamInfo<typename T::ParamType>>()))>> =
        true;
