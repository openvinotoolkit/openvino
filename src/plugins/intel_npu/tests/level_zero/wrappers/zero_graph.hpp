// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "driver_compiler_adapter.hpp"
#include "ze_graph_ext_wrappers.hpp"

using namespace intel_npu;

constexpr std::string_view INPUTS_PRECISIONS_KEY = "--inputs_precisions";
constexpr std::string_view INPUTS_LAYOUTS_KEY = "--inputs_layouts";
constexpr std::string_view OUTPUTS_PRECISIONS_KEY = "--outputs_precisions";
constexpr std::string_view OUTPUTS_LAYOUTS_KEY = "--outputs_layouts";

// <option key>="<option value>"
constexpr std::string_view KEY_VALUE_SEPARATOR = "=";
constexpr std::string_view VALUE_DELIMITER = "\"";  // marks beginning and end of value

// Format inside "<option value>"
// <name1>:<value (precision / layout)> [<name2>:<value>]
constexpr std::string_view NAME_VALUE_SEPARATOR = ":";
constexpr std::string_view VALUES_SEPARATOR = " ";

class ZeroGraphTest : public ::testing::TestWithParam<std::tuple<int, std::string>> {
protected:
    void SetUp() override;

    void TearDown() override;

public:
    std::shared_ptr<intel_npu::ZeGraphExtWrappers> zeGraphExt;

    std::unique_ptr<DriverCompilerAdapter> compilerAdapter;

    SerializedIR serializedIR;

    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::string buildFlags;

    std::string extVersion;

    int graphDescFlag;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<int, std::string>> obj) {
        int flag;
        std::string version;
        std::tie(flag, version) = obj.param;
        return "graphDescriptorFlag=" + std::to_string(flag) + "_extVersion=" + version;
    }
};

std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision);

std::string rankToLegacyLayoutString(const size_t rank);

std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices);

std::string serializeConfig(const Config& config,
                            ze_graph_compiler_version_info_t compilerVersion,
                            const std::shared_ptr<intel_npu::ZeGraphExtWrappers> zeGraphExt);
