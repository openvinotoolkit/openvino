// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

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
    std::shared_ptr<ZeroInitStructsMock> zeroInitMock;
    
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;

    std::shared_ptr<intel_npu::ZeGraphExtWrappers> zeGraphExt;

    SerializedIR serializedIR;

    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::string extVersion;

    int graphDescFlag;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<int, std::string>> obj) {
        int flag;
        std::string version;
        std::tie(flag, version) = obj.param;
        return "graphDescriptorFlag=" + std::to_string(flag) + "_extVersion=" + version;
    }
};

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion);

void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes);

void* allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const size_t bytes, const size_t alignment) noexcept;

void deallocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, void* handle) noexcept;
