// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ze_graph_ext_wrappers.hpp"

using namespace intel_npu;

class ZeroGraphTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    void SetUp() override;

    void TearDown() override;

public:
    void serializeIR();

    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;

    std::shared_ptr<intel_npu::ZeGraphExtWrappers> zeGraphExt;

    SerializedIR serializedIR;

    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::shared_ptr<driver_compiler_utils::IRSerializer> irSerializer;

    int extVersion;

    int graphDescFlag;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<int, int>> obj) {
        int flag, version;
        std::tie(flag, version) = obj.param;
        std::string targetDevice = ov::test::utils::DEVICE_NPU;

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "graphDescriptorFlag=" + std::to_string(flag) << "_";
        result << "extVersion=" + std::to_string(ZE_MAJOR_VERSION(version)) + "." +
                      std::to_string(ZE_MINOR_VERSION(version));
        return result.str();
    }
};

void* allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                           const size_t bytes,
                           const size_t alignment) noexcept;

void deallocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, void* handle) noexcept;
