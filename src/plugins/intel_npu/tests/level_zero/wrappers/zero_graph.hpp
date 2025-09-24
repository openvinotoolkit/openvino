// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/npu_test_env_cfg.hpp"

using namespace intel_npu;

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
        std::string targetDevice = ov::test::utils::DEVICE_NPU;

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "graphDescriptorFlag=" + std::to_string(flag) << "_";
        result << "extVersion=" + version;
        return result.str();
    }
};

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion);

void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes);

void* allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const size_t bytes, const size_t alignment) noexcept;

void deallocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, void* handle) noexcept;
