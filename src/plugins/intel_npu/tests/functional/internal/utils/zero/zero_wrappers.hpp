// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ze_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/llm_builders.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "vcl_serializer.hpp"
#include "ze_graph_ext_wrappers.hpp"
#include "zero_tensor.hpp"

using CompilationParams = std::string;

namespace ov {
namespace test {
namespace behavior {
class ZeroCommandListsTests : public ov::test::behavior::OVPluginTestBase,
                              public testing::WithParamInterface<CompilationParams> {
protected:
    ov::AnyMap configuration;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> _init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> _options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config _npu_config = ::intel_npu::Config(_options);

    ::intel_npu::SerializedIR serializeIR(const std::shared_ptr<ov::Model>& model) {
        auto compilerProperties = _init_struct->getCompilerProperties();
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        return ::intel_npu::driver_compiler_utils::serializeIR(model,
                                                               compilerProperties.compilerVersion,
                                                               maxOpsetVersion,
                                                               /* useBaseModelSerialize = */ true);
    }

    bool bypassUmdCache() {
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                if (configItem.first == ov::cache_dir.name()) {
                    const auto set_cache_dir = configItem.second;
                    if (!set_cache_dir.empty()) {
                        return true;
                    }
                }
                if (configItem.first == ov::intel_npu::bypass_umd_caching.name()) {
                    if (configItem.second.as<bool>()) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string target_device;
        ov::AnyMap configuration;
        target_device = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '_');
        target_device = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        target_device = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        _init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        _init_struct = nullptr;
        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroCommandListsTests, UpdateMutableCommandListPerformance) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // auto model = ov::test::utils::make_llm_kv_cache_pattern(/* batch = */ {1},
    //                                                        /* n_heads = */ {64},
    //                                                        /* n_features = */ {64},
    //                                                        /* element_type = */ ov::element::f32,
    //                                                        /* concat_axis = */ 2,
    //                                                        /* stateful = */ false,
    //                                                        /* fuse_cache_reorder = */ true,
    //                                                        /* build_state_initializer = */ false,
    //                                                        /* num_groups = */ 1,
    //                                                        /* kv_cache_trim = */ true,
    //                                                        /* kv_cache_reorder = */ true);

    auto model = ov::test::utils::make_2_input_subtract(/* ov::Shape input_shape = {1, 3, 24, 24},
                                                           ov::element::Type type = ov::element::f32 */);

    auto zeGraphExt = std::make_shared<::intel_npu::ZeGraphExtWrappers>(_init_struct);
    ::intel_npu::GraphDescriptor graphDescriptor = {};
    ::intel_npu::NetworkMetadata networkMetadata = {};
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializeIR(model), "", bypassUmdCache()));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal = ::intel_npu::zeroUtils::findCommandQueueGroupOrdinal(
                           _init_struct->getDevice(),
                           ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    OV_ASSERT_NO_THROW(networkMetadata = zeGraphExt->getNetworkMeta(graphDescriptor));

    {
        ::intel_npu::CommandList commandList(_init_struct, initCommandQueueOrdinal);
        OV_ASSERT_NO_THROW(
            commandList.appendGraphExecute(static_cast<ze_graph_handle_t>(graphDescriptor._handle), nullptr));

        uint32_t ioIndex = 0;
        std::vector<std::shared_ptr<::intel_npu::ZeroTensor>> zeroTensorPool(networkMetadata.inputs.size(), nullptr);
        for (const auto& input : networkMetadata.inputs) {
            zeroTensorPool.at(ioIndex++) =
                std::make_shared<::intel_npu::ZeroTensor>(_init_struct,
                                                          _npu_config,
                                                          input.precision,
                                                          input.shapeFromCompiler.get_shape(),
                                                          /* is_input = */ true);
        }

        size_t duration_sequential_update_mutable_command_list = 0;
        size_t duration_chained_update_mutable_command_list = 0;
        std::vector<std::pair<ze_mutable_graph_argument_exp_desc_t, std::optional<ze_graph_argument_value_strides_t>>>
            descs(zeroTensorPool.size(), std::make_pair(ze_mutable_graph_argument_exp_desc_t{}, std::nullopt));

        ioIndex = 0;
        for (const auto& tensor : zeroTensorPool) {
            auto sequential_update_mutable_command_list_time_start{std::chrono::high_resolution_clock::now()};
            commandList.updateMutableCommandList(ioIndex,
                                                 static_cast<const unsigned char*>(tensor->data()) +
                                                     (0 * tensor->get_byte_size()) / /* number_of_command_lists = */ 1);
            auto sequential_update_mutable_command_list_time_end{std::chrono::high_resolution_clock::now()};
            duration_sequential_update_mutable_command_list +=
                std::chrono::duration_cast<std::chrono::microseconds>(sequential_update_mutable_command_list_time_end -
                                                                      sequential_update_mutable_command_list_time_start)
                    .count();
            auto chained_update_mutable_command_list_time_start{std::chrono::high_resolution_clock::now()};
            commandList.updateMutableCommandList(ioIndex,
                                                 static_cast<const unsigned char*>(tensor->data()) +
                                                     (0 * tensor->get_byte_size() / /* number_of_command_lists = */ 1),
                                                 descs);
            auto chained_update_mutable_command_list_time_end{std::chrono::high_resolution_clock::now()};
            duration_chained_update_mutable_command_list +=
                std::chrono::duration_cast<std::chrono::microseconds>(chained_update_mutable_command_list_time_end -
                                                                      chained_update_mutable_command_list_time_start)
                    .count();
            ++ioIndex;
        }

        using ::intel_npu::ze_result_to_description;
        using ::intel_npu::ze_result_to_string;
        using ::intel_npu::zeCommandListUpdateMutableCommandsExp;
        auto submit_chained_update_mutable_command_list_time_start{std::chrono::high_resolution_clock::now()};
        ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC,
                                                                      &descs.at(0).first,
                                                                      0};

        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeCommandListUpdateMutableCommandsExp",
            zeCommandListUpdateMutableCommandsExp(commandList.handle(), &mutable_commands_exp_desc_t));
        auto submit_chained_update_mutable_command_list_time_end{std::chrono::high_resolution_clock::now()};
        duration_chained_update_mutable_command_list +=
            std::chrono::duration_cast<std::chrono::microseconds>(submit_chained_update_mutable_command_list_time_end -
                                                                  submit_chained_update_mutable_command_list_time_start)
                .count();

        std::cout << std::fixed << "duration_sequential_update_mutable_command_list = "
                  << duration_sequential_update_mutable_command_list << " ms" << std::endl;
        std::cout << std::fixed
                  << "duration_chained_update_mutable_command_list = " << duration_chained_update_mutable_command_list
                  << " ms" << std::endl;
    }
    zeGraphExt->destroyGraph(graphDescriptor);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
