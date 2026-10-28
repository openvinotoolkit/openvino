// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <level_zero/ze_api.h>

#include <array>
#include <optional>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

using intel_npu::CommandQueueDesc;
using intel_npu::CommandQueueParams;

namespace {

constexpr std::array<ze_command_queue_qos_ext_t, 6> allZeQosLevels = {ZE_COMMAND_QUEUE_QOS_EXT_DEFAULT,
                                                                      ZE_COMMAND_QUEUE_QOS_EXT_HIGH,
                                                                      ZE_COMMAND_QUEUE_QOS_EXT_MEDIUM,
                                                                      ZE_COMMAND_QUEUE_QOS_EXT_LOW,
                                                                      ZE_COMMAND_QUEUE_QOS_EXT_UTILITY,
                                                                      ZE_COMMAND_QUEUE_QOS_EXT_ECO};

constexpr std::array<ov::QoS, 6> allOVQosLevels =
    {ov::QoS::DEFAULT, ov::QoS::HIGH, ov::QoS::MEDIUM, ov::QoS::LOW, ov::QoS::UTILITY, ov::QoS::ECO};

CommandQueueDesc makeSharedDesc(std::optional<ze_command_queue_qos_ext_t> qos) {
    return CommandQueueDesc{ZE_COMMAND_QUEUE_PRIORITY_NORMAL, std::nullopt, qos, 0, nullptr, true};
}

intel_npu::Config makeCommandQueueConfig() {
    auto desc = std::make_shared<intel_npu::OptionsDesc>();
    desc->add<intel_npu::MODEL_PRIORITY>();
    desc->add<intel_npu::WORKLOAD_TYPE>();
    desc->add<intel_npu::QOS>();
    desc->add<intel_npu::TURBO>();
    desc->add<intel_npu::RUN_INFERENCES_SEQUENTIALLY>();
    desc->add<intel_npu::SHARED_COMMON_QUEUE>();
    return intel_npu::Config(desc);
}

}  // namespace

TEST(CommandQueueDescQos, DefaultConstructedDescHasNoQos) {
    CommandQueueDesc desc;
    EXPECT_FALSE(desc.qos().has_value());
}

TEST(CommandQueueDescQos, ConstructorStoresQos) {
    for (auto level : allZeQosLevels) {
        auto desc = makeSharedDesc(level);
        ASSERT_TRUE(desc.qos().has_value());
        EXPECT_EQ(desc.qos().value(), level);
    }

    auto desc = makeSharedDesc(std::nullopt);
    EXPECT_FALSE(desc.qos().has_value());
}

TEST(CommandQueueDescQos, ConstructorKeepsTheOtherFields) {
    int owner = 1;
    CommandQueueDesc desc{ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH,
                          ZE_WORKLOAD_TYPE_BACKGROUND,
                          ZE_COMMAND_QUEUE_QOS_EXT_LOW,
                          3,
                          &owner,
                          false};

    EXPECT_EQ(desc.priority(), ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH);
    ASSERT_TRUE(desc.workload().has_value());
    EXPECT_EQ(desc.workload().value(), ZE_WORKLOAD_TYPE_BACKGROUND);
    ASSERT_TRUE(desc.qos().has_value());
    EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_LOW);
    EXPECT_EQ(desc.options(), 3u);
    EXPECT_FALSE(desc.sharedCommonQueue());
}

TEST(CommandQueueDescQos, SameQosGivesEqualDescsAndKeys) {
    for (auto level : allZeQosLevels) {
        auto lhs = makeSharedDesc(level);
        auto rhs = makeSharedDesc(level);
        EXPECT_TRUE(lhs == rhs);
        EXPECT_EQ(lhs.key(), rhs.key());
    }

    EXPECT_TRUE(makeSharedDesc(std::nullopt) == makeSharedDesc(std::nullopt));
    EXPECT_EQ(makeSharedDesc(std::nullopt).key(), makeSharedDesc(std::nullopt).key());
}

TEST(CommandQueueDescQos, DifferentQosGivesDifferentDescsAndKeys) {
    std::vector<std::optional<ze_command_queue_qos_ext_t>> values{std::nullopt};
    values.insert(values.end(), allZeQosLevels.begin(), allZeQosLevels.end());

    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            auto lhs = makeSharedDesc(values[i]);
            auto rhs = makeSharedDesc(values[j]);
            EXPECT_FALSE(lhs == rhs) << "qos index " << i << " vs " << j;
            EXPECT_NE(lhs.key(), rhs.key()) << "qos index " << i << " vs " << j;
        }
    }
}

// An explicitly requested DEFAULT level must not be pooled together with a queue created without any QoS descriptor:
// the first one is created through ze_command_queue_qos_ext_desc_t, the second one is not.
TEST(CommandQueueDescQos, NoQosAndExplicitDefaultQosAreDistinct) {
    auto noQos = makeSharedDesc(std::nullopt);
    auto defaultQos = makeSharedDesc(ZE_COMMAND_QUEUE_QOS_EXT_DEFAULT);

    EXPECT_FALSE(noQos == defaultQos);
    EXPECT_NE(noQos.key(), defaultQos.key());
}

TEST(CommandQueueDescQos, SetQosUpdatesValueAndKey) {
    auto desc = makeSharedDesc(std::nullopt);
    const auto initialKey = desc.key();

    desc.setQos(ZE_COMMAND_QUEUE_QOS_EXT_ECO);
    ASSERT_TRUE(desc.qos().has_value());
    EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_ECO);
    EXPECT_NE(desc.key(), initialKey);

    // The descriptor updated through setQos must be indistinguishable from one created with the same level.
    auto expected = makeSharedDesc(ZE_COMMAND_QUEUE_QOS_EXT_ECO);
    EXPECT_TRUE(desc == expected);
    EXPECT_EQ(desc.key(), expected.key());
}

TEST(CommandQueueDescQos, ResettingQosRestoresTheInitialKey) {
    auto desc = makeSharedDesc(std::nullopt);
    const auto initialKey = desc.key();

    desc.setQos(ZE_COMMAND_QUEUE_QOS_EXT_HIGH);
    desc.setQos(std::nullopt);

    EXPECT_FALSE(desc.qos().has_value());
    EXPECT_EQ(desc.key(), initialKey);
    EXPECT_TRUE(desc == makeSharedDesc(std::nullopt));
}

TEST(CommandQueueDescQos, SetQosDoesNotChangeTheOtherFields) {
    int owner = 1;
    CommandQueueDesc desc{ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW,
                          ZE_WORKLOAD_TYPE_DEFAULT,
                          std::nullopt,
                          1,
                          &owner,
                          false};

    desc.setQos(ZE_COMMAND_QUEUE_QOS_EXT_UTILITY);

    EXPECT_EQ(desc.priority(), ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW);
    ASSERT_TRUE(desc.workload().has_value());
    EXPECT_EQ(desc.workload().value(), ZE_WORKLOAD_TYPE_DEFAULT);
    EXPECT_EQ(desc.options(), 1u);
    EXPECT_FALSE(desc.sharedCommonQueue());
}

// Changing the priority or the workload type must not drop the QoS level and vice versa.
TEST(CommandQueueDescQos, QosIsKeptWhenOtherFieldsChange) {
    auto desc = makeSharedDesc(ZE_COMMAND_QUEUE_QOS_EXT_MEDIUM);

    desc.setPriority(ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH);
    desc.setWorkload(ZE_WORKLOAD_TYPE_BACKGROUND);

    ASSERT_TRUE(desc.qos().has_value());
    EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_MEDIUM);

    CommandQueueDesc expected{ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH,
                              ZE_WORKLOAD_TYPE_BACKGROUND,
                              ZE_COMMAND_QUEUE_QOS_EXT_MEDIUM,
                              0,
                              nullptr,
                              true};
    EXPECT_TRUE(desc == expected);
    EXPECT_EQ(desc.key(), expected.key());
}

// With the shared common queue disabled the owner tag still separates queues, regardless of the QoS level.
TEST(CommandQueueDescQos, OwnerTagStillSeparatesNonSharedDescsWithTheSameQos) {
    int ownerA = 1;
    int ownerB = 2;
    CommandQueueDesc descA{ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                           std::nullopt,
                           ZE_COMMAND_QUEUE_QOS_EXT_LOW,
                           0,
                           &ownerA,
                           false};
    CommandQueueDesc descB{ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                           std::nullopt,
                           ZE_COMMAND_QUEUE_QOS_EXT_LOW,
                           0,
                           &ownerB,
                           false};

    EXPECT_FALSE(descA == descB);
    EXPECT_NE(descA.key(), descB.key());
}

TEST(CommandQueueParamsFromConfig, DefaultConfigMatchesTheOptionDefaults) {
    auto config = makeCommandQueueConfig();
    auto params = CommandQueueParams::from_config(config);

    EXPECT_EQ(params.modelPriority, intel_npu::MODEL_PRIORITY::defaultValue());
    EXPECT_FALSE(params.workloadType.has_value()) << "An unset workload type must not be forwarded to the queue";
    EXPECT_FALSE(params.qos.has_value()) << "An unset QoS must not be forwarded to the queue";
    EXPECT_EQ(params.turbo, intel_npu::TURBO::defaultValue());
    EXPECT_EQ(params.runInferencesSequentially, intel_npu::RUN_INFERENCES_SEQUENTIALLY::defaultValue());
    EXPECT_EQ(params.sharedCommonQueue, intel_npu::SHARED_COMMON_QUEUE::defaultValue());
}

TEST(CommandQueueParamsFromConfig, ForwardsEveryQosLevel) {
    for (auto level : allOVQosLevels) {
        auto config = makeCommandQueueConfig();
        config.update({{ov::qos.name(), ov::Any(level).as<std::string>()}});

        auto params = CommandQueueParams::from_config(config);
        ASSERT_TRUE(params.qos.has_value());
        EXPECT_EQ(params.qos.value(), level);
    }
}

TEST(CommandQueueParamsFromConfig, ForwardsAllCommandQueueOptions) {
    auto config = makeCommandQueueConfig();
    config.update({{ov::hint::model_priority.name(), "HIGH"},
                   {ov::workload_type.name(), "EFFICIENT"},
                   {ov::qos.name(), "ECO"},
                   {ov::intel_npu::turbo.name(), "YES"},
                   {ov::intel_npu::run_inferences_sequentially.name(), "YES"},
                   {ov::intel_npu::shared_common_queue.name(), "NO"}});

    auto params = CommandQueueParams::from_config(config);

    EXPECT_EQ(params.modelPriority, ov::hint::Priority::HIGH);
    ASSERT_TRUE(params.workloadType.has_value());
    EXPECT_EQ(params.workloadType.value(), ov::WorkloadType::EFFICIENT);
    ASSERT_TRUE(params.qos.has_value());
    EXPECT_EQ(params.qos.value(), ov::QoS::ECO);
    EXPECT_TRUE(params.turbo);
    EXPECT_TRUE(params.runInferencesSequentially);
    EXPECT_FALSE(params.sharedCommonQueue);
}

TEST(CommandQueueParamsFromConfig, DefaultConstructedParamsMatchTheOptionDefaults) {
    CommandQueueParams params;
    auto fromConfig = CommandQueueParams::from_config(makeCommandQueueConfig());

    EXPECT_EQ(params.modelPriority, fromConfig.modelPriority);
    EXPECT_EQ(params.workloadType, fromConfig.workloadType);
    EXPECT_EQ(params.qos, fromConfig.qos);
    EXPECT_EQ(params.turbo, fromConfig.turbo);
    EXPECT_EQ(params.runInferencesSequentially, fromConfig.runInferencesSequentially);
    EXPECT_EQ(params.sharedCommonQueue, fromConfig.sharedCommonQueue);
}

TEST(ZeroUtilsQos, ToZeQueueQosMapsEveryLevel) {
    for (size_t i = 0; i < allOVQosLevels.size(); ++i) {
        EXPECT_EQ(intel_npu::zeroUtils::toZeQueueQos(allOVQosLevels[i]), allZeQosLevels[i]);
    }
}

TEST(ZeroUtilsQos, ToOVQosMapsEveryLevel) {
    for (size_t i = 0; i < allZeQosLevels.size(); ++i) {
        EXPECT_EQ(intel_npu::zeroUtils::toOVQos(allZeQosLevels[i]), allOVQosLevels[i]);
    }
}

TEST(ZeroUtilsQos, ConversionsAreInverse) {
    for (auto level : allOVQosLevels) {
        EXPECT_EQ(intel_npu::zeroUtils::toOVQos(intel_npu::zeroUtils::toZeQueueQos(level)), level);
    }
    for (auto level : allZeQosLevels) {
        EXPECT_EQ(intel_npu::zeroUtils::toZeQueueQos(intel_npu::zeroUtils::toOVQos(level)), level);
    }
}

TEST(ZeroUtilsQos, ToOVQosRejectsUnknownLevel) {
    EXPECT_THROW(intel_npu::zeroUtils::toOVQos(static_cast<ze_command_queue_qos_ext_t>(42)), ov::Exception);
    EXPECT_THROW(intel_npu::zeroUtils::toOVQos(ZE_COMMAND_QUEUE_QOS_EXT_FORCE_UINT32), ov::Exception);
}

TEST(ZeroUtilsQos, ToZeQueueQosRejectsUnknownLevel) {
    EXPECT_THROW(intel_npu::zeroUtils::toZeQueueQos(static_cast<ov::QoS>(42)), ov::Exception);
}
