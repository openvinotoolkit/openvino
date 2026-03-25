// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common/functions.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "compiler_schedules_sections.hpp"
#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/static_capability.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {
// duplicate from tests/functional/behavior/weights_separation.hpp
// should we move it to common/utils?
std::shared_ptr<ov::Model> createTestModel(const bool addWeightlessCacheAttribute = true,
                                           const bool alternativeWeights = false) {
    using namespace ov;

    constexpr auto precision = element::f32;
    const float weightsValue = !alternativeWeights ? 1.0f : 2.0f;
    auto weights = std::make_shared<op::v0::Constant>(precision, Shape{5}, std::vector<float>{weightsValue});
    auto input = std::make_shared<op::v0::Parameter>(precision, Shape{1});
    auto add = std::make_shared<op::v1::Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    if (addWeightlessCacheAttribute) {
        weights->get_rt_info()[WeightlessCacheAttribute::get_type_info_static()] =
            WeightlessCacheAttribute(weights->get_byte_size(), 0, weights->get_element_type());
    }

    auto model = std::make_shared<ov::Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
    ov::util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
    return model;
}
}  // namespace

namespace ov::test::behavior {
using namespace ::intel_npu;

class ELFSchedulesSections : public OVPluginTestBase,
                             public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    // maybe we can use SetUpTestSuite() for the compilation part
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();

        // do we care about the contents of the model? Or about its size?
        std::shared_ptr<ov::Model> model;
        if (configuration.count(ov::intel_npu::weightless_blob.name()) &&
            configuration.at(ov::intel_npu::weightless_blob.name()).as<bool>()) {
            model = createTestModel(true);
        } else {
            model = buildSingleLayerSoftMaxNetwork();
        }

        CompiledModel compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, target_device, configuration));

        std::stringstream stream;
        compiled_model.export_model(stream);

        auto blobSize = BlobReader::get_npu_region_size(stream);
        blob = ov::Tensor(ov::element::u8, ov::Shape{blobSize});
        stream.read(blob.data<char>(), static_cast<std::streamsize>(blobSize));

        reader = std::make_unique<BlobReader>(blob);
        reader->register_reader(PredefinedSectionType::ELF_MAIN_SCHEDULE, ELFMainScheduleSection::read);
        reader->register_reader(PredefinedSectionType::ELF_INIT_SCHEDULES, ELFInitSchedulesSection::read);

        std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
        for (auto token : CRE::DEFAULT_PLUGIN_CAPABILITIES_TOKENS) {
            caps[token] = std::make_shared<StaticCapability>(token);
        }
        reader->read(caps);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap configuration;
    // should I use cached Core instead?
    ov::Core core;
    ov::Tensor blob;
    std::unique_ptr<BlobReader> reader;
};

TEST_P(ELFSchedulesSections, MainScheduleSectionPresent) {
    auto section = reader->retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE);
    ASSERT_NE(section, nullptr);
}

TEST_P(ELFSchedulesSections, MainScheduleSectionNonEmpty) {
    auto section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
        reader->retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE));
    ASSERT_NE(section, nullptr);
    EXPECT_GT(section->get_schedule().get_byte_size(), 0);
}

TEST_P(ELFSchedulesSections, MainScheduleDataPageAligned) {
    auto section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
        reader->retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE));
    ASSERT_NE(section, nullptr);

    auto* blob_begin = static_cast<const uint8_t*>(blob.data());
    auto* schedule_ptr = static_cast<const uint8_t*>(section->get_schedule().data());
    auto offset_within_blob = static_cast<size_t>(schedule_ptr - blob_begin);
    EXPECT_EQ(offset_within_blob % ::intel_npu::utils::STANDARD_PAGE_SIZE, 0);
}

using ELFSchedulesWeightsSeparation = ELFSchedulesSections;

TEST_P(ELFSchedulesWeightsSeparation, InitSchedulesSectionPresent) {
    auto section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
        reader->retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));
    ASSERT_NE(section, nullptr);

    auto schedules = section->get_schedules();
    ASSERT_GT(schedules.size(), 0);
    for (const auto& schedule : schedules) {
        EXPECT_GT(schedule.get_byte_size(), 0);
    }
}

TEST_P(ELFSchedulesWeightsSeparation, InitSchedulesDataPageAligned) {
    auto section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
        reader->retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));
    ASSERT_NE(section, nullptr);
    auto schedules = section->get_schedules();
    ASSERT_GT(schedules.size(), 0);

    auto* blob_begin = static_cast<const uint8_t*>(blob.data());
    auto* schedule_ptr = static_cast<const uint8_t*>(schedules.front().data());
    auto offset_within_blob = static_cast<size_t>(schedule_ptr - blob_begin);
    EXPECT_EQ(offset_within_blob % ::intel_npu::utils::STANDARD_PAGE_SIZE, 0);
}

// not sure if this test makes sense to be kept
TEST_P(ELFSchedulesWeightsSeparation, InitSchedulesContainedInsideBlob) {
    auto section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
        reader->retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));
    ASSERT_NE(section, nullptr);

    auto* blob_begin = static_cast<const uint8_t*>(blob.data());
    auto* blob_end = blob_begin + blob.get_byte_size();
    for (const auto& schedule : section->get_schedules()) {
        auto* data = static_cast<const uint8_t*>(schedule.data());
        EXPECT_GE(data, blob_begin);
        EXPECT_LT(data, blob_end);
    }
}

using ELFSchedulesDriver = ELFSchedulesSections;

TEST_P(ELFSchedulesDriver, InitSchedulesSectionAbsent) {
    auto section = reader->retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES);
    EXPECT_EQ(section, nullptr);
}

}  // namespace ov::test::behavior
