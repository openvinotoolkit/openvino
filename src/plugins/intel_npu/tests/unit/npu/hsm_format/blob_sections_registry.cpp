// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_sections_registry.hpp"

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_version_section.hpp"
#include "intel_npu/common/runtime_requirements.hpp"
#include "utils.hpp"

using namespace intel_npu;
using testing::_;

namespace {

const auto BATCH_SIZE_PTR = std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE);

}

using BlobSectionsRegistryTest = testing::Test;

TEST_F(BlobSectionsRegistryTest, EmptyRegistryGetters) {
    BlobSectionsRegistry registry;
    ASSERT_EQ(registry.retrieve_section(0), nullptr);
    ASSERT_EQ(registry.retrieve_unique_section_of_type(SectionTypeCode::BATCH_SIZE), nullptr);
    ASSERT_TRUE(registry.retrieve_sections(SectionTypeCode::BATCH_SIZE).empty());
    ASSERT_TRUE(registry.get_write_queue().empty());
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::BATCH_SIZE), 0);
    ASSERT_TRUE(registry.get_content_summary().empty());
}

TEST_F(BlobSectionsRegistryTest, RegisterNullSection) {
    OV_EXPECT_THROW(BlobSectionsRegistry().register_section(nullptr), ov::Exception, _);
}

TEST_F(BlobSectionsRegistryTest, RegisterSameSectionTwice) {
    BlobSectionsRegistry registry;
    const auto section = std::make_shared<BatchSizeSection>(0);
    OV_ASSERT_NO_THROW(registry.register_section(section));
    OV_EXPECT_THROW(registry.register_section(section), ov::Exception, _);
}

TEST_F(BlobSectionsRegistryTest, RegisterSectionThatHasID) {
    BlobSectionsRegistry registry1, registry2;
    const auto section = std::make_shared<BatchSizeSection>(0);
    OV_ASSERT_NO_THROW(registry1.register_section(section));
    OV_EXPECT_THROW(registry2.register_section(section), ov::Exception, _);
}

TEST_F(BlobSectionsRegistryTest, RegisterRuntimeRequirements) {
    BlobSectionsRegistry registry;
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    OV_EXPECT_THROW(registry.register_section(std::make_shared<RuntimeRequirementsSection>(runtime_requirements)),
                    ov::Exception,
                    _);
}

TEST_F(BlobSectionsRegistryTest, SectionsOverflow) {
    BlobSectionsRegistry registry;

    for (size_t i = 0; i < std::numeric_limits<uint16_t>::max(); ++i) {
        OV_ASSERT_NO_THROW(registry.register_section(std::make_shared<BatchSizeSection>(0)));
    }
    OV_EXPECT_THROW(registry.register_section(std::make_shared<BatchSizeSection>(0)), ov::Exception, _);
}

TEST_F(BlobSectionsRegistryTest, DuplicateIdAsArgument) {
    BlobSectionsRegistry registry;
    OV_ASSERT_NO_THROW(registry.register_section(std::make_shared<BatchSizeSection>(0), 0));
    OV_EXPECT_THROW(registry.register_section(std::make_shared<BatchSizeSection>(0), 0), ov::Exception, _);
}

TEST_F(BlobSectionsRegistryTest, ArgumentIdLeadsToOverflow) {
    BlobSectionsRegistry registry;
    OV_EXPECT_THROW(
        registry.register_section(std::make_shared<BatchSizeSection>(0), std::numeric_limits<uint16_t>::max()),
        ov::Exception,
        _);
}

TEST_F(BlobSectionsRegistryTest, ArgumentIdIncreasesAttributeId) {
    BlobSectionsRegistry registry;
    registry.register_section(std::make_shared<BatchSizeSection>(0), 1);
    const auto section = std::make_shared<BatchSizeSection>(0);
    registry.register_section(section);
    ASSERT_TRUE(section->get_id().has_value());
    ASSERT_EQ(section->get_id()->get_id(), 2);
}

TEST_F(BlobSectionsRegistryTest, NormalRegistryGetters) {
    BlobSectionsRegistry registry;
    const auto section_1 = std::make_shared<BatchSizeSection>(0);
    const auto section_2 = std::make_shared<BatchSizeSection>(1);
    const auto section_3 = std::make_shared<CompilerVersionSection>(2);
    registry.register_section(section_1);
    registry.register_section(section_2);
    registry.register_section(section_3);

    const auto retrieved_section_1 = registry.retrieve_section(0);
    const auto casted_section_1 = std::dynamic_pointer_cast<BatchSizeSection>(retrieved_section_1);
    ASSERT_TRUE(casted_section_1);
    ASSERT_EQ(casted_section_1->get_batch_size(), 0);
    ASSERT_TRUE(casted_section_1->get_id().has_value());
    ASSERT_EQ(casted_section_1->get_id()->get_id(), 0);

    const auto retrieved_section_2 = registry.retrieve_section(1);
    const auto casted_section_2 = std::dynamic_pointer_cast<BatchSizeSection>(retrieved_section_2);
    ASSERT_TRUE(casted_section_2);
    ASSERT_EQ(casted_section_2->get_batch_size(), 1);
    ASSERT_TRUE(casted_section_2->get_id().has_value());
    ASSERT_EQ(casted_section_2->get_id()->get_id(), 1);

    const auto retrieved_section_3 = registry.retrieve_section(2);
    const auto casted_section_3 = std::dynamic_pointer_cast<CompilerVersionSection>(retrieved_section_3);
    ASSERT_TRUE(casted_section_3);
    ASSERT_EQ(casted_section_3->get_compiler_version(), 2);
    ASSERT_TRUE(casted_section_3->get_id().has_value());
    ASSERT_EQ(casted_section_3->get_id()->get_id(), 2);

    ASSERT_EQ(registry.retrieve_section(3), nullptr);

    auto retrieved_section = registry.retrieve_unique_section_of_type(SectionTypeCode::BATCH_SIZE);
    ASSERT_TRUE(retrieved_section == retrieved_section_1 || retrieved_section == retrieved_section_2);
    ASSERT_EQ(registry.retrieve_unique_section_of_type(SectionTypeCode::IO_LAYOUTS), nullptr);

    const std::vector<std::shared_ptr<ISection>> retrieved_sections =
        registry.retrieve_sections(SectionTypeCode::BATCH_SIZE);
    ASSERT_EQ(retrieved_sections.size(), 2);
    ASSERT_TRUE(retrieved_sections.at(0) == retrieved_section_1 || retrieved_sections.at(0) == retrieved_section_2);
    ASSERT_TRUE(retrieved_sections.at(1) == retrieved_section_1 || retrieved_sections.at(1) == retrieved_section_2);
    ASSERT_TRUE(registry.retrieve_sections(SectionTypeCode::IO_LAYOUTS).empty());

    std::queue<std::shared_ptr<ISection>> write_queue = registry.get_write_queue();
    ASSERT_EQ(write_queue.size(), 3);
    retrieved_section = write_queue.front();
    ASSERT_EQ(retrieved_section, retrieved_section_1);
    write_queue.pop();
    retrieved_section = write_queue.front();
    ASSERT_EQ(retrieved_section, retrieved_section_2);
    write_queue.pop();
    retrieved_section = write_queue.front();
    ASSERT_EQ(retrieved_section, retrieved_section_3);

    ASSERT_TRUE(registry.has_section_of_type(SectionTypeCode::BATCH_SIZE));
    ASSERT_TRUE(registry.has_section_of_type(SectionTypeCode::COMPILER_VERSION));
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::IO_LAYOUTS));

    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::BATCH_SIZE), 2);
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::COMPILER_VERSION), 1);
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::IO_LAYOUTS), 0);

    const std::unordered_map<SectionType, size_t> summary = registry.get_content_summary();
    ASSERT_EQ(summary.size(), 2);
    ASSERT_TRUE(summary.count(SectionTypeCode::BATCH_SIZE));
    ASSERT_TRUE(summary.count(SectionTypeCode::COMPILER_VERSION));
    ASSERT_FALSE(summary.count(SectionTypeCode::IO_LAYOUTS));
    ASSERT_EQ(summary.at(SectionTypeCode::BATCH_SIZE), 2);
    ASSERT_EQ(summary.at(SectionTypeCode::COMPILER_VERSION), 1);
}

TEST_F(BlobSectionsRegistryTest, EmptyRegistryRuntimeRequirements) {
    BlobSectionsRegistry registry;
    const auto section = registry.build_runtime_requirements_section();
    ASSERT_TRUE(section);
    const auto casted_section = std::dynamic_pointer_cast<RuntimeRequirementsSection>(section);
    ASSERT_TRUE(casted_section);
    const RuntimeRequirements runtime_requirements = casted_section->get_runtime_requirements();
    ASSERT_TRUE(runtime_requirements.get_sections_requirements().empty());
    ASSERT_TRUE(runtime_requirements.get_section_id_to_type_mapping().empty());
    ASSERT_EQ(runtime_requirements.get_cre().get_expression_length(), 0);
}

TEST_F(BlobSectionsRegistryTest, NormalRegistryRuntimeRequirements) {
    BlobSectionsRegistry registry;
    const std::vector<std::shared_ptr<CREToken>> mock_tokens{std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS),
                                                             CRE::AND_PTR,
                                                             std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS),
                                                             std::make_shared<SectionID>(2)};

    registry.register_section(std::make_shared<BatchSizeSection>(0));
    registry.register_section(std::make_shared<CompilerVersionSection>(0));
    registry.register_section(
        std::make_shared<MockSection>(mock_tokens, std::string(STRING_THAT_EVALUATES_TO_SUPPORTED)));

    const auto section = registry.build_runtime_requirements_section();
    ASSERT_TRUE(section);
    const auto casted_section = std::dynamic_pointer_cast<RuntimeRequirementsSection>(section);
    ASSERT_TRUE(casted_section);
    const RuntimeRequirements runtime_requirements = casted_section->get_runtime_requirements();

    const std::map<SectionID, std::string> sections_requirements = runtime_requirements.get_sections_requirements();
    ASSERT_EQ(sections_requirements.size(), 1);
    ASSERT_TRUE(sections_requirements.count(2));
    ASSERT_EQ(sections_requirements.at(2), STRING_THAT_EVALUATES_TO_SUPPORTED.data());

    const std::unordered_map<SectionID, SectionType> id_to_type = runtime_requirements.get_section_id_to_type_mapping();
    ASSERT_EQ(id_to_type.size(), 3);
    ASSERT_TRUE(id_to_type.count(0));
    ASSERT_TRUE(id_to_type.count(1));
    ASSERT_TRUE(id_to_type.count(2));
    ASSERT_EQ(id_to_type.at(0), SectionType(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(id_to_type.at(1), SectionType(SectionTypeCode::COMPILER_VERSION));
    ASSERT_EQ(id_to_type.at(2), SectionType(SectionTypeCode::IO_LAYOUTS));

    const CRE result_cre = runtime_requirements.get_cre();
    CRE reference_cre_1{{BATCH_SIZE_PTR}};
    reference_cre_1.append_to_expression(mock_tokens);
    CRE reference_cre_2{mock_tokens};
    reference_cre_2.append_to_expression(std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_PTR});
    // The order may differ since the result depends on an unordered map
    ASSERT_TRUE(result_cre == reference_cre_1 || result_cre == reference_cre_2);
}
