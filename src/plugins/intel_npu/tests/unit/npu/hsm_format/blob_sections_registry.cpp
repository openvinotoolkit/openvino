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

namespace {}

using BlobSectionsRegistryTest = testing::Test;

// std::shared_ptr<ISection> retrieve_section(const SectionID& id) const;

//     std::shared_ptr<ISection> retrieve_any_section(const SectionType section_type) const;

//     std::vector<std::shared_ptr<ISection>> retrieve_sections(const SectionType type) const;

//     std::queue<std::shared_ptr<ISection>> get_write_queue() const;

//     bool has_section_of_type(const SectionType section_type) const;
//     size_t count_sections_of_type(const SectionType section_type) const;
//     std::unordered_map<SectionType, size_t> get_content_summary() const;

TEST_F(BlobSectionsRegistryTest, EmptyRegistryGetters) {
    BlobSectionsRegistry registry;
    ASSERT_EQ(registry.retrieve_section(0), nullptr);
    ASSERT_EQ(registry.retrieve_any_section(SectionTypeCode::BATCH_SIZE), nullptr);
    ASSERT_TRUE(registry.retrieve_sections(SectionTypeCode::BATCH_SIZE).empty());
    ASSERT_TRUE(registry.get_write_queue().empty());
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::BATCH_SIZE), 0);
    ASSERT_TRUE(registry.get_content_summary().empty());
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
}
