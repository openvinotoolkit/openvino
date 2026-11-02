// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_version_section.hpp"
#include "hsm_read_write.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

using namespace intel_npu;
using testing::_;

using ReadParams = std::
    tuple<bool, bool, std::string, CRE, std::map<SectionID, std::string>, std::unordered_map<SectionID, SectionType>>;

namespace {

const auto MOCK_INSTANCE_EVALUATOR = std::make_shared<MockInstanceEvaluator>();
const auto MOCK_TYPE_EVALUATOR_TRUE = std::make_shared<MockTypeEvaluator>(true);
const auto MOCK_TYPE_EVALUATOR_FALSE = std::make_shared<MockTypeEvaluator>(false);

const auto ELF_MAIN_SCHEDULE_PTR = std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE);
const auto BATCH_SIZE_PTR = std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE);
const auto COMPILER_VERSION_PTR = std::make_shared<SectionType>(SectionTypeCode::COMPILER_VERSION);
const auto ID_0_PTR = std::make_shared<SectionID>(0);
const auto ID_2_PTR = std::make_shared<SectionID>(2);

}  // namespace

// write_to(stream, const BlobSectionsRegistry& blob_sections_registry);
// read_from(
//     ov::runtime::hsm::Reader& base_reader,
//     const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>&
//     section_readers, const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
//     const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators)

class HSMReadWriteTest : public testing::TestWithParam<bool> {
public:
    HSMReadWriteTest() : section({}) {}

    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        bool tensor_source = obj.param;

        return std::string(tensor_source ? "tensor" : "stream");
    }

protected:
    void SetUp() override {
        batch_size_section_1 = std::make_shared<BatchSizeSection>(0);
        batch_size_section_2 = std::make_shared<BatchSizeSection>(1);
        compiler_version_section = std::make_shared<CompilerVersionSection>(2);
        sections_registry.register_section(batch_size_section_1);
        sections_registry.register_section(compiler_version_section);
        sections_registry.register_section(batch_size_section_2);

        section_readers[SectionTypeCode::BATCH_SIZE] = BatchSizeSection::decode;
        section_readers[SectionTypeCode::COMPILER_VERSION] = CompilerVersionSection::decode;
    }

    std::stringstream stream;
    BlobSectionsRegistry sections_registry;
    std::shared_ptr<BatchSizeSection> batch_size_section_1;
    std::shared_ptr<BatchSizeSection> batch_size_section_2;
    std::shared_ptr<CompilerVersionSection> compiler_version_section;
    std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>> section_readers;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators;
    std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators;
};

TEST_P(RuntimeRequirementsSectionReadTest, Read) {
    auto read_section = RuntimeRequirementsSection::decode(section);

    if (!expect_failure) {
        ASSERT_TRUE(read_section);
        auto casted_section = std::dynamic_pointer_cast<RuntimeRequirementsSection>(read_section.value());
        ASSERT_TRUE(casted_section);
        const RuntimeRequirements runtime_requirements = casted_section->get_runtime_requirements();
        ASSERT_TRUE(runtime_requirements.get_cre() == expected_cre);
        ASSERT_TRUE(runtime_requirements.get_sections_requirements() == expected_sections_requirements);
        ASSERT_TRUE(runtime_requirements.get_section_id_to_type_mapping() == expected_id_to_type_mapping);
    } else {
        ASSERT_FALSE(read_section);
    }
}

std::vector<ReadParams> test_cases{
    make_test_params(true, ""),
    make_test_params(true, "version=3.1"),
    make_test_params(true, "version=2.0"),
    make_test_params(true, "version=4.0"),
    make_test_params(true, "VERSION=3.0"),
    make_test_params(true, "versio=3.0"),
    make_test_params(true, "version=3.0;"),
    make_test_params(true, "version="),
    make_test_params(true, "version=3.0;cre="),
    make_test_params(true, "custom=CONTENT"),
    make_test_params(true, "version=3.0;cre=(.BATCH_SIZE"),
    make_test_params(true, "version=3.0;cre=BATCH_SIZE;batch_size=CUSTOM"),  // Missing section ID
    make_test_params(true, "version=3.0;cre=BATCH_SIZE;batch_size_1=CUSTOM;io_layouts_1=CUSTOM"),
    make_test_params(true, "version=3.0;cre=BATCH_SIZE;batch_size1=CUSTOM"),
    make_test_params(false, "version=3.0;cre=BATCH_SIZE", CRE({BATCH_SIZE_PTR})),
    make_test_params(false,
                     "version=3.0;batch_size_0=CUSTOM",
                     CRE(),
                     {{0, "CUSTOM"}},
                     {{0, SectionTypeCode::BATCH_SIZE}}),
    make_test_params(false,
                     "version=3.0;cre=BATCH_SIZE_2.AND.COMPILER_VERSION_0;batch_size_1=CUSTOM",
                     CRE({BATCH_SIZE_PTR, ID_2_PTR, CRE::AND_PTR, COMPILER_VERSION_PTR, ID_0_PTR}),
                     {{1, "CUSTOM"}},
                     {{1, SectionTypeCode::BATCH_SIZE}}),
    make_test_params(
        false,
        "version=3.0;cre=BATCH_SIZE_2.AND.COMPILER_VERSION_0;batch_size_2=CUSTOM1;compiler_version_0=CUSTOM2",
        CRE({BATCH_SIZE_PTR, ID_2_PTR, CRE::AND_PTR, COMPILER_VERSION_PTR, ID_0_PTR}),
        {{2, "CUSTOM1"}, {0, "CUSTOM2"}},
        {{2, SectionTypeCode::BATCH_SIZE}, {0, SectionTypeCode::COMPILER_VERSION}}),
    make_test_params(
        false,
        "version=3.0;compiler_version_0=CUSTOM2;cre=BATCH_SIZE_2.AND.COMPILER_VERSION_0;batch_size_2=CUSTOM1",
        CRE({BATCH_SIZE_PTR, ID_2_PTR, CRE::AND_PTR, COMPILER_VERSION_PTR, ID_0_PTR}),
        {{2, "CUSTOM1"}, {0, "CUSTOM2"}},
        {{2, SectionTypeCode::BATCH_SIZE}, {0, SectionTypeCode::COMPILER_VERSION}}),
};

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         RuntimeRequirementsSectionReadTest,
                         testing::ValuesIn(augment_cases_using_stream_type(test_cases)),
                         RuntimeRequirementsSectionReadTest::getTestCaseName);
