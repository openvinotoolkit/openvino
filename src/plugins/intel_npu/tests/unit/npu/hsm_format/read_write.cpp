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

ov::runtime::hsm::Reader create_base_reader(std::stringstream& stream, const bool use_tensor_source) {
    std::optional<ov::runtime::hsm::Reader> reader;
    if (!use_tensor_source) {
        reader = ov::runtime::hsm::Reader::open(stream);
        OPENVINO_ASSERT(reader);
        return std::move(reader).value();
    }

    const std::string content = stream.str();
    reader = ov::runtime::hsm::Reader::open(reinterpret_cast<const std::byte*>(content.data()), content.size());
    OPENVINO_ASSERT(reader);
    return std::move(reader).value();
}

}  // namespace

// write_to(stream, const BlobSectionsRegistry& blob_sections_registry);
// read_from(
//     ov::runtime::hsm::Reader& base_reader,
//     const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>&
//     section_readers, const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
//     const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators)

using HSMWriteFailure = ::testing::Test;

TEST_F(HSMWriteFailure, WriteInvalidStream) {
    BlobSectionsRegistry sections_registry;
    std::stringstream stream;
    stream.seekg(1);
    OV_EXPECT_THROW(hsm::write_to(stream, sections_registry), ov::Exception, _);
}

class HSMReadWriteTest : public testing::TestWithParam<bool> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        bool tensor_source = obj.param;

        return std::string(tensor_source ? "tensor" : "stream");
    }

protected:
    void SetUp() override {
        use_tensor_source = GetParam();

        batch_size_section_1 = std::make_shared<BatchSizeSection>(0);
        batch_size_section_2 = std::make_shared<BatchSizeSection>(1);
        compiler_version_section = std::make_shared<CompilerVersionSection>(2);
        sections_registry.register_section(batch_size_section_1);
        sections_registry.register_section(compiler_version_section);
        sections_registry.register_section(batch_size_section_2);

        section_readers[SectionTypeCode::BATCH_SIZE] = BatchSizeSection::decode;
        section_readers[SectionTypeCode::COMPILER_VERSION] = CompilerVersionSection::decode;

        type_evaluators[SectionTypeCode::BATCH_SIZE] = MOCK_TYPE_EVALUATOR_TRUE;
        type_evaluators[SectionTypeCode::COMPILER_VERSION] = MOCK_TYPE_EVALUATOR_TRUE;
        // The mock section uses the IO layouts code since it cannot insert a new one by design
        type_evaluators[SectionTypeCode::IO_LAYOUTS] = MOCK_TYPE_EVALUATOR_TRUE;

        instance_evaluators[SectionTypeCode::BATCH_SIZE] = MOCK_INSTANCE_EVALUATOR;
        instance_evaluators[SectionTypeCode::COMPILER_VERSION] = MOCK_INSTANCE_EVALUATOR;
        instance_evaluators[SectionTypeCode::IO_LAYOUTS] = MOCK_INSTANCE_EVALUATOR;
    }

    bool use_tensor_source;
    std::stringstream stream;
    BlobSectionsRegistry sections_registry;
    std::shared_ptr<BatchSizeSection> batch_size_section_1;
    std::shared_ptr<BatchSizeSection> batch_size_section_2;
    std::shared_ptr<CompilerVersionSection> compiler_version_section;
    std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>> section_readers;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators;
    std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators;
};

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         HSMReadWriteTest,
                         testing::ValuesIn(std::vector<bool>({true, false})),
                         HSMReadWriteTest::getTestCaseName);
