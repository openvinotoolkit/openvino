// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_version_section.hpp"
#include "hsm_read_write.hpp"
#include "intel_npu/common/runtime_requirements.hpp"
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
const auto IO_LAYOUTS_PTR = std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS);
const auto ID_0_PTR = std::make_shared<SectionID>(0);
const auto ID_2_PTR = std::make_shared<SectionID>(2);
const auto ID_3_PTR = std::make_shared<SectionID>(3);
const auto ID_4_PTR = std::make_shared<SectionID>(4);

const SectionType IO_LAYOUTS_TYPE(SectionTypeCode::IO_LAYOUTS);

const auto NPU_DEVICE_ID = static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu);

class IncompatibleRuntimeRequirementsWriter : public ov::runtime::hsm::ISectionEncoder {
public:
    void encode(const ov::runtime::hsm::SectionSink& sink) const override {
        const std::string content("version=1.0");
        sink({reinterpret_cast<const std::byte*>(content.data()), content.size()});
    }
};

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

ov::runtime::hsm::SectionTagReserved build_section_tag(const SectionType& section_type,
                                                       const SectionID& section_id,
                                                       const bool is_inline) {
    const ov::runtime::hsm::SectionTag tag =
        ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(section_type.get_code()), is_inline);
    std::array<uint8_t, 4> reserved_bytes{static_cast<uint8_t>(static_cast<uint8_t>(section_id.get_id()) & 0xFF),
                                          static_cast<uint8_t>(static_cast<uint8_t>(section_id.get_id() >> 8) & 0xFF),
                                          0,
                                          0};
    return ov::runtime::hsm::SectionTagReserved(tag, reserved_bytes);
}

ov::runtime::hsm::SectionTagReserved build_section_tag(const std::shared_ptr<ISection>& section) {
    return build_section_tag(section->get_type(), section->get_id().value(), section->is_inline());
}

}  // namespace

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

        section_readers[SectionTypeCode::RUNTIME_REQUIREMENTS] = RuntimeRequirementsSection::decode;
        section_readers[SectionTypeCode::BATCH_SIZE] = BatchSizeSection::decode;
        section_readers[SectionTypeCode::COMPILER_VERSION] = CompilerVersionSection::decode;
        section_readers[SectionTypeCode::IO_LAYOUTS] =
            [](const ov::runtime::hsm::Section&) -> std::optional<std::shared_ptr<ISection>> {
            return std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>({}), std::nullopt);
        };

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

TEST_P(HSMReadWriteTest, MultipleRuntimeRequirements) {
    auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);
    const auto runtime_requirements_section = sections_registry.build_runtime_requirements_section();
    const auto runtime_requirements_tag = build_section_tag(runtime_requirements_section);
    deferred_writer->add_section(NPU_DEVICE_ID,
                                 runtime_requirements_tag,
                                 runtime_requirements_section,
                                 runtime_requirements_section->get_alignment());
    deferred_writer->add_section(NPU_DEVICE_ID,
                                 runtime_requirements_tag,
                                 runtime_requirements_section,
                                 runtime_requirements_section->get_alignment());

    deferred_writer->finalize();

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                    ov::Exception,
                    _);
}

TEST_P(HSMReadWriteTest, SameIdTwice) {
    auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);

    std::queue<std::shared_ptr<ISection>> write_queue = sections_registry.get_write_queue();
    ASSERT_FALSE(write_queue.empty());
    const std::shared_ptr<ISection>& section = write_queue.front();
    write_queue.pop();
    auto section_tag = build_section_tag(section);
    deferred_writer->add_section(NPU_DEVICE_ID, section_tag, section->encode(), section->get_alignment());
    deferred_writer->add_section(NPU_DEVICE_ID, section_tag, section->encode(), section->get_alignment());
    deferred_writer->finalize();

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                    ov::Exception,
                    _);
}

TEST_P(HSMReadWriteTest, MissingRuntimeRequirementsReader) {
    section_readers.erase(SectionTypeCode::RUNTIME_REQUIREMENTS);
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                    ov::Exception,
                    _);
}

TEST_P(HSMReadWriteTest, OptionalSectionsIfMissingRuntimeRequirements) {
    section_readers.erase(SectionTypeCode::BATCH_SIZE);
    auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);

    deferred_writer->add_section(NPU_DEVICE_ID,
                                 build_section_tag(batch_size_section_1),
                                 batch_size_section_1->encode(),
                                 batch_size_section_1->get_alignment());
    deferred_writer->finalize();

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    BlobSectionsRegistry resulted_registry;
    std::optional<RuntimeRequirements> runtime_requirements;
    OV_ASSERT_NO_THROW(std::tie(resulted_registry, runtime_requirements) =
                           hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators));
    ASSERT_FALSE(resulted_registry.has_section_of_type(SectionTypeCode::BATCH_SIZE));
    ASSERT_FALSE(runtime_requirements.has_value());
}

TEST_P(HSMReadWriteTest, OptionalSectionsIfIncompatibleRuntimeRequirements) {
    section_readers.erase(SectionTypeCode::BATCH_SIZE);
    auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);

    deferred_writer->add_section(NPU_DEVICE_ID,
                                 build_section_tag(SectionTypeCode::RUNTIME_REQUIREMENTS, 5, false),
                                 std::make_shared<IncompatibleRuntimeRequirementsWriter>());
    deferred_writer->add_section(NPU_DEVICE_ID,
                                 build_section_tag(batch_size_section_1),
                                 batch_size_section_1->encode(),
                                 batch_size_section_1->get_alignment());
    deferred_writer->finalize();

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    BlobSectionsRegistry resulted_registry;
    std::optional<RuntimeRequirements> runtime_requirements;
    OV_ASSERT_NO_THROW(std::tie(resulted_registry, runtime_requirements) =
                           hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators));
    ASSERT_FALSE(resulted_registry.has_section_of_type(SectionTypeCode::BATCH_SIZE));
    ASSERT_FALSE(runtime_requirements.has_value());
}

TEST_P(HSMReadWriteTest, SupportedRuntimeRequirementsButMissingReader) {
    section_readers.erase(SectionTypeCode::IO_LAYOUTS);
    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{IO_LAYOUTS_PTR, ID_3_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_SUPPORTED)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW_HAS_SUBSTRING(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                                  ov::Exception,
                                  "No section reader found for mandatory section");
}

TEST_P(HSMReadWriteTest, UnknownRuntimeRequirementsButMissingReader) {
    section_readers.erase(SectionTypeCode::IO_LAYOUTS);
    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{IO_LAYOUTS_PTR, ID_3_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_UNKNOWN)));
    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{IO_LAYOUTS_PTR, ID_4_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_SUPPORTED)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW_HAS_SUBSTRING(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                                  ov::Exception,
                                  "No section reader found for mandatory section");
}

TEST_P(HSMReadWriteTest, UnsupportedRuntimeRequirements) {
    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{IO_LAYOUTS_PTR, ID_3_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_UNSUPPORTED)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    OV_EXPECT_THROW_HAS_SUBSTRING(hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators),
                                  ov::Exception,
                                  "The imported model is not compatible");
}

TEST_P(HSMReadWriteTest, RuntimeRequirementsNotPartOfRegistry) {
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    const auto& [registry, runtime_requirements] =
        hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators);
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::RUNTIME_REQUIREMENTS));

    ASSERT_TRUE(runtime_requirements.has_value());
    ASSERT_TRUE(runtime_requirements->get_cre() == CRE({BATCH_SIZE_PTR}));
    ASSERT_TRUE(runtime_requirements->get_sections_requirements().empty());
    ASSERT_TRUE(runtime_requirements->get_section_id_to_type_mapping().empty());
}

TEST_P(HSMReadWriteTest, UnsupportedSectionInstanceSkipped) {
    section_readers[SectionTypeCode::IO_LAYOUTS] =
        [](const ov::runtime::hsm::Section&) -> std::optional<std::shared_ptr<ISection>> {
        OPENVINO_NOT_IMPLEMENTED;
    };

    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{CRE::NOT_PTR, IO_LAYOUTS_PTR, ID_3_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_UNSUPPORTED)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    BlobSectionsRegistry registry;
    std::optional<RuntimeRequirements> runtime_requirements;
    OV_ASSERT_NO_THROW(std::tie(registry, runtime_requirements) =
                           hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators));
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::IO_LAYOUTS));
    ASSERT_TRUE(runtime_requirements.has_value());
}

TEST_P(HSMReadWriteTest, UnsupportedSectionTypeSkipped) {
    type_evaluators[SectionTypeCode::IO_LAYOUTS] = MOCK_TYPE_EVALUATOR_FALSE;
    section_readers[SectionTypeCode::IO_LAYOUTS] =
        [](const ov::runtime::hsm::Section&) -> std::optional<std::shared_ptr<ISection>> {
        OPENVINO_NOT_IMPLEMENTED;
    };

    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{CRE::NOT_PTR, IO_LAYOUTS_PTR, ID_3_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_SUPPORTED)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    BlobSectionsRegistry registry;
    std::optional<RuntimeRequirements> runtime_requirements;
    OV_ASSERT_NO_THROW(std::tie(registry, runtime_requirements) =
                           hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators));
    ASSERT_FALSE(registry.has_section_of_type(SectionTypeCode::IO_LAYOUTS));
    ASSERT_TRUE(runtime_requirements.has_value());
}

TEST_P(HSMReadWriteTest, MixedCases) {
    sections_registry.register_section(
        std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{IO_LAYOUTS_PTR,
                                                                             ID_3_PTR,
                                                                             CRE::AND_PTR,
                                                                             IO_LAYOUTS_PTR,
                                                                             ID_4_PTR,
                                                                             CRE::OR_PTR,
                                                                             BATCH_SIZE_PTR,
                                                                             CRE::OR_PTR,
                                                                             COMPILER_VERSION_PTR},
                                      std::string(STRING_THAT_EVALUATES_TO_SUPPORTED)));
    sections_registry.register_section(std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>{},
                                                                     std::string(STRING_THAT_EVALUATES_TO_UNKNOWN)));
    hsm::write_to(stream, sections_registry);

    ov::runtime::hsm::Reader base_reader = create_base_reader(stream, use_tensor_source);
    const auto& [registry, runtime_requirements] =
        hsm::read_from(base_reader, section_readers, type_evaluators, instance_evaluators);
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::IO_LAYOUTS), 2);
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::COMPILER_VERSION), 1);
    ASSERT_EQ(registry.count_sections_of_type(SectionTypeCode::BATCH_SIZE), 2);

    // Check the contents
    auto section = registry.retrieve_section(0);
    ASSERT_TRUE(section);
    const auto casted_section_0 = std::dynamic_pointer_cast<BatchSizeSection>(section);
    ASSERT_TRUE(casted_section_0);
    ASSERT_EQ(casted_section_0->get_batch_size(), batch_size_section_1->get_batch_size());

    section = registry.retrieve_section(1);
    ASSERT_TRUE(section);
    const auto casted_section_1 = std::dynamic_pointer_cast<CompilerVersionSection>(section);
    ASSERT_TRUE(casted_section_1);
    ASSERT_EQ(casted_section_1->get_compiler_version(), compiler_version_section->get_compiler_version());

    section = registry.retrieve_section(2);
    ASSERT_TRUE(section);
    const auto casted_section_2 = std::dynamic_pointer_cast<BatchSizeSection>(section);
    ASSERT_TRUE(casted_section_2);
    ASSERT_EQ(casted_section_2->get_batch_size(), batch_size_section_2->get_batch_size());

    section = registry.retrieve_section(3);
    ASSERT_TRUE(section);
    const auto casted_section_3 = std::dynamic_pointer_cast<MockSection>(section);
    ASSERT_TRUE(casted_section_3);

    section = registry.retrieve_section(4);
    ASSERT_TRUE(section);
    const auto casted_section_4 = std::dynamic_pointer_cast<MockSection>(section);
    ASSERT_TRUE(casted_section_4);

    const CRE cre_reference({IO_LAYOUTS_PTR,
                             ID_3_PTR,
                             CRE::AND_PTR,
                             IO_LAYOUTS_PTR,
                             ID_4_PTR,
                             CRE::OR_PTR,
                             BATCH_SIZE_PTR,
                             CRE::OR_PTR,
                             COMPILER_VERSION_PTR,
                             CRE::AND_PTR,
                             CRE::OPEN_PTR,
                             BATCH_SIZE_PTR,
                             CRE::CLOSE_PTR});
    const std::map<SectionID, std::string> sections_requirements_reference(
        {{3, STRING_THAT_EVALUATES_TO_SUPPORTED.data()}, {4, STRING_THAT_EVALUATES_TO_UNKNOWN.data()}});
    const std::unordered_map<SectionID, SectionType> id_to_type_reference{{3, IO_LAYOUTS_TYPE}, {4, IO_LAYOUTS_TYPE}};
    ASSERT_TRUE(runtime_requirements.has_value());
    ASSERT_TRUE(runtime_requirements->get_cre() == cre_reference);
    ASSERT_TRUE(runtime_requirements->get_sections_requirements() == sections_requirements_reference);
    ASSERT_TRUE(runtime_requirements->get_section_id_to_type_mapping() == id_to_type_reference);
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         HSMReadWriteTest,
                         testing::ValuesIn(std::vector<bool>({true, false})),
                         HSMReadWriteTest::getTestCaseName);
