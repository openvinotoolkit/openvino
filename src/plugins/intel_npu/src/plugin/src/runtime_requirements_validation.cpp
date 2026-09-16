// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime_requirements_validation.hpp"

#include "compiler_schedule_instance_evaluator.hpp"
#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_source.hpp"
#include "intel_npu/common/runtime_requirements.hpp"
#include "intel_npu/common/supported_section_type_evaluator.hpp"
#include "metadata.hpp"

namespace {

using namespace intel_npu;

/**
 * @brief Evaluates the second version of the runtime requirements format, corresponding to the blob format v2
 * (header-sections-manifest)
 */
ov::CompatibilityCheck validateCompatibilityDescriptorFormatV2(
    std::string_view runtimeRequirements,
    const ov::SoPtr<intel_npu::IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper) {
    // Need to create a few object to connect to the API used within the import path
    BlobSource source(
        ov::Tensor(ov::element::Type_t::u8, ov::Shape({runtimeRequirements.size()}), runtimeRequirements.data()));
    BlobReaderInterface readerInterface(source, 0, runtimeRequirements.size(), 0, runtimeRequirements.size());

    std::shared_ptr<RuntimeRequirementsSection> runtimeRequirementsSection;
    try {
        runtimeRequirementsSection =
            std::dynamic_pointer_cast<RuntimeRequirementsSection>(RuntimeRequirementsSection::read(readerInterface));
    } catch (...) {
        // E.g. unsupported version
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    // Build the section type & instance evaluators
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators;
    std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators;

    // This evaluator can be shared, since all it does is to return "true"
    const auto supported_section_type_evaluator = std::make_shared<SupportedSectionTypeEvaluator>();
    for (const SectionType& type : ALREADY_SUPPORTED_SECTION_TYPES) {
        type_evaluators[type] = supported_section_type_evaluator;
    }

    const auto compiler_schedules_instance_evaluator =
        std::make_shared<CompilerScheduleInstanceEvaluator>(backend, optionSupportHelper);
    instance_evaluators[SectionTypeCode::ELF_MAIN_SCHEDULE] = compiler_schedules_instance_evaluator;
    instance_evaluators[SectionTypeCode::DYNAMIC_SCHEDULE] = compiler_schedules_instance_evaluator;

    try {
        return runtimeRequirementsSection->get_runtime_requirements().get_compatibility_check_result(
            type_evaluators,
            instance_evaluators);
    } catch (...) {
        // Since the same checks would be performed at import time, a throw now means a throw while importing
        return ov::CompatibilityCheck::UNSUPPORTED;
    }
}

/**
 * @brief Evaluates the first version of the runtime requirements format, corresponding to the blob format v1 (compiler
 * schedules followed by metadata fields)
 */
ov::CompatibilityCheck validateCompatibilityDescriptorFormatV1(
    std::string_view runtimeRequirements,
    const ov::SoPtr<intel_npu::IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper) {
    std::unique_ptr<MetadataBase> metadata = nullptr;
    try {
        metadata = read_as_text(runtimeRequirements);
    } catch (...) {
        // The same failure would be expected at import time as well
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    const auto compilerRuntimeRequirements = metadata->get_compatibility_descriptor();

    if (!compilerRuntimeRequirements.has_value() || compilerRuntimeRequirements->empty()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
    return CompilerScheduleInstanceEvaluator(backend, optionSupportHelper)
        .evaluate(compilerRuntimeRequirements.value());
}

}  // namespace

namespace intel_npu {

ov::CompatibilityCheck validateCompatibilityDescriptor(
    const ov::AnyMap& arguments,
    const ov::SoPtr<IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper) {
    if (arguments.empty() || arguments.find(ov::runtime_requirements.name()) == arguments.end()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto& runtimeRequirements = arguments.at(ov::runtime_requirements.name()).as<const std::string&>();
    if (runtimeRequirements.empty()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    bool is_v2 = false;
    try {
        is_v2 = is_runtime_requirements_format_v2(runtimeRequirements);
    } catch (...) {
        // Failed to parse the string
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    return is_v2 ? validateCompatibilityDescriptorFormatV2(runtimeRequirements, backend, optionSupportHelper)
                 : validateCompatibilityDescriptorFormatV1(runtimeRequirements, backend, optionSupportHelper);
}

}  // namespace intel_npu
