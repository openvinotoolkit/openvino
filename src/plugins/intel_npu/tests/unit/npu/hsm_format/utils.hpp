// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection_instance_evaluator.hpp"
#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/common/section_type.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace intel_npu {

constexpr std::string_view STRING_THAT_EVALUATES_TO_SUPPORTED = "1";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNSUPPORTED = "";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNKNOWN = "0";

ov::runtime::hsm::SectionSink create_generic_section_sink(std::ostream& stream);

ov::runtime::hsm::ManifestEntry create_manifest_entry_of_type(const SectionTypeCode type,
                                                              const bool is_inline = false,
                                                              const bool any_device = false);

// TODO could use gtests "MOCK"
class MockTypeEvaluator : public ISectionTypeEvaluator {
public:
    MockTypeEvaluator(const bool result);

private:
    bool evaluate() const override;

    bool m_result;
};

class MockInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override;
};

class MockSection : public ISection {
public:
    MockSection(const std::vector<std::shared_ptr<CREToken>>& cre_tokens,
                const std::optional<std::string>& individual_requirements,
                const ov::runtime::hsm::SectionAlignment& alignment = {});

    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    std::optional<std::string> get_individual_compatibility_requirements() const override;

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

private:
    std::vector<std::shared_ptr<CREToken>> m_cre_tokens;
    std::optional<std::string> m_individual_requirements;
};

}  // namespace intel_npu
