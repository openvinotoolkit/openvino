// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

namespace {

constexpr intel_npu::ISection::SectionID CRE_SECTION_ID = 100;

}  // namespace

namespace intel_npu {

CRE::CRE() : m_expression({CRE::AND}) {}

size_t CRE::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(m_expression.data()), m_expression.size());
    return m_expression.size();
}

void CRE::append_to_expression(const CRE::Token requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");
    m_expression.push_back(requirement_token);
}

void CRE::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    m_expression.insert(m_expression.end(), requirement_tokens.begin(), requirement_tokens.end());
}

CRESection::CRESection() : ISection(CRE_SECTION_ID) {}

void CRESection::append_to_expression(const CRE::Token requirement_token) {
    m_cre.append_to_expression(requirement_token);
}

void CRESection::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    m_cre.append_to_expression(requirement_tokens);
}

void CRESection::write(std::ostream& stream, BlobWriter* writer) {
    writer->offset += m_cre.write(stream);
}

}  // namespace intel_npu
