// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

namespace {

constexpr intel_npu::ISection::SectionID CRE_SECTION_ID = 100;

constexpr intel_npu::CREToken AND = 50000;
constexpr intel_npu::CREToken OR = 50001;
constexpr intel_npu::CREToken OPEN = 50002;
constexpr intel_npu::CREToken CLOSE = 50003;

const std::unordered_set<intel_npu::CREToken> RESERVED_TOKENS{AND, OR, OPEN, CLOSE};

}  // namespace

namespace intel_npu {

CRESection::CRESection() : ISection(CRE_SECTION_ID), m_expression({AND}) {}

void CRESection::append_to_expression(const CREToken requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token));
    m_expression.push_back(requirement_token);
}

void CRESection::write(std::ostream& stream, BlobWriter* writer) {
    stream.write(reinterpret_cast<const char*>(m_expression.data()), m_expression.size());
    writer->offset += m_expression.size();
}

}  // namespace intel_npu
