// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

namespace {

constexpr intel_npu::ISection::SectionID ELF_MAIN_SCHEDULE_SECTION_ID = 102;
constexpr intel_npu::ISection::SectionID ELF_INIT_SCHEDULES_SECTION_ID = 103;

}  // namespace

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(const std::shared_ptr<IGraph>& graph)
    : ISection(ELF_MAIN_SCHEDULE_SECTION_ID),
      m_graph(graph) {}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph)
    : ISection(ELF_INIT_SCHEDULES_SECTION_ID),
      m_weightless_graph(weightless_graph) {}

}  // namespace intel_npu
