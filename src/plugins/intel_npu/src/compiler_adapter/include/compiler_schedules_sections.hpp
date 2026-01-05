// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "isection.hpp"
#include "weightless_graph.hpp"

namespace intel_npu {

class ELFMainScheduleSection final : public ISection {
public:
    ELFMainScheduleSection(const std::shared_ptr<Graph>& graph);

    void write(std::ostream& stream, BlobWriter* writer) override;

private:
    std::shared_ptr<Graph> m_graph;
};

class ELFInitSchedulesSection final : public ISection {
public:
    ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph);

    void write(std::ostream& stream, BlobWriter* writer) override;

private:
    std::shared_ptr<WeightlessGraph> m_weightless_graph;
};

}  // namespace intel_npu
