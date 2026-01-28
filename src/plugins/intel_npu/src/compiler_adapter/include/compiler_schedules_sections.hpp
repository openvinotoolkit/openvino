// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"
#include "weightless_graph.hpp"

namespace intel_npu {

class ELFMainScheduleSection final : public ISection {
public:
    ELFMainScheduleSection(const std::shared_ptr<Graph>& graph);

    ELFMainScheduleSection(ov::Tensor main_schedule);

    void write(std::ostream& stream, BlobWriter* writer) override;

    void set_graph(const std::shared_ptr<Graph>& graph);

    ov::Tensor get_schedule() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    std::shared_ptr<Graph> m_graph;
    ov::Tensor m_main_schedule;
};

class ELFInitSchedulesSection final : public ISection {
public:
    ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph);

    ELFInitSchedulesSection(std::vector<ov::Tensor>& init_schedules);

    void write(std::ostream& stream, BlobWriter* writer) override;

    void set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph);

    std::vector<ov::Tensor> get_schedules() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    std::shared_ptr<WeightlessGraph> m_weightless_graph;
    std::vector<ov::Tensor> m_init_schedules;
};

}  // namespace intel_npu
