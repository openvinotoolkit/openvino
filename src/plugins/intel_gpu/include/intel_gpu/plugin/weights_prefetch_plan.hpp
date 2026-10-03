// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace ov::op::v0 {
class Constant;
}

namespace ov {
class Model;
}

namespace ov::intel_gpu {

size_t get_mmap_weights_prefetch_budget(bool allow_prefetch = true) noexcept;
void trace_mmap_weight_event(const char* event, const ov::op::v0::Constant& constant) noexcept;

class WeightsPrefetchPlan {
public:
    WeightsPrefetchPlan(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops, size_t byte_budget);
    ~WeightsPrefetchPlan();

    WeightsPrefetchPlan(const WeightsPrefetchPlan&) = delete;
    WeightsPrefetchPlan& operator=(const WeightsPrefetchPlan&) = delete;

    void prime() noexcept;
    void before(const std::shared_ptr<ov::Node>& op) noexcept;
    void after(const std::shared_ptr<ov::Node>& op) noexcept;

private:
    enum class State { unscheduled, issued, ready, consumed };

    struct Region {
        std::shared_ptr<ov::op::v0::Constant> constant;
        ov::weight_sharing::DataID source_id = ov::weight_sharing::invalid_source_id;
        size_t offset = 0;
        size_t size = 0;
        State state = State::unscheduled;
    };

    void fill_window() noexcept;
    bool overlaps_issued(const Region& candidate) const noexcept;

    size_t m_byte_budget = 0;
    size_t m_in_flight_bytes = 0;
    size_t m_next_to_issue = 0;
    std::vector<Region> m_regions;
    std::unordered_map<const ov::Node*, size_t> m_region_by_node;
};

class ConstantFoldingPrefetchPlan {
public:
    ConstantFoldingPrefetchPlan(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops, size_t byte_budget);
    ~ConstantFoldingPrefetchPlan();

    ConstantFoldingPrefetchPlan(const ConstantFoldingPrefetchPlan&) = delete;
    ConstantFoldingPrefetchPlan& operator=(const ConstantFoldingPrefetchPlan&) = delete;

    void prime() noexcept;
    void advance(const std::shared_ptr<const ov::Node>& consumer) noexcept;
    void finish() noexcept;
    bool empty() const noexcept;
    bool contains(const ov::Node* consumer) const noexcept;

private:
    enum class State { unscheduled, issued, ready, consumed };

    struct Region {
        std::shared_ptr<ov::op::v0::Constant> constant;
        ov::weight_sharing::DataID source_id = ov::weight_sharing::invalid_source_id;
        size_t offset = 0;
        size_t size = 0;
        const ov::Node* last_consumer = nullptr;
        State state = State::unscheduled;
    };

    void before(const ov::Node* consumer) noexcept;
    void after(const ov::Node* consumer) noexcept;
    void fill_window() noexcept;
    bool overlaps_issued(const Region& candidate) const noexcept;

    size_t m_byte_budget = 0;
    size_t m_in_flight_bytes = 0;
    size_t m_next_to_issue = 0;
    const ov::Node* m_active_consumer = nullptr;
    std::vector<Region> m_regions;
    std::unordered_map<const ov::Node*, std::vector<size_t>> m_regions_by_consumer;
};

class ConstantFoldingPrefetchState : public ov::pass::ConstantFolding::Observer {
public:
    explicit ConstantFoldingPrefetchState(size_t byte_budget);

    void on_pass_begin(const std::shared_ptr<ov::Model>& model) noexcept override;
    bool defer_pre_calculated_values(const std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<const ov::Node>& node) noexcept override;
    void before_ordered_node(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::Node>& node) noexcept override;
    void on_pass_end(const std::shared_ptr<ov::Model>& model) noexcept override;
    void finish() noexcept;

private:
    size_t m_byte_budget = 0;
    std::vector<std::unique_ptr<ConstantFoldingPrefetchPlan>> m_plans;
};

}  // namespace ov::intel_gpu
