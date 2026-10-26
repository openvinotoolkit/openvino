// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/constant_folding_prefetch.hpp"

#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <unordered_map>
#include <utility>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace ov::util {

namespace ws = ov::weight_sharing;

// Tracks the mmap-backed constants consumed by a single ConstantFolding run and keeps a bounded
// window of them prefetched ahead of the ordered folding loop. One plan is created per (possibly
// nested) run_on_model() invocation; nested subgraph runs stack their own plans.
class ConstantFoldingWeightsPrefetch::Plan {
public:
    Plan(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops, size_t byte_budget);
    ~Plan();

    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    void prime() noexcept;
    bool contains(const ov::Node* consumer) const noexcept;
    void advance(const ov::Node* consumer) noexcept;
    void finish() noexcept;

private:
    enum class State { unscheduled, issued, ready, consumed };

    struct Region {
        std::shared_ptr<ov::op::v0::Constant> constant;
        ws::DataID source_id = ws::invalid_source_id;
        size_t offset = 0;
        size_t size = 0;
        const ov::Node* last_consumer = nullptr;
        State state = State::unscheduled;
        std::shared_future<void> future;
    };

    void before(const ov::Node* consumer) noexcept;
    void after(const ov::Node* consumer) noexcept;
    void fill_window() noexcept;
    bool overlaps_issued(const Region& candidate) const noexcept;
    void wait_overlapping(const Region& candidate) noexcept;

    size_t m_byte_budget = 0;
    size_t m_in_flight_bytes = 0;
    size_t m_next_to_issue = 0;
    const ov::Node* m_active_consumer = nullptr;
    std::vector<Region> m_regions;
    std::unordered_map<const ov::Node*, std::vector<size_t>> m_regions_by_consumer;
};

ConstantFoldingWeightsPrefetch::Plan::Plan(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops,
                                           size_t byte_budget)
    : m_byte_budget(byte_budget) {
    if (m_byte_budget == 0) {
        return;
    }

    std::unordered_map<const ov::op::v0::Constant*, size_t> region_by_constant;
    for (const auto& consumer : ordered_ops) {
        if (ov::pass::constant_folding_is_disabled(consumer) ||
            !consumer->can_constant_fold(consumer->input_values())) {
            continue;
        }

        for (const auto& input : consumer->input_values()) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
            if (!constant) {
                continue;
            }
            const auto source_id = ws::Extension::get_constant_source_id(*constant);
            if (source_id == ws::invalid_source_id) {
                continue;
            }

            size_t index = 0;
            if (const auto found = region_by_constant.find(constant.get()); found != region_by_constant.end()) {
                index = found->second;
            } else {
                const size_t size = constant->get_byte_size();
                const size_t offset = ws::Extension::get_constant_id(*constant);
                if (size == 0 || offset > std::numeric_limits<size_t>::max() - size) {
                    continue;
                }

                index = m_regions.size();
                region_by_constant.emplace(constant.get(), index);
                m_regions.push_back(Region{constant, source_id, offset, size, consumer.get(), State::unscheduled, {}});
            }

            auto& consumer_regions = m_regions_by_consumer[consumer.get()];
            if (std::find(consumer_regions.begin(), consumer_regions.end(), index) == consumer_regions.end()) {
                consumer_regions.push_back(index);
            }
            m_regions[index].last_consumer = consumer.get();
        }
    }
}

ConstantFoldingWeightsPrefetch::Plan::~Plan() {
    finish();
}

void ConstantFoldingWeightsPrefetch::Plan::prime() noexcept {
    fill_window();
}

bool ConstantFoldingWeightsPrefetch::Plan::contains(const ov::Node* consumer) const noexcept {
    return m_regions_by_consumer.count(consumer) != 0;
}

void ConstantFoldingWeightsPrefetch::Plan::advance(const ov::Node* consumer) noexcept {
    if (consumer == m_active_consumer) {
        return;
    }
    if (m_active_consumer) {
        after(m_active_consumer);
    }
    m_active_consumer = consumer;
    before(m_active_consumer);
}

void ConstantFoldingWeightsPrefetch::Plan::finish() noexcept {
    if (m_active_consumer) {
        after(m_active_consumer);
        m_active_consumer = nullptr;
    }
    for (auto& region : m_regions) {
        if (region.state != State::issued && region.state != State::ready) {
            continue;
        }
        if (region.state == State::issued && region.future.valid()) {
            region.future.wait();
        }
        ws::Extension::hint_evict(*region.constant);
        region.future = {};
        m_in_flight_bytes -= region.size;
        region.state = State::consumed;
    }
}

void ConstantFoldingWeightsPrefetch::Plan::before(const ov::Node* consumer) noexcept {
    const auto found = m_regions_by_consumer.find(consumer);
    if (found == m_regions_by_consumer.end()) {
        return;
    }

    for (const auto index : found->second) {
        auto& region = m_regions[index];
        if (region.state == State::unscheduled) {
            if (overlaps_issued(region)) {
                // The same backing pages are already being populated for another logical input.
                // Waiting through the overlapping tasks covers this constant as well.
                wait_overlapping(region);
                region.state = State::ready;
            } else {
                region.future = ws::Extension::hint_prefetch(*region.constant);
                region.state = State::issued;
            }
            m_in_flight_bytes += region.size;
            while (m_next_to_issue < m_regions.size() && m_regions[m_next_to_issue].state != State::unscheduled) {
                ++m_next_to_issue;
            }
        }
        if (region.state == State::issued) {
            if (region.future.valid()) {
                region.future.wait();
            }
            region.state = State::ready;
        }
    }
}

void ConstantFoldingWeightsPrefetch::Plan::after(const ov::Node* consumer) noexcept {
    const auto found = m_regions_by_consumer.find(consumer);
    if (found == m_regions_by_consumer.end()) {
        return;
    }

    for (const auto index : found->second) {
        auto& region = m_regions[index];
        if (region.last_consumer != consumer || region.state == State::consumed || region.state == State::unscheduled) {
            continue;
        }
        if (region.state == State::issued && region.future.valid()) {
            region.future.wait();
        }
        ws::Extension::hint_evict(*region.constant);
        region.future = {};
        m_in_flight_bytes -= region.size;
        region.state = State::consumed;
        fill_window();
    }
}

void ConstantFoldingWeightsPrefetch::Plan::fill_window() noexcept {
    while (m_next_to_issue < m_regions.size()) {
        auto& region = m_regions[m_next_to_issue];
        const bool fits = region.size <= m_byte_budget - std::min(m_byte_budget, m_in_flight_bytes);
        const bool allow_single_oversized = m_in_flight_bytes == 0;
        if ((!fits && !allow_single_oversized) || overlaps_issued(region)) {
            break;
        }

        region.future = ws::Extension::hint_prefetch(*region.constant);
        region.state = State::issued;
        m_in_flight_bytes += region.size;
        ++m_next_to_issue;
    }
}

bool ConstantFoldingWeightsPrefetch::Plan::overlaps_issued(const Region& candidate) const noexcept {
    const size_t candidate_end = candidate.offset + candidate.size;
    for (const auto& region : m_regions) {
        if ((region.state != State::issued && region.state != State::ready) ||
            region.source_id != candidate.source_id || &region == &candidate) {
            continue;
        }
        const size_t region_end = region.offset + region.size;
        if (candidate.offset < region_end && region.offset < candidate_end) {
            return true;
        }
    }
    return false;
}

void ConstantFoldingWeightsPrefetch::Plan::wait_overlapping(const Region& candidate) noexcept {
    const size_t candidate_end = candidate.offset + candidate.size;
    for (auto& region : m_regions) {
        if (region.state != State::issued || region.source_id != candidate.source_id || &region == &candidate) {
            continue;
        }
        const size_t region_end = region.offset + region.size;
        if (candidate.offset < region_end && region.offset < candidate_end && region.future.valid()) {
            region.future.wait();
        }
    }
}

ConstantFoldingWeightsPrefetch::ConstantFoldingWeightsPrefetch(size_t byte_budget) : m_byte_budget(byte_budget) {}

ConstantFoldingWeightsPrefetch::~ConstantFoldingWeightsPrefetch() = default;

void ConstantFoldingWeightsPrefetch::on_pass_begin(const std::shared_ptr<ov::Model>& model) noexcept {
    try {
        auto plan = std::make_unique<Plan>(model->get_ordered_ops(), m_byte_budget);
        plan->prime();
        m_plans.push_back(std::move(plan));
    } catch (...) {
        // Prefetching is an optimization hint only; never affect folding correctness.
        m_plans.push_back(nullptr);
    }
}

bool ConstantFoldingWeightsPrefetch::defer_pre_calculated_values(const std::shared_ptr<ov::Model>&,
                                                                 const std::shared_ptr<const ov::Node>& node) noexcept {
    if (m_plans.empty() || !m_plans.back()) {
        return false;
    }
    return m_plans.back()->contains(node.get());
}

void ConstantFoldingWeightsPrefetch::before_ordered_node(const std::shared_ptr<ov::Model>&,
                                                         const std::shared_ptr<const ov::Node>& node) noexcept {
    if (!m_plans.empty() && m_plans.back()) {
        m_plans.back()->advance(node.get());
    }
}

void ConstantFoldingWeightsPrefetch::on_pass_end(const std::shared_ptr<ov::Model>&) noexcept {
    if (m_plans.empty()) {
        return;
    }
    if (m_plans.back()) {
        m_plans.back()->finish();
    }
    m_plans.pop_back();
}

void ConstantFoldingWeightsPrefetch::finish() noexcept {
    while (!m_plans.empty()) {
        if (m_plans.back()) {
            m_plans.back()->finish();
        }
        m_plans.pop_back();
    }
}

}  // namespace ov::util
