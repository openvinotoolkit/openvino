// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/weights_prefetch_plan.hpp"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __linux__
#    include <sys/mman.h>
#    include <unistd.h>
#endif

#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu {

namespace {

bool trace_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("OV_GPU_MMAP_WEIGHTS_PREFETCH_TRACE");
        return value && *value != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}

uint64_t process_read_bytes() noexcept {
#ifdef __linux__
    auto* io = std::fopen("/proc/self/io", "r");
    if (!io) {
        return 0;
    }

    char line[128] = {};
    unsigned long long value = 0;
    while (std::fgets(line, sizeof(line), io)) {
        if (std::sscanf(line, "read_bytes: %llu", &value) == 1) {
            break;
        }
    }
    std::fclose(io);
    return static_cast<uint64_t>(value);
#else
    return 0;
#endif
}

std::pair<size_t, size_t> resident_pages(const ov::op::v0::Constant& constant) noexcept {
#ifdef __linux__
    try {
        const auto size = constant.get_byte_size();
        const auto* data = constant.get_data_ptr<char>();
        const auto page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        if (!data || size == 0 || page_size == 0) {
            return {};
        }

        const auto raw_begin = reinterpret_cast<uintptr_t>(data);
        const auto begin = raw_begin - raw_begin % page_size;
        if (size > std::numeric_limits<uintptr_t>::max() - raw_begin) {
            return {};
        }
        const auto raw_end = raw_begin + size;
        const auto end = raw_end + (page_size - raw_end % page_size) % page_size;
        const auto pages = (end - begin) / page_size;
        std::vector<unsigned char> residency(pages);
        if (::mincore(reinterpret_cast<void*>(begin), end - begin, residency.data()) != 0) {
            return {0, pages};
        }

        size_t resident = 0;
        for (const auto state : residency) {
            resident += (state & 1U) != 0;
        }
        return {resident, pages};
    } catch (...) {
        return {};
    }
#else
    return {};
#endif
}

double elapsed_ms() noexcept {
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

size_t get_mmap_weights_prefetch_budget(bool allow_prefetch) noexcept {
    if (!allow_prefetch) {
        return 0;
    }
    const char* value = std::getenv("OV_GPU_MMAP_WEIGHTS_PREFETCH_BUDGET");
    if (!value || *value == '\0') {
        return 0;
    }

    size_t budget = 0;
    const char* end = value + std::char_traits<char>::length(value);
    const auto parsed = std::from_chars(value, end, budget);
    return parsed.ec == std::errc{} && parsed.ptr == end ? budget : 0;
}

void trace_mmap_weight_event(const char* event, const ov::op::v0::Constant& constant) noexcept {
    if (!trace_enabled()) {
        return;
    }

    const auto [resident, pages] = resident_pages(constant);
    std::fprintf(stderr,
                 "OV_GPU_MMAP_TRACE t_ms=%.3f event=%s op=%s source=%llu offset=%llu size=%zu "
                 "read_bytes=%llu resident=%zu pages=%zu\n",
                 elapsed_ms(),
                 event,
                 constant.get_friendly_name().c_str(),
                 static_cast<unsigned long long>(ov::wsh::Extension::get_constant_source_id(constant)),
                 static_cast<unsigned long long>(ov::wsh::Extension::get_constant_id(constant)),
                 constant.get_byte_size(),
                 static_cast<unsigned long long>(process_read_bytes()),
                 resident,
                 pages);
    std::fflush(stderr);
}

WeightsPrefetchPlan::WeightsPrefetchPlan(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops,
                                         size_t byte_budget)
    : m_byte_budget(byte_budget) {
    if (m_byte_budget == 0) {
        return;
    }

    m_regions.reserve(ordered_ops.size());
    for (const auto& op : ordered_ops) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || !ov::wsh::Extension::supports_async_prefetch(*constant)) {
            continue;
        }

        const size_t size = constant->get_byte_size();
        const size_t offset = ov::wsh::Extension::get_constant_id(*constant);
        if (size == 0 || offset > std::numeric_limits<size_t>::max() - size) {
            continue;
        }

        const size_t index = m_regions.size();
        m_region_by_node.emplace(op.get(), index);
        m_regions.push_back(Region{constant,
                                   ov::wsh::Extension::get_constant_source_id(*constant),
                                   offset,
                                   size,
                                   State::unscheduled});
        if (trace_enabled()) {
            bool duplicate = false;
            bool overlap = false;
            for (size_t previous = 0; previous < index; ++previous) {
                const auto& other = m_regions[previous];
                if (other.source_id != m_regions.back().source_id) {
                    continue;
                }
                duplicate |= other.offset == offset && other.size == size;
                overlap |= offset < other.offset + other.size && other.offset < offset + size;
            }
            std::fprintf(stderr,
                         "OV_GPU_MMAP_TRACE t_ms=%.3f event=plan index=%zu op=%s source=%llu offset=%zu "
                         "size=%zu duplicate=%d overlap=%d budget=%zu read_bytes=%llu\n",
                         elapsed_ms(),
                         index,
                         constant->get_friendly_name().c_str(),
                         static_cast<unsigned long long>(m_regions.back().source_id),
                         offset,
                         size,
                         duplicate,
                         overlap,
                         m_byte_budget,
                         static_cast<unsigned long long>(process_read_bytes()));
        }
    }

    if (trace_enabled()) {
        std::fprintf(stderr,
                     "OV_GPU_MMAP_TRACE t_ms=%.3f event=plan_ready regions=%zu budget=%zu read_bytes=%llu\n",
                     elapsed_ms(),
                     m_regions.size(),
                     m_byte_budget,
                     static_cast<unsigned long long>(process_read_bytes()));
        std::fflush(stderr);
    }
}

WeightsPrefetchPlan::~WeightsPrefetchPlan() {
    for (auto& region : m_regions) {
        if (region.state != State::issued && region.state != State::ready) {
            continue;
        }
        if (region.state == State::issued) {
            ov::wsh::Extension::wait_prefetch(*region.constant);
        }
        ov::wsh::Extension::hint_evict(*region.constant);
        region.state = State::consumed;
    }
}

void WeightsPrefetchPlan::prime() noexcept {
    if (trace_enabled()) {
        std::fprintf(stderr,
                     "OV_GPU_MMAP_TRACE t_ms=%.3f event=prime inflight=%zu next=%zu read_bytes=%llu\n",
                     elapsed_ms(),
                     m_in_flight_bytes,
                     m_next_to_issue,
                     static_cast<unsigned long long>(process_read_bytes()));
    }
    fill_window();
}

void WeightsPrefetchPlan::before(const std::shared_ptr<ov::Node>& op) noexcept {
    const auto found = m_region_by_node.find(op.get());
    if (found == m_region_by_node.end()) {
        return;
    }

    auto& region = m_regions[found->second];
    trace_mmap_weight_event("before", *region.constant);
    if (region.state == State::unscheduled) {
        ov::wsh::Extension::hint_prefetch_async(*region.constant);
        region.state = State::issued;
        m_in_flight_bytes += region.size;
        while (m_next_to_issue < m_regions.size() && m_regions[m_next_to_issue].state != State::unscheduled) {
            ++m_next_to_issue;
        }
    }
    if (region.state == State::issued) {
        trace_mmap_weight_event("wait_begin", *region.constant);
        ov::wsh::Extension::wait_prefetch(*region.constant);
        region.state = State::ready;
        trace_mmap_weight_event("wait_end", *region.constant);
    }
}

void WeightsPrefetchPlan::after(const std::shared_ptr<ov::Node>& op) noexcept {
    const auto found = m_region_by_node.find(op.get());
    if (found == m_region_by_node.end()) {
        return;
    }

    auto& region = m_regions[found->second];
    trace_mmap_weight_event("after_primitive", *region.constant);
    if (region.state == State::issued) {
        ov::wsh::Extension::wait_prefetch(*region.constant);
    }
    if (region.state == State::issued || region.state == State::ready) {
        ov::wsh::Extension::hint_evict(*region.constant);
        m_in_flight_bytes -= region.size;
        region.state = State::consumed;
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "OV_GPU_MMAP_TRACE t_ms=%.3f event=consume op=%s index=%zu inflight=%zu next=%zu "
                         "read_bytes=%llu\n",
                         elapsed_ms(),
                         region.constant->get_friendly_name().c_str(),
                         found->second,
                         m_in_flight_bytes,
                         m_next_to_issue,
                         static_cast<unsigned long long>(process_read_bytes()));
        }
        fill_window();
    }
}

void WeightsPrefetchPlan::fill_window() noexcept {
    while (m_next_to_issue < m_regions.size()) {
        auto& region = m_regions[m_next_to_issue];
        const bool fits = region.size <= m_byte_budget - std::min(m_byte_budget, m_in_flight_bytes);
        const bool allow_single_oversized = m_in_flight_bytes == 0;
        const bool overlap = overlaps_issued(region);
        if ((!fits && !allow_single_oversized) || overlap) {
            if (trace_enabled()) {
                std::fprintf(stderr,
                             "OV_GPU_MMAP_TRACE t_ms=%.3f event=window_blocked index=%zu op=%s fits=%d "
                             "oversized=%d overlap=%d inflight=%zu budget=%zu read_bytes=%llu\n",
                             elapsed_ms(),
                             m_next_to_issue,
                             region.constant->get_friendly_name().c_str(),
                             fits,
                             allow_single_oversized,
                             overlap,
                             m_in_flight_bytes,
                             m_byte_budget,
                             static_cast<unsigned long long>(process_read_bytes()));
            }
            break;
        }

        trace_mmap_weight_event("issue_begin", *region.constant);
        ov::wsh::Extension::hint_prefetch_async(*region.constant);
        region.state = State::issued;
        m_in_flight_bytes += region.size;
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "OV_GPU_MMAP_TRACE t_ms=%.3f event=issue_end op=%s index=%zu state=issued inflight=%zu "
                         "budget=%zu read_bytes=%llu\n",
                         elapsed_ms(),
                         region.constant->get_friendly_name().c_str(),
                         m_next_to_issue,
                         m_in_flight_bytes,
                         m_byte_budget,
                         static_cast<unsigned long long>(process_read_bytes()));
        }
        ++m_next_to_issue;
    }
}

bool WeightsPrefetchPlan::overlaps_issued(const Region& candidate) const noexcept {
    const size_t candidate_end = candidate.offset + candidate.size;
    for (const auto& region : m_regions) {
        if ((region.state != State::issued && region.state != State::ready) ||
            region.source_id != candidate.source_id) {
            continue;
        }
        const size_t region_end = region.offset + region.size;
        if (candidate.offset < region_end && region.offset < candidate_end) {
            return true;
        }
    }
    return false;
}

ConstantFoldingPrefetchPlan::ConstantFoldingPrefetchPlan(
    const std::vector<std::shared_ptr<ov::Node>>& ordered_ops,
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
            if (!constant || !ov::wsh::Extension::supports_async_prefetch(*constant)) {
                continue;
            }

            size_t index = 0;
            if (const auto found = region_by_constant.find(constant.get()); found != region_by_constant.end()) {
                index = found->second;
            } else {
                const size_t size = constant->get_byte_size();
                const size_t offset = ov::wsh::Extension::get_constant_id(*constant);
                if (size == 0 || offset > std::numeric_limits<size_t>::max() - size) {
                    continue;
                }

                index = m_regions.size();
                region_by_constant.emplace(constant.get(), index);
                m_regions.push_back(Region{constant,
                                           ov::wsh::Extension::get_constant_source_id(*constant),
                                           offset,
                                           size,
                                           consumer.get(),
                                           State::unscheduled});
                if (trace_enabled()) {
                    bool duplicate = false;
                    bool overlap = false;
                    for (size_t previous = 0; previous < index; ++previous) {
                        const auto& other = m_regions[previous];
                        if (other.source_id != m_regions.back().source_id) {
                            continue;
                        }
                        duplicate |= other.offset == offset && other.size == size;
                        overlap |= offset < other.offset + other.size && other.offset < offset + size;
                    }
                    std::fprintf(stderr,
                                 "OV_GPU_MMAP_TRACE t_ms=%.3f event=fold_plan index=%zu consumer=%s op=%s "
                                 "source=%llu offset=%zu size=%zu duplicate=%d overlap=%d budget=%zu "
                                 "read_bytes=%llu\n",
                                 elapsed_ms(),
                                 index,
                                 consumer->get_friendly_name().c_str(),
                                 constant->get_friendly_name().c_str(),
                                 static_cast<unsigned long long>(m_regions.back().source_id),
                                 offset,
                                 size,
                                 duplicate,
                                 overlap,
                                 m_byte_budget,
                                 static_cast<unsigned long long>(process_read_bytes()));
                }
            }

            auto& consumer_regions = m_regions_by_consumer[consumer.get()];
            if (std::find(consumer_regions.begin(), consumer_regions.end(), index) == consumer_regions.end()) {
                consumer_regions.push_back(index);
            }
            m_regions[index].last_consumer = consumer.get();
        }
    }

    if (trace_enabled()) {
        std::fprintf(stderr,
                     "OV_GPU_MMAP_TRACE t_ms=%.3f event=fold_plan_ready regions=%zu consumers=%zu budget=%zu "
                     "read_bytes=%llu\n",
                     elapsed_ms(),
                     m_regions.size(),
                     m_regions_by_consumer.size(),
                     m_byte_budget,
                     static_cast<unsigned long long>(process_read_bytes()));
        std::fflush(stderr);
    }
}

ConstantFoldingPrefetchPlan::~ConstantFoldingPrefetchPlan() {
    finish();
    for (auto& region : m_regions) {
        if (region.state == State::issued) {
            ov::wsh::Extension::wait_prefetch(*region.constant);
        }
    }
}

void ConstantFoldingPrefetchPlan::prime() noexcept {
    fill_window();
}

void ConstantFoldingPrefetchPlan::advance(const std::shared_ptr<const ov::Node>& consumer) noexcept {
    if (consumer.get() == m_active_consumer) {
        return;
    }
    if (m_active_consumer) {
        after(m_active_consumer);
    }
    m_active_consumer = consumer.get();
    before(m_active_consumer);
}

void ConstantFoldingPrefetchPlan::finish() noexcept {
    if (m_active_consumer) {
        after(m_active_consumer);
        m_active_consumer = nullptr;
    }
    for (auto& region : m_regions) {
        if (region.state != State::issued && region.state != State::ready) {
            continue;
        }
        if (region.state == State::issued) {
            ov::wsh::Extension::wait_prefetch(*region.constant);
        }
        ov::wsh::Extension::hint_evict(*region.constant);
        m_in_flight_bytes -= region.size;
        region.state = State::consumed;
    }
}

bool ConstantFoldingPrefetchPlan::empty() const noexcept {
    return m_regions.empty();
}

bool ConstantFoldingPrefetchPlan::contains(const ov::Node* consumer) const noexcept {
    return m_regions_by_consumer.count(consumer) != 0;
}

void ConstantFoldingPrefetchPlan::before(const ov::Node* consumer) noexcept {
    const auto found = m_regions_by_consumer.find(consumer);
    if (found == m_regions_by_consumer.end()) {
        return;
    }

    for (const auto index : found->second) {
        auto& region = m_regions[index];
        if (region.state == State::unscheduled) {
            if (overlaps_issued(region)) {
                // The same backing pages are already being populated for another logical input.
                // Waiting through this constant covers all overlapping tasks owned by the mapping.
                ov::wsh::Extension::wait_prefetch(*region.constant);
                region.state = State::ready;
            } else {
                ov::wsh::Extension::hint_prefetch_async(*region.constant);
                region.state = State::issued;
            }
            m_in_flight_bytes += region.size;
            while (m_next_to_issue < m_regions.size() && m_regions[m_next_to_issue].state != State::unscheduled) {
                ++m_next_to_issue;
            }
        }
        if (region.state == State::issued) {
            trace_mmap_weight_event("fold_wait_begin", *region.constant);
            ov::wsh::Extension::wait_prefetch(*region.constant);
            region.state = State::ready;
            trace_mmap_weight_event("fold_wait_end", *region.constant);
        }
    }
}

void ConstantFoldingPrefetchPlan::after(const ov::Node* consumer) noexcept {
    const auto found = m_regions_by_consumer.find(consumer);
    if (found == m_regions_by_consumer.end()) {
        return;
    }

    for (const auto index : found->second) {
        auto& region = m_regions[index];
        if (region.last_consumer != consumer || region.state == State::consumed || region.state == State::unscheduled) {
            continue;
        }
        if (region.state == State::issued) {
            ov::wsh::Extension::wait_prefetch(*region.constant);
        }
        trace_mmap_weight_event("fold_consume", *region.constant);
        ov::wsh::Extension::hint_evict(*region.constant);
        trace_mmap_weight_event("fold_evict_end", *region.constant);
        m_in_flight_bytes -= region.size;
        region.state = State::consumed;
        fill_window();
    }
}

void ConstantFoldingPrefetchPlan::fill_window() noexcept {
    while (m_next_to_issue < m_regions.size()) {
        auto& region = m_regions[m_next_to_issue];
        const bool fits = region.size <= m_byte_budget - std::min(m_byte_budget, m_in_flight_bytes);
        const bool allow_single_oversized = m_in_flight_bytes == 0;
        if ((!fits && !allow_single_oversized) || overlaps_issued(region)) {
            break;
        }

        trace_mmap_weight_event("fold_issue_begin", *region.constant);
        ov::wsh::Extension::hint_prefetch_async(*region.constant);
        region.state = State::issued;
        m_in_flight_bytes += region.size;
        ++m_next_to_issue;
    }
}

bool ConstantFoldingPrefetchPlan::overlaps_issued(const Region& candidate) const noexcept {
    const size_t candidate_end = candidate.offset + candidate.size;
    for (const auto& region : m_regions) {
        if ((region.state != State::issued && region.state != State::ready) ||
            region.source_id != candidate.source_id) {
            continue;
        }
        const size_t region_end = region.offset + region.size;
        if (candidate.offset < region_end && region.offset < candidate_end) {
            return true;
        }
    }
    return false;
}

ConstantFoldingPrefetchState::ConstantFoldingPrefetchState(size_t byte_budget) : m_byte_budget(byte_budget) {}

void ConstantFoldingPrefetchState::on_pass_begin(const std::shared_ptr<ov::Model>& model) noexcept {
    try {
        auto plan = std::make_unique<ConstantFoldingPrefetchPlan>(model->get_ordered_ops(), m_byte_budget);
        plan->prime();
        m_plans.push_back(std::move(plan));
    } catch (...) {
        m_plans.push_back(nullptr);
    }
}

bool ConstantFoldingPrefetchState::defer_pre_calculated_values(const std::shared_ptr<ov::Model>&,
                                                               const std::shared_ptr<const ov::Node>& node) noexcept {
    if (m_plans.empty() || !m_plans.back()) {
        return false;
    }
    return m_plans.back()->contains(node.get());
}

void ConstantFoldingPrefetchState::before_ordered_node(const std::shared_ptr<ov::Model>&,
                                                       const std::shared_ptr<const ov::Node>& node) noexcept {
    if (!m_plans.empty() && m_plans.back()) {
        m_plans.back()->advance(node);
    }
}

void ConstantFoldingPrefetchState::on_pass_end(const std::shared_ptr<ov::Model>&) noexcept {
    if (m_plans.empty()) {
        return;
    }
    if (m_plans.back()) {
        m_plans.back()->finish();
    }
    m_plans.pop_back();
}

void ConstantFoldingPrefetchState::finish() noexcept {
    while (!m_plans.empty()) {
        if (m_plans.back()) {
            m_plans.back()->finish();
        }
        m_plans.pop_back();
    }
}

}  // namespace ov::intel_gpu
