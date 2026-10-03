// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <utility>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/memory_prefetch.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::weight_sharing {

namespace {

std::optional<size_t> getenv_mib(const char* name) {
    const auto value = ov::util::getenv_string(name);
    size_t parsed = 0;
    const auto end = value.data() + value.size();
    if (value.empty() || std::from_chars(value.data(), end, parsed).ptr != end) {
        return std::nullopt;
    }
    return parsed << 20;
}

struct Range {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

std::optional<Range> data_range(const ov::op::v0::Constant& constant) {
    const auto size = constant.get_byte_size();
    if (size < static_cast<size_t>(ov::util::get_system_page_size())) {
        return std::nullopt;
    }
    const auto begin = reinterpret_cast<uintptr_t>(constant.get_data_ptr());
    if (begin == 0) {
        return std::nullopt;
    }
    return Range{begin, begin + size};
}

}  // namespace

std::optional<PrefetchScheduler::Config> PrefetchScheduler::Config::from_env(std::string_view site) {
    return from_env(site, Config{});
}

std::optional<PrefetchScheduler::Config> PrefetchScheduler::Config::from_env(std::string_view site,
                                                                             const Config& defaults) {
    if (!ov::util::getenv_bool("OV_WEIGHTS_PREFETCH")) {
        return std::nullopt;
    }
    if (const auto sites = ov::util::getenv_string("OV_WEIGHTS_PREFETCH_SITES"); !sites.empty()) {
        if (ov::util::split_by_delimiter(sites, ',').count(std::string(site)) == 0) {
            return std::nullopt;
        }
    }
    Config config = defaults;
    config.site = site;
    config.trace = ov::util::getenv_bool("OV_WEIGHTS_PREFETCH_TRACE");
    if (const auto populate = getenv_mib("OV_WEIGHTS_PREFETCH_POPULATE_MB")) {
        config.populate_bytes = *populate;
    }
    if (const auto readahead = getenv_mib("OV_WEIGHTS_PREFETCH_READAHEAD_MB")) {
        config.readahead_bytes = *readahead;
    }
    return config;
}

struct PrefetchScheduler::Impl {
    struct Region {
        std::vector<std::weak_ptr<ov::op::v0::Constant>> constants;
        size_t size = 0;
        size_t first_step = 0;
        size_t last_step = 0;
        bool evict = true;

        bool populate_issued = false;
        bool read_ahead_issued = false;
        bool in_populate_window = false;
        bool in_read_ahead_window = false;
        bool reached = false;
        bool done = false;

        // Strong references are held only while background work may touch the data.
        std::vector<std::shared_ptr<ov::op::v0::Constant>> keepalive;
        std::vector<ov::util::PrefetchToken> tokens;

        bool ready() const noexcept {
            return std::all_of(tokens.begin(), tokens.end(), [](const ov::util::PrefetchToken& token) {
                return token.ready();
            });
        }

        void wait() noexcept {
            for (auto& token : tokens) {
                token.wait();
            }
        }

        void cancel() noexcept {
            for (auto& token : tokens) {
                token.cancel();
            }
        }

        void release() noexcept {
            wait();
            tokens.clear();
            keepalive.clear();
        }

        // Prefetches the data of the constants which are still alive. Ranges are computed again on
        // every issue, so data of the constants which died meanwhile is never touched.
        // Returns false if none of the constants is alive anymore.
        bool issue(ov::util::PrefetchMode mode) {
            std::vector<Range> ranges;
            for (const auto& observer : constants) {
                if (auto constant = observer.lock()) {
                    if (const auto range = data_range(*constant)) {
                        ranges.push_back(*range);
                    }
                    if (std::find(keepalive.begin(), keepalive.end(), constant) == keepalive.end()) {
                        keepalive.push_back(std::move(constant));
                    }
                }
            }
            std::sort(ranges.begin(), ranges.end(), [](const Range& lhs, const Range& rhs) {
                return lhs.begin < rhs.begin;
            });
            for (size_t i = 0; i < ranges.size();) {
                auto merged = ranges[i];
                for (++i; i < ranges.size() && ranges[i].begin < merged.end; ++i) {
                    merged.end = std::max(merged.end, ranges[i].end);
                }
                auto token = ov::util::prefetch_async(reinterpret_cast<const void*>(merged.begin),
                                                      static_cast<size_t>(merged.end - merged.begin),
                                                      mode);
                if (token) {
                    tokens.push_back(std::move(token));
                }
            }
            return !keepalive.empty();
        }
    };

    Config m_config;
    std::vector<Region> m_regions;
    std::vector<size_t> m_order;                      // region indices by first step
    std::vector<std::vector<size_t>> m_step_regions;  // step -> regions read in it
    std::vector<std::vector<size_t>> m_last_regions;  // step -> regions read for the last time in it
    size_t m_next_reach = 0;
    size_t m_next_populate = 0;
    size_t m_next_read_ahead = 0;
    size_t m_populated_ahead = 0;
    size_t m_read_ahead = 0;
    Stats m_stats;

    Impl(const Plan& plan, const Config& config) : m_config(config) {
        struct Entry {
            Range range;
            size_t step;
            std::shared_ptr<ov::op::v0::Constant> constant;
            bool evict;
        };
        std::vector<Entry> entries;
        for (size_t step = 0; step < plan.size(); ++step) {
            for (const auto& use : plan[step]) {
                if (auto constant = use.constant.lock()) {
                    if (const auto range = data_range(*constant)) {
                        entries.push_back({*range, step, std::move(constant), use.evict_after_last_use});
                    }
                }
            }
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
            return lhs.range.begin < rhs.range.begin;
        });

        // Constants sharing data (e.g. tied weights) form a single region.
        m_step_regions.resize(plan.size());
        m_last_regions.resize(plan.size());
        for (size_t i = 0; i < entries.size();) {
            Region region;
            auto end = entries[i].range.end;
            const auto begin = entries[i].range.begin;
            region.first_step = entries[i].step;
            region.last_step = entries[i].step;
            std::unordered_set<const ov::op::v0::Constant*> seen;
            const auto index = m_regions.size();
            for (; i < entries.size() && entries[i].range.begin < end; ++i) {
                const auto& entry = entries[i];
                end = std::max(end, entry.range.end);
                region.first_step = std::min(region.first_step, entry.step);
                region.last_step = std::max(region.last_step, entry.step);
                region.evict &= entry.evict;
                if (seen.insert(entry.constant.get()).second) {
                    region.constants.emplace_back(entry.constant);
                }
                auto& step_regions = m_step_regions[entry.step];
                if (step_regions.empty() || step_regions.back() != index) {
                    step_regions.push_back(index);
                }
            }
            region.size = static_cast<size_t>(end - begin);
            m_last_regions[region.last_step].push_back(index);
            m_regions.push_back(std::move(region));
        }

        m_order.resize(m_regions.size());
        for (size_t i = 0; i < m_order.size(); ++i) {
            m_order[i] = i;
        }
        std::stable_sort(m_order.begin(), m_order.end(), [this](size_t lhs, size_t rhs) {
            return m_regions[lhs].first_step < m_regions[rhs].first_step;
        });
        m_stats.regions = m_regions.size();
        fill();
    }

    ~Impl() {
        for (auto& region : m_regions) {
            region.cancel();
        }
        for (auto& region : m_regions) {
            region.wait();
            // Populated for a step which never came: nothing will read it, give the memory back.
            if (region.populate_issued && !region.reached && region.evict) {
                evict(region);
            }
        }
        if (m_config.trace && !m_regions.empty()) {
            std::fprintf(stderr,
                         "[weights_prefetch] site=%s steps=%zu regions=%zu populated=%zu read_ahead=%zu late=%zu "
                         "waited=%zu evicted=%zu peak_populated_ahead_mb=%zu peak_read_ahead_mb=%zu\n",
                         m_config.site.c_str(),
                         m_step_regions.size(),
                         m_stats.regions,
                         m_stats.populated,
                         m_stats.read_ahead,
                         m_stats.late,
                         m_stats.waited,
                         m_stats.evicted,
                         m_stats.peak_populated_ahead >> 20,
                         m_stats.peak_read_ahead >> 20);
        }
    }

    void leave_windows(Region& region) noexcept {
        if (region.in_populate_window) {
            m_populated_ahead -= region.size;
            region.in_populate_window = false;
        }
        if (region.in_read_ahead_window) {
            m_read_ahead -= region.size;
            region.in_read_ahead_window = false;
        }
    }

    void evict(Region& region) noexcept {
        for (const auto& observer : region.constants) {
            if (const auto constant = observer.lock()) {
                Extension::hint_evict(*constant);
            }
        }
        ++m_stats.evicted;
    }

    // Returns false if the region is gone: all its constants died.
    bool populate(Region& region) {
        region.populate_issued = true;
        if (!region.issue(ov::util::PrefetchMode::populate)) {
            region.done = true;
        }
        return !region.done;
    }

    void fill() {
        for (; m_next_populate < m_order.size(); ++m_next_populate) {
            auto& region = m_regions[m_order[m_next_populate]];
            if (region.populate_issued || region.reached || region.done) {
                continue;
            }
            if (m_config.populate_bytes == 0 ||
                (m_populated_ahead > 0 && m_populated_ahead + region.size > m_config.populate_bytes)) {
                break;
            }
            leave_windows(region);
            if (!populate(region)) {
                continue;
            }
            region.in_populate_window = true;
            m_populated_ahead += region.size;
            ++m_stats.populated;
            m_stats.peak_populated_ahead = std::max(m_stats.peak_populated_ahead, m_populated_ahead);
        }

        for (m_next_read_ahead = std::max(m_next_read_ahead, m_next_populate); m_next_read_ahead < m_order.size();
             ++m_next_read_ahead) {
            auto& region = m_regions[m_order[m_next_read_ahead]];
            if (region.populate_issued || region.read_ahead_issued || region.reached || region.done) {
                continue;
            }
            if (m_config.readahead_bytes == 0 ||
                (m_read_ahead > 0 && m_read_ahead + region.size > m_config.readahead_bytes)) {
                break;
            }
            region.read_ahead_issued = true;
            if (!region.issue(ov::util::PrefetchMode::readahead)) {
                region.done = true;
                continue;
            }
            region.in_read_ahead_window = true;
            m_read_ahead += region.size;
            ++m_stats.read_ahead;
            m_stats.peak_read_ahead = std::max(m_stats.peak_read_ahead, m_read_ahead);
        }
    }

    void begin(size_t step) {
        // Regions of the steps skipped by the consumer are not ahead anymore.
        for (; m_next_reach < m_order.size() && m_regions[m_order[m_next_reach]].first_step <= step; ++m_next_reach) {
            auto& region = m_regions[m_order[m_next_reach]];
            region.reached = true;
            leave_windows(region);
        }
        if (step >= m_step_regions.size()) {
            return;
        }
        for (const auto index : m_step_regions[step]) {
            auto& region = m_regions[index];
            if (region.done) {
                continue;
            }
            if (!region.populate_issued) {
                ++m_stats.late;
                populate(region);
            } else if (!region.ready()) {
                ++m_stats.waited;
            }
            region.wait();
        }
        fill();
    }

    void end(size_t step) {
        if (step >= m_last_regions.size()) {
            return;
        }
        for (const auto index : m_last_regions[step]) {
            auto& region = m_regions[index];
            if (region.done) {
                continue;
            }
            region.cancel();
            region.wait();
            leave_windows(region);
            if (region.evict) {
                evict(region);
            }
            region.release();
            region.done = true;
        }
        fill();
    }
};

PrefetchScheduler::PrefetchScheduler(const Plan& plan, const Config& config)
    : m_impl(std::make_unique<Impl>(plan, config)) {}

PrefetchScheduler::~PrefetchScheduler() = default;

void PrefetchScheduler::begin(size_t step) noexcept {
    try {
        m_impl->begin(step);
    } catch (...) {
        // Prefetching is only an optimization, the data is faulted in on access anyway.
    }
}

void PrefetchScheduler::end(size_t step) noexcept {
    try {
        m_impl->end(step);
    } catch (...) {
    }
}

const PrefetchScheduler::Stats& PrefetchScheduler::get_stats() const noexcept {
    return m_impl->m_stats;
}

}  // namespace ov::weight_sharing
