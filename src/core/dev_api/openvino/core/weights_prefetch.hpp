// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov::op::v0 {
class Constant;
}  // namespace ov::op::v0

namespace ov::weight_sharing {

/**
 * @brief Prefetches the data of constants just ahead of the place they are read, without growing
 * the resident set size beyond a fixed budget.
 *
 * The consumer describes the order it reads constants in as a sequence of steps (e.g. nodes of a
 * graph in execution order), then brackets the processing of every step with @ref begin and
 * @ref end. The scheduler keeps two windows ahead of the current step:
 *  - populate window (@ref Config::populate_bytes): data is mapped into the process, so it counts
 *    towards the resident set size; the window is the only RSS overhead versus no prefetch at all;
 *  - readahead window (@ref Config::readahead_bytes): data is only read into the OS page cache,
 *    which is free in terms of resident set size. Opt-in: on fast storage the far reads compete
 *    with the near ones for the bandwidth and do not pay off.
 *
 * After the last step reading a constant, its pages may be released from the process
 * (@ref Use::evict_after_last_use) when the consumer has copied the data elsewhere.
 *
 * Constants whose data is not memory mapped are handled as well, prefetching them is a cheap no-op.
 * The scheduler is not thread safe: a single thread drives it.
 */
class OPENVINO_API PrefetchScheduler {
public:
    struct OPENVINO_API Config {
        size_t populate_bytes = 64UL << 20;  //!< Bytes mapped into the process ahead of the current step.
        size_t readahead_bytes = 0;          //!< Bytes read into the page cache beyond the populate window.
        std::string site;                    //!< Call site name, used for diagnostics.
        bool trace = false;                  //!< Print @ref Stats to stderr on destruction.

        /**
         * @brief Reads the configuration of the given call site from the environment.
         *
         * OV_WEIGHTS_PREFETCH=1 enables the feature, OV_WEIGHTS_PREFETCH_SITES=<comma separated list>
         * restricts it to some call sites, OV_WEIGHTS_PREFETCH_POPULATE_MB and
         * OV_WEIGHTS_PREFETCH_READAHEAD_MB override the budgets, OV_WEIGHTS_PREFETCH_TRACE=1 enables
         * diagnostics.
         *
         * @param site     Name of the call site.
         * @param defaults Budgets suitable for the call site, the environment overrides them. Sites
         *                 whose data stays resident anyway may afford a larger populate window.
         * @return Configuration or std::nullopt when prefetch is disabled for the site.
         */
        static std::optional<Config> from_env(std::string_view site, const Config& defaults);
        static std::optional<Config> from_env(std::string_view site);
    };

    /** @brief A constant read entirely within a step. */
    struct Use {
        std::weak_ptr<ov::op::v0::Constant> constant;
        bool evict_after_last_use = false;  //!< Data is not read again after the last step using it.
    };

    using Step = std::vector<Use>;
    using Plan = std::vector<Step>;

    /** @brief Counters for diagnostics and tests. */
    struct Stats {
        size_t regions = 0;               //!< Distinct memory regions in the plan.
        size_t populated = 0;             //!< Regions populated ahead of their step.
        size_t read_ahead = 0;            //!< Regions read into the page cache ahead of their step.
        size_t late = 0;                  //!< Regions reached before any prefetch was issued for them.
        size_t waited = 0;                //!< Regions reached while their prefetch was still running.
        size_t evicted = 0;               //!< Regions released after their last use.
        size_t peak_populated_ahead = 0;  //!< Max bytes populated ahead of the current step.
        size_t peak_read_ahead = 0;       //!< Max bytes read ahead beyond the populate window.
    };

    PrefetchScheduler(const Plan& plan, const Config& config);
    ~PrefetchScheduler();

    PrefetchScheduler(const PrefetchScheduler&) = delete;
    PrefetchScheduler& operator=(const PrefetchScheduler&) = delete;

    /**
     * @brief Marks the start of a step: the data of the step constants is resident when the call
     * returns, and no background work touches it anymore.
     */
    void begin(size_t step) noexcept;

    /** @brief Marks the end of a step: releases the constants read for the last time and moves the windows on. */
    void end(size_t step) noexcept;

    const Stats& get_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::weight_sharing

namespace ov {
namespace wsh = ov::weight_sharing;
}  // namespace ov
