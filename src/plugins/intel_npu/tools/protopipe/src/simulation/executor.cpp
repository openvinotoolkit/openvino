//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"
#include "utils/error.hpp"

#include <chrono>

PipelinedExecutor::PipelinedExecutor(cv::GStreamingCompiled&& compiled): m_compiled(std::move(compiled)) {
}

PipelinedExecutor::Output PipelinedExecutor::runLoop(cv::GRunArgs&& inputs, Callback callback,
                                                     ITermCriterion::Ptr criterion) {
    if (!criterion) {
        THROW_ERROR("Termination criterion hasn't been specified!");
    }

    using namespace std::chrono;
    using clock_t = high_resolution_clock;

    m_compiled.setSource(std::move(inputs));
    criterion->init();

    const auto start_tick = clock_t::now();
    m_compiled.start();
    while (criterion->check()) {
        if (!callback(m_compiled)) {
            break;
        }
        criterion->update();
    }
    const auto end_tick = clock_t::now();
    // NB: Some frames might be in queue just wait until they processed.
    // They shouldn't be taken into account since execution is over.
    m_compiled.stop();
    return Output{static_cast<uint64_t>(duration_cast<microseconds>(end_tick - start_tick).count())};
}

SyncExecutor::SyncExecutor(cv::GCompiled&& compiled): m_compiled(std::move(compiled)) {
}

SyncExecutor::Output SyncExecutor::runLoop(Callback callback, ITermCriterion::Ptr criterion) {
    if (!criterion) {
        THROW_ERROR("Termination criterion hasn't been specified!");
    }

    using namespace std::chrono;
    using clock_t = high_resolution_clock;

    const auto start_tick = clock_t::now();
    criterion->init();
    while (criterion->check()) {
        if (!callback(m_compiled)) {
            break;
        }
        criterion->update();
    }
    const auto end_tick = clock_t::now();
    return Output{static_cast<uint64_t>(duration_cast<microseconds>(end_tick - start_tick).count())};
}

void SyncExecutor::reset() {
    m_compiled.prepareForNewStream();
}
