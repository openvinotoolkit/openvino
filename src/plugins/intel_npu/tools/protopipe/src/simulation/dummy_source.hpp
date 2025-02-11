//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <thread>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/streaming/source.hpp>  // cv::gapi::wip::IStreamSource

#include "utils/timer.hpp"
#include "utils/utils.hpp"

class DummySource final : public cv::gapi::wip::IStreamSource {
public:
    using Ptr = std::shared_ptr<DummySource>;

    explicit DummySource(const uint64_t frames_interval_in_us, const bool drop_frames,
                         const bool disable_high_resolution_timer);

    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;
    void reset();

private:
    uint64_t m_latency_in_us;
    bool m_drop_frames;
    IWaitable::Ptr m_timer;

    cv::Mat m_mat;
    int64_t m_next_tick_ts = -1;
    int64_t m_curr_seq_id = 0;
};
