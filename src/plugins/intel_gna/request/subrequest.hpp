// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>

#include "request_status.hpp"

namespace GNAPluginNS {
namespace request {

class Subrequest {
public:
    using EnqueueHandler = std::function<uint32_t()>;
    using WaitHandler = std::function<RequestStatus(uint32_t request_id, int64_t timeout_milliseconds)>;

    Subrequest(EnqueueHandler enqueue_handler, WaitHandler wait_handler);

    Subrequest(const Subrequest&) = default;
    Subrequest(Subrequest&&) = default;
    Subrequest& operator=(const Subrequest&) = default;
    Subrequest& operator=(Subrequest&&) = default;

    RequestStatus wait(int64_t timeout_miliseconds);
    void enqueue();

    bool is_pending() const;
    bool is_aborted() const;
    bool is_completed() const;

private:
    RequestStatus status_{RequestStatus::kNone};

    uint32_t request_id_{0};
    EnqueueHandler enqueue_handler_;
    WaitHandler wait_handler_;
};

}  // namespace request
}  // namespace GNAPluginNS
