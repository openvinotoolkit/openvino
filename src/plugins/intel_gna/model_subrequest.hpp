// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>

#include "request_status.hpp"

namespace GNAPluginNS {

class ModelSubrequest {
public:
    using EnqueueHandler = std::function<uint32_t()>;
    using WaitHandler = std::function<RequestStatus(uint32_t request_id, int64_t timeout_milliseconds)>;

    ModelSubrequest(EnqueueHandler enqueue_handler, WaitHandler wait_handler);

    ModelSubrequest(const ModelSubrequest&) = default;
    ModelSubrequest(ModelSubrequest&&) = default;
    ModelSubrequest& operator=(const ModelSubrequest&) = default;
    ModelSubrequest& operator=(ModelSubrequest&&) = default;

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

}  // namespace GNAPluginNS
