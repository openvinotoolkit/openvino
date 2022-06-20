// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>

enum class GNARequestWaitStatus;

namespace GNAPluginNS {

class ModelSubrequest {
public:
    enum class Status { kReady, kOngoing, kCompleted, kAborted };
    using EnqueueHandler = std::function<uint32_t(uint32_t config_id)>;
    using WaitHandler = std::function<GNARequestWaitStatus(uint32_t request_id, int64_t timeout_milliseconds)>;

    ModelSubrequest(uint64_t request_config_id, EnqueueHandler enqueue_handler, WaitHandler wait_handler);

    ModelSubrequest(const ModelSubrequest&) = default;
    ModelSubrequest(ModelSubrequest&&) = default;
    ModelSubrequest& operator=(const ModelSubrequest&) = default;
    ModelSubrequest& operator=(ModelSubrequest&&) = default;

    Status wait(int64_t timeout_miliseconds);
    void enqueue();

    uint64_t request_config_id() const;
    uint32_t request_id() const;

    bool is_ongoing() const;
    bool is_aborted() const;
    bool is_ready() const;
    bool is_completed() const;

private:
    Status status_{Status::kReady};

    uint64_t request_config_id_;
    uint32_t request_id_{0};
    EnqueueHandler enqueue_handler_;
    WaitHandler wait_handler_;
};

}  // namespace GNAPluginNS
