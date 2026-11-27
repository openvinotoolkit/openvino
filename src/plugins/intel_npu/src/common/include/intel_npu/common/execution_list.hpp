// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"

namespace intel_npu {

/**
 * @brief Proof of concept, see docs/execution-list-proposal.md.
 * Runs a sequence of infer requests as a single submission. Requests run in the order they were appended.
 * @note The interface only uses virtual methods so that it can be used across the plugin library boundary.
 */
class IExecutionList {
public:
    /**
     * @brief Appends a request. The request must be created by the NPU plugin for a static model. While the list is
     * running, the request must not be used on its own.
     */
    virtual void append(const std::shared_ptr<ov::IAsyncInferRequest>& request) = 0;

    /**
     * @brief Prepares the tensors of all requests and records the command list, without submitting it. Optional: it
     * moves the one-time recording out of the first start_async(). Requests whose tensors are bound later get them
     * as updates of the recorded commands. Requests appended afterwards are not allowed.
     */
    virtual void prepare() = 0;

    /**
     * @brief Prepares the tensors of all requests and submits them. The command list is recorded on the first call,
     * unless prepare() already recorded it.
     */
    virtual void start_async() = 0;

    /**
     * @brief Waits until all requests have finished and makes their outputs available.
     */
    virtual void wait() = 0;

    /**
     * @brief Removes all requests so that the list can be filled again, e.g. with a different sequence of requests
     * for the next submission. The next start_async() records a new command list. The removed requests still can't be
     * used on their own.
     */
    virtual void reset() = 0;

    virtual ~IExecutionList() = default;
};

/**
 * @brief Implemented by the NPU remote context.
 */
class IExecutionListFactory {
public:
    /**
     * @param queue_owner Lists created with the same non-null owner submit on one shared command queue; with
     * nullptr the list gets a queue of its own. The queue is not shared with the graphs' own queues either way.
     */
    virtual std::shared_ptr<IExecutionList> create_execution_list(const void* queue_owner = nullptr) = 0;

    virtual ~IExecutionListFactory() = default;
};

}  // namespace intel_npu
