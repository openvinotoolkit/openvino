// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_infer_request.hpp"

namespace intel_npu {

/**
 * @brief Proof of concept, see docs/execution-list-proposal.md.
 * Records the graphs of several infer requests into one command list, separated by barriers, and submits them with
 * a single fence. Each graph execution has its own mutable command id, so a request whose tensors move is updated in
 * place instead of re-recording the list.
 */
class ZeroExecutionList {
public:
    // Lists with the same non-null queue_owner share one command queue; nullptr gives the list its own
    ZeroExecutionList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const void* queue_owner = nullptr);

    ZeroExecutionList(const ZeroExecutionList&) = delete;
    ZeroExecutionList& operator=(const ZeroExecutionList&) = delete;

    void append(const std::shared_ptr<ZeroInferRequest>& request);
    void prepare();
    void start_async();
    void wait();
    void reset();

private:
    struct Argument {
        uint32_t index = 0;  // Used by the driver
        const void* data = nullptr;
        std::vector<size_t> strides;  // In elements, innermost first; empty for a plain tensor

        bool operator==(const Argument& other) const {
            return index == other.index && data == other.data && strides == other.strides;
        }
    };
    using Arguments = std::vector<Argument>;

    struct Member {
        std::shared_ptr<ZeroInferRequest> request;
        uint64_t command_id = 0;
        Arguments arguments;
    };

    Arguments collect_arguments(const ZeroInferRequest& request) const;
    void record();
    // Returns false if the list has to be recorded again
    bool update_arguments();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    const void* _queue_owner;
    std::vector<Member> _members;

    std::shared_ptr<CommandQueue> _command_queue;
    std::unique_ptr<CommandList> _command_list;
    std::unique_ptr<Fence> _fence;
    bool _running = false;

    // For the logs
    uint64_t _submissions = 0;
    uint64_t _recordings = 0;
    std::chrono::steady_clock::time_point _submit_time;

    Logger _logger;
};

}  // namespace intel_npu
