// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {

// Forward declaration
class JustInferRequest;

// MoE prefill chunk task
struct MoEPrefillChunkTask {
    size_t idx;       // Sublayer index for MoE I/O lookup
    size_t real_idx;  // Real sublayer index for compiled model lookup
    size_t expert_id;
    size_t chunk_idx;
    size_t chunk_start_idx;
    size_t chunk_size;
    std::vector<size_t> token_ids;  // Tokens assigned to this expert

    // Tensors for this chunk
    ov::SoPtr<ov::ITensor> router_dest;
    ov::SoPtr<ov::ITensor> expert_input_dest;
    ov::SoPtr<ov::ITensor> expert_output;

    // Infer request handle
    ov::SoPtr<ov::IAsyncInferRequest> infer_request;
};

// Three-stage pipeline for MoE prefill inference
class MoEPrefillPipeline {
public:
    MoEPrefillPipeline(JustInferRequest* parent,
                       const ov::SoPtr<ov::ICompiledModel>& compiled_model,
                       size_t num_infer_requests);

    ~MoEPrefillPipeline();

    // Enqueue a chunk task for processing
    void enqueue(const MoEPrefillChunkTask& task);

    // Wait for all tasks to complete
    void wait_all();

    // Shutdown the pipeline
    void shutdown();

private:
    // Worker thread functions
    void preproc_worker();
    void infer_worker();
    void postproc_worker();

    // Parent request reference
    JustInferRequest* m_parent;

    // Infer request pool (queue holds ownership of all requests)
    std::queue<ov::SoPtr<ov::IAsyncInferRequest>> m_idle_requests;
    std::mutex m_idle_requests_mutex;
    std::condition_variable m_idle_requests_cv;

    // Track which expert is currently loaded in each infer request (use raw pointer as key)
    std::unordered_map<ov::IAsyncInferRequest*, size_t> m_request_loaded_expert;
    std::mutex m_request_expert_mutex;

    // Task queues
    std::queue<MoEPrefillChunkTask> m_preproc_queue;
    std::mutex m_preproc_mutex;
    std::condition_variable m_preproc_cv;
    static constexpr size_t MAX_QUEUE_DEPTH = 16;  // Maximum pending tasks in preproc queue

    std::queue<MoEPrefillChunkTask> m_infer_queue;
    std::mutex m_infer_mutex;
    std::condition_variable m_infer_cv;

    std::queue<MoEPrefillChunkTask> m_postproc_queue;
    std::mutex m_postproc_mutex;
    std::condition_variable m_postproc_cv;

    // Worker threads
    std::thread m_preproc_thread;
    std::thread m_infer_thread;
    std::thread m_postproc_thread;

    // Pipeline state
    std::atomic<bool> m_shutdown;
    std::atomic<size_t> m_tasks_enqueued;
    std::atomic<size_t> m_tasks_completed;
};

}  // namespace npuw
}  // namespace ov
