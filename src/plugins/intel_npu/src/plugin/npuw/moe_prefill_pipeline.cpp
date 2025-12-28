// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_prefill_pipeline.hpp"

#include "compiled_model.hpp"
#include "just_sync_infer_request.hpp"
#include "logging.hpp"

namespace ov {
namespace npuw {

MoEPrefillPipeline::MoEPrefillPipeline(JustInferRequest* parent,
                                       const ov::SoPtr<ov::ICompiledModel>& compiled_model,
                                       size_t num_infer_requests)
    : m_parent(parent),
      m_shutdown(false),
      m_tasks_enqueued(0),
      m_tasks_completed(0) {
    // Initialize infer request pool
    for (size_t i = 0; i < num_infer_requests; ++i) {
        // Create a new infer request for this expert
        auto new_request = compiled_model->create_infer_request();
        m_idle_requests.push(new_request);
    }

    // Start worker threads
    m_preproc_thread = std::thread(&MoEPrefillPipeline::preproc_worker, this);
    m_infer_thread = std::thread(&MoEPrefillPipeline::infer_worker, this);
    m_postproc_thread = std::thread(&MoEPrefillPipeline::postproc_worker, this);
}

MoEPrefillPipeline::~MoEPrefillPipeline() {
    shutdown();
}

void MoEPrefillPipeline::enqueue(const MoEPrefillChunkTask& task) {
    {
        std::unique_lock<std::mutex> lock(m_preproc_mutex);
        // Wait if queue is full (backpressure mechanism)
        m_preproc_cv.wait(lock, [this]() {
            return m_shutdown || m_preproc_queue.size() < MAX_QUEUE_DEPTH;
        });

        if (m_shutdown) {
            return;  // Don't enqueue if shutting down
        }

        m_preproc_queue.push(task);
        m_tasks_enqueued++;
    }
    m_preproc_cv.notify_one();
}

void MoEPrefillPipeline::wait_all() {
    // Wait until all enqueued tasks are completed
    while (m_tasks_completed < m_tasks_enqueued) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void MoEPrefillPipeline::shutdown() {
    if (m_shutdown.exchange(true)) {
        return;  // Already shut down
    }

    // Wake up all worker threads
    m_preproc_cv.notify_all();
    m_infer_cv.notify_all();
    m_postproc_cv.notify_all();
    m_idle_requests_cv.notify_all();

    // Join all threads
    if (m_preproc_thread.joinable()) {
        m_preproc_thread.join();
    }
    if (m_infer_thread.joinable()) {
        m_infer_thread.join();
    }
    if (m_postproc_thread.joinable()) {
        m_postproc_thread.join();
    }

    // Infer requests will be automatically cleaned up when m_idle_requests is destroyed
}

void MoEPrefillPipeline::preproc_worker() {
    while (!m_shutdown) {
        MoEPrefillChunkTask task;

        // Get task from preproc queue
        {
            std::unique_lock<std::mutex> lock(m_preproc_mutex);
            m_preproc_cv.wait(lock, [this]() {
                return m_shutdown || !m_preproc_queue.empty();
            });

            if (m_shutdown && m_preproc_queue.empty()) {
                return;
            }

            task = std::move(m_preproc_queue.front());
            m_preproc_queue.pop();
        }
        // Notify enqueue that there's space available
        m_preproc_cv.notify_one();

        // Get an idle infer request
        ov::SoPtr<ov::IAsyncInferRequest> infer_request;
        {
            std::unique_lock<std::mutex> lock(m_idle_requests_mutex);
            m_idle_requests_cv.wait(lock, [this]() {
                return m_shutdown || !m_idle_requests.empty();
            });

            if (m_shutdown) {
                return;
            }

            infer_request = m_idle_requests.front();
            m_idle_requests.pop();
        }

        task.infer_request = infer_request;

        // Unpack expert weights if this infer request doesn't have this expert loaded
        bool need_unpack = false;
        {
            std::lock_guard<std::mutex> lock(m_request_expert_mutex);
            auto* req_ptr = infer_request._ptr.get();  // Get raw pointer from SoPtr's _ptr member
            auto it = m_request_loaded_expert.find(req_ptr);
            if (it == m_request_loaded_expert.end() || it->second != task.expert_id) {
                need_unpack = true;
                m_request_loaded_expert[req_ptr] = task.expert_id;
            }
        }

        if (need_unpack) {
            m_parent->unpack_moe_expert_closure(task.idx, infer_request, task.expert_id);
        }

        // Get compiled model and MoE configuration
        auto compiled_model = infer_request->get_compiled_model();
        auto& comp_model_desc = m_parent->m_npuw_model->m_compiled_submodels[task.real_idx];
        const auto& moe_experts = comp_model_desc.moe_experts.value();

        // Get input/output ports using proper indices from MoE configuration
        NPUW_ASSERT(moe_experts._router_scores_idx.has_value());
        NPUW_ASSERT(moe_experts._expert_input_param_idx.has_value());

        const auto& oport = compiled_model->outputs()[0];
        const auto& router_iport = compiled_model->inputs()[moe_experts._router_scores_idx.value()];
        const auto& expert_input_iport = compiled_model->inputs()[moe_experts._expert_input_param_idx.value()];

        // Get tensors from the infer request
        task.router_dest = infer_request->get_tensor(router_iport);
        task.expert_input_dest = infer_request->get_tensor(expert_input_iport);
        task.expert_output = infer_request->get_tensor(oport);

        // Get source tensors from parent's MoE I/O cache
        auto router_source = m_parent->m_moe_io[task.idx].router_scores;
        auto expert_input_source = m_parent->m_moe_io[task.idx].expert_input;

        // Perform gather operations using parent's helper methods
        m_parent->gather_router_scores(router_source,
                                       task.router_dest,
                                       task.expert_id,
                                       task.token_ids,
                                       task.chunk_start_idx,
                                       task.chunk_size);

        m_parent->gather_expert_inputs(expert_input_source,
                                       task.expert_input_dest,
                                       task.token_ids,
                                       task.chunk_start_idx,
                                       task.chunk_size);

        // Pass to infer queue
        {
            std::unique_lock<std::mutex> lock(m_infer_mutex);
            m_infer_queue.push(std::move(task));
        }
        m_infer_cv.notify_one();
    }
}

void MoEPrefillPipeline::infer_worker() {
    while (!m_shutdown) {
        MoEPrefillChunkTask task;

        // Get task from infer queue
        {
            std::unique_lock<std::mutex> lock(m_infer_mutex);
            m_infer_cv.wait(lock, [this]() {
                return m_shutdown || !m_infer_queue.empty();
            });

            if (m_shutdown && m_infer_queue.empty()) {
                return;
            }

            task = std::move(m_infer_queue.front());
            m_infer_queue.pop();
        }

        // Start async inference (tensors are already set in preproc worker)
        task.infer_request->start_async();

        // Immediately pass to postproc queue (async handoff)
        {
            std::unique_lock<std::mutex> lock(m_postproc_mutex);
            m_postproc_queue.push(std::move(task));
        }
        m_postproc_cv.notify_one();
    }
}

void MoEPrefillPipeline::postproc_worker() {
    while (!m_shutdown) {
        MoEPrefillChunkTask task;

        // Get task from postproc queue
        {
            std::unique_lock<std::mutex> lock(m_postproc_mutex);
            m_postproc_cv.wait(lock, [this]() {
                return m_shutdown || !m_postproc_queue.empty();
            });

            if (m_shutdown && m_postproc_queue.empty()) {
                return;
            }

            task = std::move(m_postproc_queue.front());
            m_postproc_queue.pop();
        }

        // Wait for inference to complete
        task.infer_request->wait();

        // Get necessary parameters for scattering
        auto output_shape = task.expert_output->get_shape();
        size_t embed_dim = (output_shape.size() == 4) ? output_shape[3] : output_shape[1];

        // Get input token count from parent's MoE I/O cache
        auto expert_input_source = m_parent->m_moe_io[task.idx].expert_input;
        auto input_shape = expert_input_source->get_shape();
        size_t input_token_count = (input_shape.size() == 4) ? input_shape[2] : input_shape[0];

        // Scatter expert outputs back to global buffer using parent's helper
        m_parent->scatter_expert_outputs(task.expert_output,
                                         task.token_ids,
                                         task.chunk_start_idx,
                                         task.chunk_size,
                                         task.expert_id,
                                         embed_dim,
                                         input_token_count,
                                         m_parent->m_moe_token_to_experts);

        // Return infer request to idle pool
        {
            std::unique_lock<std::mutex> lock(m_idle_requests_mutex);
            m_idle_requests.push(task.infer_request);
        }
        m_idle_requests_cv.notify_one();

        // Mark task as completed
        m_tasks_completed++;
    }
}

}  // namespace npuw
}  // namespace ov
