// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <vector>

#include "compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace tp {

InferRequest::InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(compiled_model) {
    const auto& rank_compiled = m_compiled_model->get_rank_compiled();

    // Create one infer request per rank.
    m_rank_requests.reserve(rank_compiled.size());
    for (const auto& rank_model : rank_compiled) {
        m_rank_requests.push_back(rank_model->create_infer_request());
    }

    // Pre-allocate TP model output tensors (check_tensors() needs non-null).
    for (const auto& output : compiled_model->outputs()) {
        auto ps = output.get_partial_shape();
        ov::Shape shape;
        if (ps.is_static()) {
            shape = ps.get_shape();
        } else if (ps.rank().is_static()) {
            shape.resize(ps.rank().get_length(), 0);
            for (int64_t d = 0; d < ps.rank().get_length(); ++d) {
                shape[d] = ps[d].is_static() ? ps[d].get_length() : 0;
            }
        } else {
            shape = {0};
        }
        allocate_tensor(output, [&](ov::SoPtr<ov::ITensor>& t) {
            t = ov::make_tensor(output.get_element_type(), shape);
        });
    }
}

void InferRequest::infer() {
    const auto& rank_compiled = m_compiled_model->get_rank_compiled();
    const size_t num_ranks = rank_compiled.size();

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    // 1. Set user inputs on all rank requests.
    const auto& user_inputs = m_compiled_model->inputs();
    const auto& rank0_inputs = m_rank_requests[0]->get_compiled_model()->inputs();

    for (size_t i = 0; i < user_inputs.size(); ++i) {
        auto tensor = get_tensor(user_inputs[i]);
        for (auto& req : m_rank_requests) {
            req->set_tensor(rank0_inputs[i], tensor);
        }
    }

    auto t1 = clock::now();

    // 2. Launch all ranks in parallel.
    if (num_ranks == 1) {
        m_rank_requests[0]->infer();
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(num_ranks);
        for (size_t rank = 0; rank < num_ranks; ++rank) {
            futures.push_back(std::async(std::launch::async, [this, rank]() {
                m_rank_requests[rank]->infer();
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    }

    auto t2 = clock::now();

    // 3. Collect outputs from rank 0.
    const auto& outputs = m_compiled_model->outputs();
    const auto& rank0_outputs = m_rank_requests[0]->get_compiled_model()->outputs();

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = m_rank_requests[0]->get_tensor(rank0_outputs[i]);
        set_tensor(outputs[i], tensor);
    }

    auto t3 = clock::now();

    double ms_set = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ms_collect = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cerr << "[TP] Infer breakdown: set_inputs=" << ms_set
              << "ms  infer=" << ms_infer
              << "ms  collect=" << ms_collect
              << "ms  total=" << (ms_set + ms_infer + ms_collect) << "ms\n";
}

std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    return m_rank_requests[0]->query_state();
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return m_rank_requests[0]->get_profiling_info();
}

}  // namespace tp
}  // namespace ov
