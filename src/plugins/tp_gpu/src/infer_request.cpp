// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace tp_gpu {

namespace {

// Fan-out wrapper that broadcasts reset()/set_state() to every per-rank state
// sharing the same variable_id.  Without this, calling state.reset() only
// affects rank 0 — rank 1's KV cache silently keeps accumulating across
// inference calls, producing massive iter-to-iter slowdowns and wrong
// numerics on the second+ infer.
class FanOutVariableState : public ov::IVariableState {
public:
    FanOutVariableState(const std::string& name,
                        std::vector<ov::SoPtr<ov::IVariableState>> per_rank)
        : ov::IVariableState(name), m_per_rank(std::move(per_rank)) {}

    void reset() override {
        static const bool dbg = std::getenv("TP_DBG") != nullptr;
        if (dbg) {
            // Verify the per-rank wrappers are distinct objects (paired by
            // name, not aliased to a single rank).  If two pointers ever
            // matched, reset() would only affect one rank — a critical bug.
            for (size_t r = 1; r < m_per_rank.size(); ++r) {
                if (m_per_rank[r]._ptr == m_per_rank[0]._ptr) {
                    std::cerr << "[TP][STATE][BUG] FanOut '" << get_name()
                              << "' rank " << r
                              << " aliases rank 0 (same pointer)\n";
                }
            }
        }
        // Diagnostic toggle: TP_NO_RESET=1 makes reset() a complete no-op
        // on every rank.  Used only to verify whether the iter-to-iter
        // slowdown is caused by reset() side-effects (e.g. the GPU plugin's
        // VariableState::reset() invokes m_shape_predictor->reset(), which
        // wipes prefetch shape history for ALL primitives in the network,
        // forcing the next infer through the cold dynamic-allocation path).
        // If iter2 is fast with TP_NO_RESET=1, the slowdown is reset-induced
        // and not an actual model-state issue.
        static const bool no_reset = std::getenv("TP_NO_RESET") != nullptr;
        if (no_reset) {
            return;
        }
        static const bool prof = std::getenv("TP_PROF") != nullptr;
        using clk = std::chrono::steady_clock;
        // Aggregate reset timings across all variables, dump once per
        // batch (state.reset() is typically called for ~num_layers states
        // back-to-back; reporting per-variable would be too noisy).
        static thread_local std::vector<double> agg_ms;
        static thread_local size_t agg_n = 0;
        auto t0 = clk::now();
        for (auto& s : m_per_rank)
            s->reset();
        if (prof) {
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (agg_ms.size() < m_per_rank.size()) agg_ms.assign(m_per_rank.size(), 0.0);
            agg_ms[0] += ms; // single timer covers all ranks (sequential calls)
            ++agg_n;
            if (agg_n % 32 == 0) {
                std::cerr << "[TP][STATE] reset agg over " << agg_n
                          << " calls: total_ms=" << agg_ms[0] << "\n";
            }
        }
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        // NOTE: assumes the caller-supplied tensor is valid for every rank.
        // KV-cache state is sharded per rank in TP; full-tensor set_state is
        // not meaningful here, but reset()-then-fresh-prefill is the typical
        // path and is fully covered by reset() above.
        for (auto& s : m_per_rank)
            s->set_state(state);
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        // Returns rank 0's shard.  Full reconstruction across ranks would
        // require a gather and isn't needed for current use cases.
        return m_per_rank.empty() ? ov::SoPtr<ov::ITensor>{} : m_per_rank.front()->get_state();
    }

private:
    std::vector<ov::SoPtr<ov::IVariableState>> m_per_rank;
};

}  // namespace

InferRequest::InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(compiled_model) {
    const auto& rank_compiled = m_compiled_model->get_rank_compiled();

    // Create one infer request per rank.
    m_rank_requests.reserve(rank_compiled.size());
    for (const auto& rank_model : rank_compiled) {
        m_rank_requests.push_back(rank_model->create_infer_request());
    }

    // Pre-allocate tensors for every port.  Callers are allowed to read a
    // tensor back before they have ever set one -- GenAI's stateful LLM
    // pipeline does exactly that with `get_tensor("attention_mask").set_shape()`
    // at the start of every generate() -- and the base class hands out a null
    // SoPtr until something is stored.  Dynamic dimensions start at 0, so the
    // tensor is empty until the caller reshapes or replaces it.
    auto allocate_port = [this](const ov::Output<const ov::Node>& port) {
        const auto& ps = port.get_partial_shape();
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
        allocate_tensor(port, [&](ov::SoPtr<ov::ITensor>& t) {
            t = ov::make_tensor(port.get_element_type(), shape);
        });
    };

    for (const auto& input : compiled_model->inputs()) {
        allocate_port(input);
    }
    for (const auto& output : compiled_model->outputs()) {
        allocate_port(output);
    }
}

void InferRequest::infer() {
    // A CompiledModel owns one coordinator shared by all of its requests.
    // Serialize complete outer inferences so two requests cannot mix ranks
    // in the same rendezvous epoch or overwrite shared L0 command lists.
    [[maybe_unused]] auto inference_guard = m_compiled_model->lock_inference();

    const auto& rank_compiled = m_compiled_model->get_rank_compiled();
    const size_t num_ranks = rank_compiled.size();
    static const bool profiling_enabled = std::getenv("TP_PROF") != nullptr;

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
    std::vector<double> per_rank_ms(profiling_enabled ? num_ranks : 0, 0.0);
    if (num_ranks == 1) {
        auto r0 = clock::now();
        m_rank_requests[0]->infer();
        if (profiling_enabled) {
            per_rank_ms[0] = std::chrono::duration<double, std::milli>(clock::now() - r0).count();
        }
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(num_ranks);
        for (size_t rank = 0; rank < num_ranks; ++rank) {
            futures.push_back(std::async(std::launch::async, [this, rank, &per_rank_ms]() {
                auto r0 = clock::now();
                m_rank_requests[rank]->infer();
                if (!per_rank_ms.empty()) {
                    per_rank_ms[rank] = std::chrono::duration<double, std::milli>(clock::now() - r0).count();
                }
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

    if (profiling_enabled) {
        double ms_set = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double ms_collect = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cerr << "[TP] Infer breakdown: set_inputs=" << ms_set
                  << "ms  infer=" << ms_infer
                  << "ms  collect=" << ms_collect
                  << "ms  total=" << (ms_set + ms_infer + ms_collect) << "ms";
        for (size_t r = 0; r < per_rank_ms.size(); ++r) {
            std::cerr << "  r" << r << "=" << per_rank_ms[r] << "ms";
        }
        std::cerr << "\n";
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    if (m_rank_requests.size() == 1)
        return m_rank_requests[0]->query_state();

    std::lock_guard<std::mutex> lk(m_state_mutex);
    if (!m_fanout_states.empty())
        return m_fanout_states;

    // Build a name -> per-rank-list map.  std::unordered_map iteration order
    // is implementation-defined and not guaranteed to match across two
    // independent map instances even when they hold identical keys; the GPU
    // plugin's query_state() returns states by iterating its own
    // std::unordered_map<std::string, ...>, so pairing rank-r and rank-0
    // states by VECTOR INDEX is unsafe.  We pair by NAME instead.
    std::unordered_map<std::string, std::vector<ov::SoPtr<ov::IVariableState>>>
        grouped;
    size_t expected = m_rank_requests[0]->query_state().size();
    grouped.reserve(expected);

    for (size_t r = 0; r < m_rank_requests.size(); ++r) {
        auto rs = m_rank_requests[r]->query_state();
        OPENVINO_ASSERT(rs.size() == expected,
                        "[TP] per-rank state count mismatch: rank ", r,
                        " has ", rs.size(), " states, expected ", expected);
        for (auto& s : rs) {
            grouped[s->get_name()].push_back(s);
        }
    }

    m_fanout_states.reserve(grouped.size());
    for (auto& kv : grouped) {
        OPENVINO_ASSERT(kv.second.size() == m_rank_requests.size(),
                        "[TP] variable '", kv.first,
                        "' present on only ", kv.second.size(),
                        " of ", m_rank_requests.size(), " ranks");
        m_fanout_states.emplace_back(std::make_shared<FanOutVariableState>(
            kv.first, std::move(kv.second)));
    }
    return m_fanout_states;
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return m_rank_requests[0]->get_profiling_info();
}

}  // namespace tp_gpu
}  // namespace ov
