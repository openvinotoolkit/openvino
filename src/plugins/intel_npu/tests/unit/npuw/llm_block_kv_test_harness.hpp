// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared harness for request-level tests of the block-based KV cache.
//
// The tests drive a real LLMCompiledModel and LLMInferRequest on top of fake
// sub-requests, so the KV pool is populated exactly as it is on a device while no
// device is needed. The fake outputs are deterministic, finite and depend only on the
// port name: two models with the same port names therefore produce the same KV bytes,
// which lets a test compare a run against a control run.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "executor.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_prefix_caching.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov::test::npuw {

// Reaches the block KV strategy of a request and the pool it owns.
struct LLMBlockKvTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static ov::npuw::LLMBlockKVCacheStrategy& block_strategy(Request& req) {
        auto* strategy = dynamic_cast<ov::npuw::LLMBlockKVCacheStrategy*>(req.m_kvcache_strategy.get());
        OPENVINO_ASSERT(strategy != nullptr, "the request does not run on the block KV strategy");
        return *strategy;
    }

    static const std::unordered_map<uint32_t, ov::npuw::LayerBlockManagers>& block_managers(
        const ov::npuw::LLMBlockKVCacheStrategy& strategy) {
        return strategy.m_kv_cache_block_managers;
    }

    static uint32_t block_size(const ov::npuw::LLMBlockKVCacheStrategy& strategy) {
        return strategy.m_block_size;
    }

    // The tensor the generate request currently reads the named past input from.
    static ov::SoPtr<ov::ITensor> generate_past(Request& req, const std::string& name) {
        return req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
    }

    // The tensor the prefill request currently reads the named past input from.
    static ov::SoPtr<ov::ITensor> prefill_past(Request& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(name));
    }
};

inline void fill_pattern(const ov::SoPtr<ov::ITensor>& tensor, size_t seed) {
    const size_t count = tensor->get_size();
    const auto type = tensor->get_element_type();
    if (type == ov::element::f16) {
        auto* data = tensor->data<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = ov::float16(std::sin(static_cast<float>(i + seed) * 0.137f) * 1.25f);
        }
    } else if (type == ov::element::f32) {
        auto* data = tensor->data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = std::sin(static_cast<float>(i + seed) * 0.137f) * 1.25f;
        }
    } else {
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class PatternedSubCompiledModel;

class PatternedSubInferRequest final : public ov::ISyncInferRequest {
public:
    explicit PatternedSubInferRequest(std::shared_ptr<const PatternedSubCompiledModel> compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
};

class PatternedSubCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    PatternedSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<const ov::IPlugin>& plugin,
                              const ov::AnyMap&)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const PatternedSubCompiledModel>(shared_from_this());
        return std::make_shared<PatternedSubInferRequest>(std::move(self));
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        ::intel_npu::make_executor("block_kv_task", 1),
                                                        ::intel_npu::make_executor("block_kv_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        // One CPU submodel, so init_pre_alloc_device() resolves to CPU staging
        // instead of requesting NPU remote tensors from the stub plugin.
        return 1;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}

private:
    std::shared_ptr<ov::Model> m_model;
};

inline PatternedSubInferRequest::PatternedSubInferRequest(
    std::shared_ptr<const PatternedSubCompiledModel> compiled_model)
    : ov::ISyncInferRequest(std::move(compiled_model)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(
            output,
            ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

inline void PatternedSubInferRequest::infer() {
    for (const auto& output : get_compiled_model()->outputs()) {
        fill_pattern(ov::ISyncInferRequest::get_tensor(output), std::hash<std::string>{}(output.get_any_name()));
    }
}

inline ov::npuw::LLMCompiledModel::CompiledModelFactory make_patterned_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
              const std::shared_ptr<const ov::IPlugin>& plugin,
              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<PatternedSubCompiledModel>(model, plugin, props);
    };
}

inline ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

inline ov::Tensor make_i64_iota(std::initializer_list<size_t> shape, int64_t start) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), start);
    return tensor;
}

inline std::vector<float> to_floats(const ov::SoPtr<ov::ITensor>& tensor) {
    ov::Tensor dense(tensor->get_element_type(), tensor->get_shape());
    tensor->copy_to(ov::get_tensor_impl(dense)._ptr);
    std::vector<float> out(dense.get_size());
    if (dense.get_element_type() == ov::element::f16) {
        const auto* data = dense.data<ov::float16>();
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<float>(data[i]);
        }
    } else {
        const auto* data = dense.data<float>();
        std::copy_n(data, out.size(), out.begin());
    }
    return out;
}

// One conversation driver over a freshly compiled model. The context is 64 prompt
// tokens plus 32 response tokens.
class LLMKvTestPipeline {
public:
    explicit LLMKvTestPipeline(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& extra_props = {}) {
        m_plugin = std::make_shared<NullPlugin>();
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "32"}};
        for (const auto& [key, value] : extra_props) {
            props[key] = value;
        }
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, m_plugin, props, make_patterned_factory());
        m_request = std::make_unique<ov::npuw::LLMInferRequest>(m_compiled);
    }

    ov::npuw::LLMInferRequest& request() {
        return *m_request;
    }

    const ov::npuw::LLMCompiledModel& compiled() const {
        return *m_compiled;
    }

    void prefill(size_t prompt_len, int64_t first_position, int64_t token = 1) {
        set_inputs(make_i64({1, prompt_len}, token),
                   make_i64({1, prompt_len}, 1),
                   make_i64_iota({1, prompt_len}, first_position));
        m_request->infer();
    }

    void generate_step(size_t live_tokens, int64_t position) {
        set_inputs(make_i64({1, 1}, 1), make_i64({1, live_tokens + 1}, 1), make_i64({1, 1}, position));
        m_request->infer();
    }

private:
    void set_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) {
        const auto& inputs = m_request->get_inputs();
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "input_ids").value(),
                              ov::get_tensor_impl(input_ids));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "attention_mask").value(),
                              ov::get_tensor_impl(attention_mask));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "position_ids").value(),
                              ov::get_tensor_impl(position_ids));
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::unique_ptr<ov::npuw::LLMInferRequest> m_request;
};

// Settings and observation helpers common to the block KV tests.
class LLMBlockKvTest : public ::testing::Test {
protected:
    // The prefill chunk size, which is also the KV block size.
    static constexpr uint32_t kBlock = 32;
    // A prompt of two full blocks, which is also the whole prompt capacity of
    // LLMKvTestPipeline.
    static constexpr size_t kTwoBlocks = 2 * kBlock;

    using Managers = std::unordered_map<uint32_t, ov::npuw::LayerBlockManagers>;

    // Block size follows the chunk size. The generate model's past KV is 95 rows, so two
    // numbered blocks cover positions 0..63 and a tail block covers the rest.
    static ov::AnyMap block_props() {
        return {{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                {"NPUW_LLM_PREFILL_CHUNK_SIZE", std::to_string(kBlock)},
                {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                {"NPUW_LLM_GENERATE_ATTENTION_HINT", "PYRAMID"},
                {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}};
    }

    static ov::AnyMap prefix_props() {
        auto props = block_props();
        props["NPUW_LLM_ENABLE_PREFIX_CACHING"] = "YES";
        return props;
    }

    static const Managers& managers(ov::npuw::LLMInferRequest& req) {
        return LLMBlockKvTestAccess::block_managers(LLMBlockKvTestAccess::block_strategy(req));
    }

    static std::vector<uint64_t> hashes() {
        return std::vector<uint64_t>(kBlock, 7u);
    }

    // Caches the block of the request's pool that starts at `token_start`.
    static std::shared_ptr<ov::npuw::KVBlock> store_block(ov::npuw::LLMInferRequest& req, size_t token_start) {
        return LLMBlockKvTestAccess::block_strategy(req).make_prefix_block(token_start, 0u, kBlock, hashes());
    }

    static const ov::npuw::PrefixBlockBlockKV& as_pool_block(const std::shared_ptr<ov::npuw::KVBlock>& block) {
        const auto* pool_block = dynamic_cast<const ov::npuw::PrefixBlockBlockKV*>(block.get());
        EXPECT_NE(pool_block, nullptr);
        return *pool_block;
    }
};

}  // namespace ov::test::npuw
