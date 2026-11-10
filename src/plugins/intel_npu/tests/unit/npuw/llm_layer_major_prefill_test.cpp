// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "just_sync_infer_request.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace ov::test::npuw {

struct LLMLayerMajorPrefillTestAccess {
    using Request = ov::npuw::LLMInferRequest;
    static uint64_t runs(const Request& req) {
        return req.m_layer_major_runs;
    }
    static auto& desc(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc;
    }
    static uint64_t present_tokens(const Request& req) {
        return req.m_tokens_in_present_chunk;
    }
    static const auto& names(const Request& req) {
        return req.m_kvcache_past_names;
    }
    static auto past(Request& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(name));
    }
    static auto present(Request& req, const std::string& name) {
        auto output = name;
        output.replace(0, std::strlen("past_key_values"), "present");
        return req.m_prefill_request->get_tensor(req.m_prefill_out_ports.at(output));
    }
    static auto base(Request& req) {
        return std::dynamic_pointer_cast<ov::npuw::JustInferRequest>(req.m_prefill_base_request);
    }
    static void chunk_major(Request& req) {
        req.m_use_layer_major_prefill = false;
    }
    static const auto& snapshots(const Request& req) {
        return req.m_layer_major_inputs;
    }
};

}  // namespace ov::test::npuw

namespace {

using Access = ov::test::npuw::LLMLayerMajorPrefillTestAccess;
using Request = ov::npuw::LLMInferRequest;

class CPUInferRequest : public ov::ISyncInferRequest {
public:
    CPUInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled, ov::InferRequest request)
        : ov::ISyncInferRequest(compiled),
          m_request(std::move(request)) {
        for (size_t i = 0; i < compiled->inputs().size(); ++i) {
            set_tensor(compiled->inputs()[i], ov::get_tensor_impl(m_request.get_input_tensor(i)));
        }
        for (size_t i = 0; i < compiled->outputs().size(); ++i) {
            set_tensor(compiled->outputs()[i], ov::get_tensor_impl(m_request.get_output_tensor(i)));
        }
    }
    void infer() override {
        for (size_t i = 0; i < get_inputs().size(); ++i) {
            m_request.set_input_tensor(i, ov::make_tensor(get_tensor(get_inputs()[i])));
        }
        for (size_t i = 0; i < get_outputs().size(); ++i) {
            m_request.set_output_tensor(i, ov::make_tensor(get_tensor(get_outputs()[i])));
        }
        m_request.infer();
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_request.get_profiling_info();
    }

private:
    ov::InferRequest m_request;
};

class CPUCompiledModel : public ov::ICompiledModel {
public:
    CPUCompiledModel(const std::shared_ptr<const ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     ov::CompiledModel compiled)
        : ov::ICompiledModel(model, plugin),
          m_compiled(std::move(compiled)) {}
    void export_model(std::ostream& stream) const override {
        m_compiled.export_model(stream);
    }
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_compiled.get_runtime_model();
    }
    void set_property(const ov::AnyMap& props) override {
        m_compiled.set_property(props);
    }
    ov::Any get_property(const std::string& name) const override {
        return m_compiled.get_property(name);
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<CPUInferRequest>(shared_from_this(), m_compiled.create_infer_request());
    }

private:
    mutable ov::CompiledModel m_compiled;
};

class LLMLayerMajorPrefillTest : public testing::TestWithParam<bool> {
protected:
    static constexpr size_t chunk = 16;
    std::shared_ptr<ov::Core> cpu = std::make_shared<ov::Core>();
    std::shared_ptr<ov::test::npuw::NullPlugin> plugin = std::make_shared<ov::test::npuw::NullPlugin>();
    std::shared_ptr<testing::NiceMock<ov::MockICore>> core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    std::shared_ptr<Request> reference, layer_major;

    void SetUp() override {
        using testing::_;
        ON_CALL(*core, get_property("NPU", ov::available_devices.name(), _))
            .WillByDefault(testing::Return(ov::Any(std::vector<std::string>{})));
        ON_CALL(*core, get_property("CPU", _, _))
            .WillByDefault([this](const std::string&, const std::string& name, const ov::AnyMap&) {
                return cpu->get_property("CPU", name);
            });
        ON_CALL(*core, get_supported_property(_, _, _))
            .WillByDefault([this](const std::string&, const ov::AnyMap& props, bool) {
                ov::AnyMap filtered;
                const auto supported = cpu->get_property("CPU", ov::supported_properties);
                for (const auto& prop : supported) {
                    if (prop.is_mutable() && props.count(prop)) {
                        filtered.emplace(prop, props.at(prop));
                    }
                }
                return filtered;
            });
        ON_CALL(
            *core,
            compile_model(testing::A<const std::shared_ptr<const ov::Model>&>(), testing::A<const std::string&>(), _))
            .WillByDefault([this](const std::shared_ptr<const ov::Model>& model,
                                  const std::string& device,
                                  const ov::AnyMap& props) {
                EXPECT_EQ(device, "CPU");
                auto config = props;
                config[ov::hint::inference_precision.name()] = ov::element::f32;
                config[ov::num_streams.name()] = 1;
                config[ov::inference_num_threads.name()] = 1;
                return ov::SoPtr<ov::ICompiledModel>{
                    std::make_shared<CPUCompiledModel>(model, plugin, cpu->compile_model(model, "CPU", config))};
            });
        plugin->set_core(core);
    }

    void compile(bool prefix = false,
                 const ov::element::Type& precision = ov::element::f32,
                 const std::string& attention = "STATIC",
                 const ov::AnyMap& extra_props = {}) {
        auto cfg = ov::test::npuw::make_test_model_config();
        cfg.num_layers = 4;
        cfg.precision = precision;
        if (attention != "STATIC") {
            cfg.num_kv_heads = 2;
            cfg.force_gqa_broadcast = true;
        }
        auto model = ov::test::npuw::build_llm_test_model(cfg);
        ov::AnyMap props{{"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM", "YES"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                         {"NPUW_LLM_PREFILL_CHUNK_SIZE", "16"},
                         {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                         {"NPUW_LLM_PREFILL_ATTENTION_HINT", attention},
                         {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                         {"NPUW_LLM_GENERATE_PYRAMID", "NO"},
                         {"NPUW_FOLD", "YES"},
                         {"NPUW_ONLINE_PIPELINE", "REP"},
                         {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                         {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
                         {"NPUW_FUNCALL_ASYNC", GetParam() ? "YES" : "NO"},
                         {"NPUW_LLM_ENABLE_PREFIX_CACHING", prefix ? "YES" : "NO"},
                         {"NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE", "16"}};
        props.insert(extra_props.begin(), extra_props.end());
        props["NPUW_LLM_PREFILL_LAYER_MAJOR"] = "NO";
        auto ref_model = std::make_shared<ov::npuw::LLMCompiledModel>(model->clone(), plugin, props);
        reference = std::make_shared<Request>(ref_model);
        props["NPUW_LLM_PREFILL_LAYER_MAJOR"] = "YES";
        auto lm_model = std::make_shared<ov::npuw::LLMCompiledModel>(model->clone(), plugin, props);
        layer_major = std::make_shared<Request>(lm_model);
        EXPECT_FALSE(ref_model->get_property("NPUW_LLM_PREFILL_LAYER_MAJOR").as<bool>());
        EXPECT_TRUE(lm_model->get_property("NPUW_LLM_PREFILL_LAYER_MAJOR").as<bool>());
        auto base = Access::base(*layer_major);
        ASSERT_NE(base, nullptr);
        ASSERT_GT(base->total_subrequests(), 2u);
        size_t calls = 0;
        for (size_t i = 0; i < base->total_subrequests(); ++i) {
            calls += base->is_repeated_function_call(i);
        }
        ASSERT_GT(calls, 0u);
    }

    void input(Request& req, size_t length, size_t position = 0, int64_t salt = 0) {
        const auto& ports = req.get_compiled_model()->inputs();
        for (const auto& port : ports) {
            const auto name = port.get_any_name();
            if (name == "input_ids" || name == "position_ids" || name == "attention_mask") {
                ov::Tensor tensor(ov::element::i64, {1, name == "attention_mask" ? position + length : length});
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    tensor.data<int64_t>()[i] = name == "attention_mask" ? 1
                                                : name == "position_ids" ? position + i
                                                                         : (i * 13 + salt + position) % 256;
                }
                req.set_tensor(port, ov::get_tensor_impl(tensor));
            }
        }
    }

    void bytes(const ov::SoPtr<ov::ITensor>& a, const ov::SoPtr<ov::ITensor>& b) {
        ASSERT_EQ(a->get_shape(), b->get_shape());
        ASSERT_EQ(a->get_element_type(), b->get_element_type());
        EXPECT_EQ(std::memcmp(a->data(), b->data(), a->get_byte_size()), 0);
    }
    static ov::SoPtr<ov::ITensor> logits(Request& req) {
        const auto port = ov::npuw::util::find_port_by_name(req.get_compiled_model()->outputs(), "logits");
        return req.get_tensor(port.value());
    }
    void compare() {
        bytes(logits(*reference), logits(*layer_major));
        EXPECT_EQ(Access::desc(*reference).num_stored_tokens, Access::desc(*layer_major).num_stored_tokens);
        EXPECT_EQ(Access::present_tokens(*reference), Access::present_tokens(*layer_major));
        for (const auto& name : Access::names(*reference)) {
            SCOPED_TRACE(name);
            bytes(Access::past(*reference, name), Access::past(*layer_major, name));
            bytes(Access::present(*reference, name), Access::present(*layer_major, name));
        }
    }
    void prefill(size_t length, bool fallback = false, int64_t salt = 0) {
        const auto runs = Access::runs(*layer_major);
        input(*reference, length, 0, salt);
        input(*layer_major, length, 0, salt);
        reference->infer();
        layer_major->infer();
        EXPECT_EQ(Access::runs(*layer_major), runs + (fallback ? 0 : 1));
        EXPECT_EQ(Access::desc(*layer_major).num_stored_tokens, length);
        EXPECT_EQ(Access::present_tokens(*layer_major), (length - 1) % chunk + 1);
        compare();
    }
    void attention_prefill(const std::string& hint) {
        compile(false, ov::element::f16, hint);
        auto base = Access::base(*layer_major);
        ASSERT_TRUE(base->layer_major_fallback_reason().empty());
        ASSERT_GT(base->layer_major_global_link_count(), 0u);
        size_t attention_subgraphs = 0;
        std::vector<ov::SoPtr<ov::IAsyncInferRequest>> primary, reserve;
        for (size_t i = 0; i < base->total_subrequests(); ++i) {
            const auto& context = base->subgraph_pipeline(i).context;
            attention_subgraphs += hint == "DYNAMIC" ? ov::npuw::attn::get_compiled_dynamic(context) != nullptr
                                                     : ov::npuw::attn::get_compiled_pyramid(context) != nullptr;
            primary.push_back(base->get_subrequest(i));
            reserve.push_back(base->get_pipeline_subrequest(i));
        }
        ASSERT_GT(attention_subgraphs, 0u);
        prefill(3 * chunk + 5);
        for (size_t i = 0; i < primary.size(); ++i) {
            EXPECT_EQ(base->get_subrequest(i)._ptr, primary[i]._ptr);
            EXPECT_EQ(base->get_pipeline_subrequest(i)._ptr, reserve[i]._ptr);
        }
        Access::chunk_major(*layer_major);
        prefill(2 * chunk + 3, true, 19);
    }
    void compile_hfa(const ov::element::Type& precision = ov::element::f16,
                     bool fused = false,
                     bool mask_skipping = false) {
        compile(false,
                precision,
                "HFA",
                {{"NPUW_ATTN_HFA_FUSED", fused ? "YES" : "NO"},
                 {"NPUW_ATTN_HFA_MASK_SKIPPING", mask_skipping ? "YES" : "NO"}});
        auto base = Access::base(*layer_major);
        ASSERT_TRUE(base->layer_major_fallback_reason().empty());
        size_t hfa_subgraphs = 0;
        for (size_t i = 0; i < base->total_subrequests(); ++i) {
            const auto* hfa = ov::npuw::attn::get_compiled_hfa(base->subgraph_pipeline(i).context);
            if (!hfa) {
                continue;
            }
            ++hfa_subgraphs;
            const auto mask_idx = hfa->_sdpa_attention_info._tile_input_indices.mask;
            EXPECT_EQ(hfa->_compiled_tile_model->inputs().size() > mask_idx, !mask_skipping);
        }
        ASSERT_GT(hfa_subgraphs, 0u);
    }
};

TEST_P(LLMLayerMajorPrefillTest, MultiChunkBitExact) {
    compile();
    prefill(3 * chunk + 5);
}
TEST_P(LLMLayerMajorPrefillTest, ExactMultipleOfChunk) {
    compile();
    prefill(2 * chunk);
}
TEST_P(LLMLayerMajorPrefillTest, SingleChunk) {
    compile();
    prefill(5);
}
TEST_P(LLMLayerMajorPrefillTest, GenerateAfterPrefill) {
    compile();
    const size_t length = 3 * chunk + 5;
    prefill(length);
    for (size_t step = 0; step < 3; ++step) {
        input(*reference, 1, length + step);
        input(*layer_major, 1, length + step);
        reference->infer();
        layer_major->infer();
        bytes(logits(*reference), logits(*layer_major));
    }
}
TEST_P(LLMLayerMajorPrefillTest, SecondPromptAfterReset) {
    compile();
    prefill(3 * chunk + 5);
    const auto snapshots = Access::snapshots(*layer_major);
    prefill(2 * chunk + 3, false, 17);
    ASSERT_EQ(Access::snapshots(*layer_major).size(), snapshots.size());
    for (size_t c = 0; c < snapshots.size(); ++c) {
        for (const auto& input : snapshots[c]) {
            EXPECT_EQ(Access::snapshots(*layer_major)[c].at(input.first)._ptr, input.second._ptr);
        }
    }
    Access::chunk_major(*layer_major);
    prefill(chunk + 7, true, 31);
}
TEST_P(LLMLayerMajorPrefillTest, FallbackWithPrefixCaching) {
    compile(true);
    prefill(3 * chunk + 5, true);
}
TEST_P(LLMLayerMajorPrefillTest, F16DynamicAttentionBitExact) {
    attention_prefill("DYNAMIC");
}
TEST_P(LLMLayerMajorPrefillTest, F16PyramidAttentionBitExact) {
    attention_prefill("PYRAMID");
}
TEST_P(LLMLayerMajorPrefillTest, HfaMultiChunkBitExact) {
    compile_hfa();
    prefill(3 * chunk + 5);
}
TEST_P(LLMLayerMajorPrefillTest, HfaExactMultipleOfChunk) {
    compile_hfa();
    prefill(2 * chunk);
}
TEST_P(LLMLayerMajorPrefillTest, HfaSingleChunk) {
    compile_hfa();
    prefill(5);
}
TEST_P(LLMLayerMajorPrefillTest, HfaF32BitExact) {
    compile_hfa(ov::element::f32);
    prefill(3 * chunk + 5);
}
TEST_P(LLMLayerMajorPrefillTest, HfaFusedBitExact) {
    compile_hfa(ov::element::f16, true);
    prefill(3 * chunk + 5);
}
TEST_P(LLMLayerMajorPrefillTest, HfaMaskSkippingBitExact) {
    compile_hfa(ov::element::f16, true, true);
    prefill(3 * chunk + 5);
}
TEST_P(LLMLayerMajorPrefillTest, HfaGenerateAfterPrefill) {
    compile_hfa();
    const size_t length = 3 * chunk + 5;
    prefill(length);
    for (size_t step = 0; step < 3; ++step) {
        input(*reference, 1, length + step);
        input(*layer_major, 1, length + step);
        reference->infer();
        layer_major->infer();
        bytes(logits(*reference), logits(*layer_major));
    }
}
TEST_P(LLMLayerMajorPrefillTest, HfaResetAndChunkMajorRestore) {
    compile_hfa();
    auto base = Access::base(*layer_major);
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> primary, reserve;
    for (size_t i = 0; i < base->total_subrequests(); ++i) {
        primary.push_back(base->get_subrequest(i));
        reserve.push_back(base->get_pipeline_subrequest(i));
    }
    prefill(3 * chunk + 5);
    prefill(2 * chunk + 3, false, 17);
    prefill(4 * chunk, false, 23);
    for (size_t i = 0; i < primary.size(); ++i) {
        EXPECT_EQ(base->get_subrequest(i)._ptr, primary[i]._ptr);
        EXPECT_EQ(base->get_pipeline_subrequest(i)._ptr, reserve[i]._ptr);
    }
    Access::chunk_major(*layer_major);
    prefill(chunk + 7, true, 31);
}

INSTANTIATE_TEST_SUITE_P(FunctionPipelining, LLMLayerMajorPrefillTest, testing::Bool());

}  // namespace
