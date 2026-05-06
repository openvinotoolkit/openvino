// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_subgraph.hpp"

#include <array>
#include <sstream>
#include <unordered_map>

#include "../attention.hpp"
#include "../compiled_model.hpp"
#include "../host_flash_attention.hpp"
#include "../infer_request_utils.hpp"
#include "../just_sync_infer_request.hpp"
#include "../logging.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/sdpa.hpp"
#include "../pyramid_attention.hpp"
#include "../serialization.hpp"

namespace ov {
namespace npuw {
namespace attn {
namespace {

constexpr uint32_t ATTN_KV_DIM = 3;

struct BehaviorIO {
    std::vector<ov::SoPtr<ov::ITensor>> inputs;
    std::vector<ov::SoPtr<ov::ITensor>> outputs;
};

struct PyramidRequestSet {
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> infer_requests;
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> pipeline_requests;
    std::vector<ov::SoPtr<ov::ITensor>> anchors;
};

struct HFARequestSet {
    enum TileIdx : std::size_t {
        REGULAR_TILE = 0,
        FINAL_TILE = 1,
        COUNT = 2,
    };

    std::array<ov::SoPtr<ov::IAsyncInferRequest>, COUNT> infer_requests{};
    std::array<ov::SoPtr<ov::IAsyncInferRequest>, COUNT> pipeline_requests{};
};

struct RuntimeState {
    std::unordered_map<std::size_t, BehaviorIO> call_io;
    runtime::attention::Selector::Ptr attention_selector;
    runtime::pyramid_attention::Selector::Ptr pyramid_selector;
    runtime::host_flash_attention::Selector::Ptr hfa_selector;
    ov::SoPtr<ov::ITensor> cached_attention_mask;
    std::optional<runtime::host_flash_attention::HFARuntimeContext> hfa_runtime_ctx;
    PyramidRequestSet pyramid_requests;
    HFARequestSet hfa_requests;
    ov::SoPtr<ov::IAsyncInferRequest> base_request;
    ov::SoPtr<ov::IAsyncInferRequest> base_pipeline_request;
};

ov::npuw::JustInferRequest& get_request(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto* request = dynamic_cast<ov::npuw::JustInferRequest*>(&ctx.infer_request);
    OPENVINO_ASSERT(request != nullptr, "Expected JustInferRequest for attention runtime behavior");
    return *request;
}

BehaviorKind get_behavior_kind(const ov::npuw::v1::subgraphs::Context& ctx) {
    if (const auto* kind = ctx.get_if<BehaviorKind>()) {
        return *kind;
    }
    OPENVINO_THROW("Attention behavior context is missing BehaviorKind");
}

template <typename T>
T* get_compiled_state(ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
}

template <typename T>
const T* get_compiled_state(const ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
}

const ov::npuw::v1::subgraphs::CompiledPipeline& get_subgraph_pipeline(ov::npuw::v1::subgraphs::InferContext& ctx,
                                                                       std::size_t real_idx) {
    return get_request(ctx).subgraph_pipeline(real_idx);
}

const ov::SoPtr<ov::ICompiledModel>& get_compiled_submodel(ov::npuw::v1::subgraphs::InferContext& ctx,
                                                           std::size_t real_idx) {
    return get_request(ctx).compiled_submodel(real_idx);
}

std::size_t get_param_base(ov::npuw::v1::subgraphs::InferContext& ctx, std::size_t real_idx) {
    return get_request(ctx).subgraph_param_base(real_idx);
}

RuntimeState& get_runtime_state(ov::npuw::v1::subgraphs::InferContext& ctx) {
    OPENVINO_ASSERT(ctx.runtime_state != nullptr, "Expected runtime state storage for attention behavior");
    auto* state = ctx.runtime_state->get_if<RuntimeState>();
    if (state == nullptr) {
        state = &ctx.runtime_state->emplace<RuntimeState>();
    }
    return *state;
}

BehaviorIO& get_behavior_io(RuntimeState& state,
                            std::size_t subgraph_idx,
                            std::size_t num_inputs,
                            std::size_t num_outputs) {
    auto& io = state.call_io[subgraph_idx];
    if (io.inputs.size() != num_inputs) {
        io.inputs.resize(num_inputs);
    }
    if (io.outputs.size() != num_outputs) {
        io.outputs.resize(num_outputs);
    }
    return io;
}

void ensure_base_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    auto& request = get_request(ctx);
    if (!state.base_request) {
        state.base_request = request.get_subrequest(ctx.real_subgraph_idx);
    }
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx) && !state.base_pipeline_request) {
        state.base_pipeline_request = request.get_pipeline_subrequest(ctx.real_subgraph_idx);
    }
}

void ensure_dynamic_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.attention_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context);
    OPENVINO_ASSERT(dynamic != nullptr, "Missing compiled dynamic attention state");

    auto& request = get_request(ctx);
    if (!ctx.compiled_model.attention_dynamic_enabled()) {
        state.attention_selector = std::make_shared<runtime::attention::All>();
        return;
    }

    state.attention_selector = runtime::attention::PositionIDs::find(*dynamic, request);
    if (!state.attention_selector) {
        LOG_WARN("Dynamic capability is enabled, but no run-time features were found.");
        state.attention_selector = std::make_shared<runtime::attention::All>();
    }
}

void ensure_pyramid_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.pyramid_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context);
    OPENVINO_ASSERT(pyramid != nullptr, "Missing compiled pyramid attention state");

    auto& request = get_request(ctx);
    const auto pyramid_count = pyramid->_compiled_models.size();
    if (!ctx.compiled_model.attention_dynamic_enabled()) {
        state.pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
        return;
    }

    state.pyramid_selector = runtime::pyramid_attention::PositionIDs::find(*pyramid, request);
    if (!state.pyramid_selector) {
        LOG_WARN("Pyramid dynamic capability is enabled, but no run-time features were found.");
        state.pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
    }
}

void ensure_hfa_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.hfa_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context);
    OPENVINO_ASSERT(hfa != nullptr, "Missing compiled HFA state");

    auto& request = get_request(ctx);
    const size_t query_size = hfa->_sdpa_attention_info._query_size;
    state.hfa_selector = runtime::host_flash_attention::PositionIDs::find(query_size, request);
    if (!state.hfa_selector) {
        OPENVINO_THROW("HFA dynamic capability is enabled, but no run-time features were found.");
    }
}

void ensure_pyramid_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (!state.pyramid_requests.infer_requests.empty()) {
        return;
    }

    ensure_base_requests(ctx, state);

    const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context);
    OPENVINO_ASSERT(pyramid != nullptr, "Missing compiled pyramid attention state");

    auto& request = get_request(ctx);
    const auto& pyramid_models = pyramid->_compiled_models;
    const size_t num_pyramid_models = pyramid_models.size();
    const bool is_piped = request.is_subrequest_pipelined(ctx.real_subgraph_idx);

    state.pyramid_requests.infer_requests.resize(num_pyramid_models);
    if (is_piped) {
        state.pyramid_requests.pipeline_requests.resize(num_pyramid_models);
    }

    for (size_t model_idx = 0; model_idx + 1 < num_pyramid_models; ++model_idx) {
        state.pyramid_requests.infer_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
        if (is_piped) {
            state.pyramid_requests.pipeline_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
        }

        const size_t num_inputs = pyramid_models[model_idx]->inputs().size();
        OPENVINO_ASSERT(num_inputs == compiled_model->inputs().size(), "Unexpected pyramid input count mismatch");

        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            const auto pyramid_input = pyramid_models[model_idx]->inputs()[input_idx];
            const auto main_input = compiled_model->inputs()[input_idx];

            auto main_tensor_ptr = state.base_request->get_tensor(main_input)->data();
            auto pyramid_tensor = state.pyramid_requests.infer_requests[model_idx]->get_tensor(pyramid_input);
            auto shared_tensor = ov::get_tensor_impl(
                ov::Tensor(pyramid_tensor->get_element_type(), pyramid_tensor->get_shape(), main_tensor_ptr));
            state.pyramid_requests.infer_requests[model_idx]->set_tensor(pyramid_input, shared_tensor);

            if (is_piped) {
                auto pipeline_tensor = state.pyramid_requests.pipeline_requests[model_idx]->get_tensor(pyramid_input);
                auto pipeline_tensor_ptr = state.base_pipeline_request->get_tensor(main_input)->data();
                auto shared_pipeline_tensor = ov::get_tensor_impl(
                    ov::Tensor(pipeline_tensor->get_element_type(), pipeline_tensor->get_shape(), pipeline_tensor_ptr));
                state.pyramid_requests.pipeline_requests[model_idx]->set_tensor(pyramid_input, shared_pipeline_tensor);
            }
        }
    }

    if (num_pyramid_models > 0) {
        const size_t last_model_idx = num_pyramid_models - 1;
        state.pyramid_requests.infer_requests[last_model_idx] = state.base_request;
        if (is_piped) {
            state.pyramid_requests.pipeline_requests[last_model_idx] = state.base_pipeline_request;
        }

        const size_t num_inputs = compiled_model->inputs().size();
        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            const auto main_input = compiled_model->inputs()[input_idx];
            state.pyramid_requests.anchors.push_back(state.base_request->get_tensor(main_input));
            if (is_piped) {
                state.pyramid_requests.anchors.push_back(state.base_pipeline_request->get_tensor(main_input));
            }
        }
    }
}

void ensure_hfa_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]) {
        return;
    }

    ensure_base_requests(ctx, state);

    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context);
    OPENVINO_ASSERT(hfa != nullptr, "Missing compiled HFA state");

    auto& request = get_request(ctx);
    const bool is_piped = request.is_subrequest_pipelined(ctx.real_subgraph_idx);

    state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE] = hfa->_compiled_tile_model->create_infer_request();
    state.hfa_requests.infer_requests[HFARequestSet::FINAL_TILE] = state.base_request;
    if (is_piped) {
        state.hfa_requests.pipeline_requests[HFARequestSet::REGULAR_TILE] =
            hfa->_compiled_tile_model->create_infer_request();
        state.hfa_requests.pipeline_requests[HFARequestSet::FINAL_TILE] = state.base_pipeline_request;
    }

    const size_t num_inputs = hfa->_compiled_tile_model->inputs().size();
    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
        const auto tile_input = hfa->_compiled_tile_model->inputs()[input_idx];
        const auto final_tile_input = hfa->_compiled_final_tile_model->inputs()[input_idx];

        auto main_tensor = state.base_request->get_tensor(final_tile_input);
        state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->set_tensor(tile_input, main_tensor);

        if (is_piped) {
            auto pipeline_tensor = state.base_pipeline_request->get_tensor(final_tile_input);
            state.hfa_requests.pipeline_requests[HFARequestSet::REGULAR_TILE]->set_tensor(tile_input, pipeline_tensor);
        }
    }

    if (!state.hfa_runtime_ctx.has_value()) {
        state.hfa_runtime_ctx.emplace();
    }

    state.hfa_runtime_ctx->initialize_mask_cache(
        *hfa,
        request.subgraph_device(ctx.real_subgraph_idx),
        [&request](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
            return request.allocate_mem(dtype, shape, device);
        });

    const auto& tile_in = hfa->_sdpa_attention_info._tile_input_indices;
    auto state_acc = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.acc]);
    auto state_max = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.max]);
    auto state_sum = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.d]);

    runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc, state_max, state_sum);
    runtime::host_flash_attention::HFARuntimeContext::StateBuffers initial_buffers{state_acc, state_max, state_sum};
    state.hfa_runtime_ctx->initialize_state_buffers(
        initial_buffers,
        *hfa,
        request.subgraph_device(ctx.real_subgraph_idx),
        [&request](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
            return request.allocate_mem(dtype, shape, device);
        });
}

void prepare_dynamic(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& state = get_runtime_state(ctx);
    ensure_dynamic_selector(ctx, state);
    state.attention_selector->prepare(get_request(ctx).history_size());
    state.cached_attention_mask = {};
}

void prepare_pyramid(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& request = get_request(ctx);
    auto& state = get_runtime_state(ctx);
    ensure_pyramid_selector(ctx, state);
    ensure_pyramid_requests(ctx, state);
    state.pyramid_selector->prepare(request.history_size());
    state.cached_attention_mask = {};

    const auto pyramid_id = state.pyramid_selector->pyramid_id();
    request.set_active_subrequest(ctx.real_subgraph_idx, state.pyramid_requests.infer_requests.at(pyramid_id));
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx)) {
        request.set_pipeline_subrequest(ctx.real_subgraph_idx, state.pyramid_requests.pipeline_requests.at(pyramid_id));
    }
}

void prepare_hfa(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& request = get_request(ctx);
    auto& state = get_runtime_state(ctx);
    ensure_hfa_selector(ctx, state);
    ensure_hfa_requests(ctx, state);
    state.hfa_selector->prepare(request.history_size());
    state.cached_attention_mask = {};
    if (state.hfa_runtime_ctx) {
        state.hfa_runtime_ctx->clear_mask_cache();
    }
    request.set_active_subrequest(ctx.real_subgraph_idx, state.base_request);
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx)) {
        request.set_pipeline_subrequest(ctx.real_subgraph_idx, state.base_pipeline_request);
    }
}

void extract_and_copy_tile(const ov::SoPtr<ov::ITensor>& source_tensor,
                           const ov::SoPtr<ov::ITensor>& dest_tensor,
                           uint32_t sequence_dim,
                           int64_t sequence_offset,
                           int64_t sequence_length,
                           const std::string& tensor_name) {
    if (!dest_tensor->is_continuous()) {
        OPENVINO_THROW("HFA tile extraction error: destination tensor for '",
                       tensor_name,
                       "' is not continuous - cannot perform direct copy");
    }

    auto source_view = ov::npuw::util::view(source_tensor, sequence_dim, sequence_offset, sequence_length);
    const auto dest_type = dest_tensor->get_element_type();
    const auto source_type = source_tensor->get_element_type();

    if (dest_type == source_type) {
        ov::npuw::util::copy_tensor_by_dim(source_view, dest_tensor, sequence_dim, sequence_dim);
        return;
    }

    LOG_WARN("Performing type conversion for " << tensor_name << " tile: " << source_type << " -> " << dest_type);
    auto intermediate_tensor = ov::Tensor(source_type, source_view->get_shape());
    ov::npuw::util::copy_tensor_by_dim(source_view,
                                       ov::get_tensor_impl(intermediate_tensor),
                                       sequence_dim,
                                       sequence_dim);

    const size_t total_elements = intermediate_tensor.get_size();
    if (dest_type == ov::element::f32 && source_type == ov::element::f16) {
        auto src_data = intermediate_tensor.data<ov::float16>();
        auto dst_data = dest_tensor->data<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
        return;
    }
    if (dest_type == ov::element::f16 && source_type == ov::element::f32) {
        auto src_data = intermediate_tensor.data<float>();
        auto dst_data = dest_tensor->data<ov::float16>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<ov::float16>(src_data[i]);
        }
        return;
    }
    OPENVINO_THROW("Unsupported type conversion for ", tensor_name, " tile: ", source_type, " -> ", dest_type);
}

bool can_reuse_tensor_zero_copy(const ov::SoPtr<ov::ITensor>& source_tensor,
                                const ov::SoPtr<ov::ITensor>& dest_tensor,
                                uint32_t sequence_dim,
                                int64_t sequence_offset,
                                int64_t tile_length) {
    const auto source_shape = source_tensor->get_shape();
    const int64_t source_full_length = static_cast<int64_t>(source_shape[sequence_dim]);
    return (sequence_offset == 0 && tile_length == source_full_length &&
            dest_tensor->get_element_type() == source_tensor->get_element_type());
}

ov::npuw::v1::subgraphs::RuntimeBehaviorFactory make_runtime_factory() {
    return [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
        class AttnBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
        public:
            explicit AttnBehavior(BehaviorKind kind) : m_kind(kind) {}

            void prepare(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    prepare_dynamic(ctx);
                    return;
                case BehaviorKind::Pyramid:
                    prepare_pyramid(ctx);
                    return;
                case BehaviorKind::HFA:
                    prepare_hfa(ctx);
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            bool bind_function_input(ov::npuw::v1::subgraphs::InferContext& ctx,
                                     std::size_t input_idx,
                                     const ov::SoPtr<ov::ITensor>& tensor) override {
                auto& state = get_runtime_state(ctx);
                const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
                const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    if (const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context)) {
                        auto& io =
                            get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                        const bool is_non_param_mask = std::none_of(dynamic->params.begin(),
                                                                    dynamic->params.end(),
                                                                    [&](auto&& param) {
                                                                        return param.idx == input_idx;
                                                                    }) &&
                                                       input_idx != dynamic->mask_idx;
                        const auto& iport = compiled_model->inputs()[input_idx];
                        if (is_non_param_mask) {
                            ctx.target_request->set_tensor(iport, tensor);
                        } else {
                            io.inputs.at(input_idx) = tensor;
                        }
                        return true;
                    }
                    return false;
                case BehaviorKind::Pyramid:
                    if (const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context)) {
                        auto& io =
                            get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                        ensure_pyramid_selector(ctx, state);
                        const auto pyramid_id = state.pyramid_selector->pyramid_id();
                        const auto& info = pyramid->_attention_infos[pyramid_id];
                        const bool is_non_param_mask = std::none_of(info.params.begin(),
                                                                    info.params.end(),
                                                                    [&](auto&& param) {
                                                                        return param.idx == input_idx;
                                                                    }) &&
                                                       input_idx != info.mask_idx;
                        const auto& iport = compiled_model->inputs()[input_idx];
                        if (is_non_param_mask) {
                            ctx.target_request->set_tensor(iport, tensor);
                        } else {
                            io.inputs.at(input_idx) = tensor;
                        }
                        return true;
                    }
                    return false;
                case BehaviorKind::HFA:
                    if (const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context)) {
                        auto& io = get_behavior_io(state,
                                                   ctx.subgraph_idx,
                                                   get_param_base(ctx, ctx.real_subgraph_idx),
                                                   hfa->_compiled_final_tile_model->outputs().size());
                        io.inputs.at(input_idx) = tensor;
                        return true;
                    }
                    return false;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            bool bind_function_output(ov::npuw::v1::subgraphs::InferContext& ctx,
                                      std::size_t output_idx,
                                      const ov::SoPtr<ov::ITensor>& tensor) override {
                (void)ctx;
                (void)output_idx;
                (void)tensor;
                return false;
            }

            void prologue(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (ctx.opaque_prologue) {
                    ctx.opaque_prologue();
                }
                auto& state = get_runtime_state(ctx);
                const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
                const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
                auto& io = get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    if (const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context)) {
                        auto mask_iport = compiled_model->inputs()[dynamic->mask_idx];
                        const auto& graph_mask = io.inputs.at(dynamic->mask_idx);
                        const auto this_case = state.attention_selector->this_case();
                        auto pos_id = state.attention_selector->length();

                        if (pos_id == -1) {
                            ctx.target_request->set_tensor(mask_iport, graph_mask);
                            return;
                        }

                        const auto past_len = state.attention_selector->past_length();
                        const auto present_len = dynamic->query_size;
                        auto set_or_copy = [&](const ov::SoPtr<ov::ITensor>& view) {
                            if (!get_request(ctx).subgraph_needs_copy(ctx.subgraph_idx)) {
                                ctx.target_request->set_tensor(mask_iport, view);
                            } else {
                                const auto& dst = ctx.target_request->get_tensor(mask_iport);
                                dst->set_shape(view->get_shape());
                                view->copy_to(dst._ptr);
                            }
                        };

                        using namespace ov::npuw::runtime;
                        if (this_case == attention::Selector::Case::GENERATE) {
                            set_or_copy(ov::npuw::util::view(ov::get_tensor_impl(dynamic->attend_all),
                                                             ATTN_KV_DIM,
                                                             0,
                                                             past_len + 1));
                            return;
                        }
                        if (this_case == attention::Selector::Case::PREFILL) {
                            if (state.cached_attention_mask) {
                                ctx.target_request->set_tensor(mask_iport, state.cached_attention_mask);
                                return;
                            }

                            auto full_mask_shape = graph_mask->get_shape();
                            auto actual_mask_shape = full_mask_shape;
                            actual_mask_shape[ATTN_KV_DIM] = present_len + past_len;

                            const auto& dst = ctx.target_request->get_tensor(mask_iport);
                            dst->set_shape(actual_mask_shape);

                            const auto& present_dst_view =
                                ov::npuw::util::view(dst, ATTN_KV_DIM, past_len, present_len);
                            const auto& present_src_view =
                                ov::npuw::util::view(graph_mask,
                                                     ATTN_KV_DIM,
                                                     full_mask_shape[ATTN_KV_DIM] - present_len,
                                                     present_len);
                            present_src_view->copy_to(present_dst_view._ptr);

                            if (past_len > 0) {
                                const auto& past_dst_view = ov::npuw::util::view(dst, ATTN_KV_DIM, 0, past_len);
                                const auto& past_src_view = ov::npuw::util::view(graph_mask, ATTN_KV_DIM, 0, past_len);
                                past_src_view->copy_to(past_dst_view._ptr);
                            }
                            state.cached_attention_mask = dst;
                            return;
                        }
                    }
                    return;
                case BehaviorKind::Pyramid:
                    if (const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context)) {
                        const auto pyramid_id = state.pyramid_selector->pyramid_id();
                        const auto& dynamic = pyramid->_attention_infos[pyramid_id];
                        auto mask_iport = pyramid->_compiled_models[pyramid_id]->inputs()[dynamic.mask_idx];
                        const auto& graph_mask = io.inputs.at(dynamic.mask_idx);
                        const auto this_case = state.pyramid_selector->this_case();
                        const auto present_len = dynamic.query_size;
                        const auto& dst = ctx.target_request->get_tensor(mask_iport);

                        auto copy_mask_segment = [&](std::size_t dst_offset,
                                                     std::size_t src_offset,
                                                     std::size_t length) {
                            if (length == 0) {
                                return;
                            }
                            const auto& dst_view = ov::npuw::util::view(dst, ATTN_KV_DIM, dst_offset, length);
                            const auto& src_view = ov::npuw::util::view(graph_mask, ATTN_KV_DIM, src_offset, length);
                            ov::npuw::util::copy_tensor_by_dim(src_view, dst_view, ATTN_KV_DIM, ATTN_KV_DIM);
                        };

                        if (state.pyramid_selector->length() == -1) {
                            ctx.target_request->set_tensor(mask_iport, graph_mask);
                            return;
                        }

                        const auto past_len = state.pyramid_selector->past_length();
                        if (state.cached_attention_mask) {
                            ctx.target_request->set_tensor(mask_iport, state.cached_attention_mask);
                            return;
                        }

                        const auto full_mask_shape = graph_mask->get_shape();
                        using namespace ov::npuw::runtime;
                        if (this_case == pyramid_attention::Selector::Case::GENERATE) {
                            const auto dst_shape = dst->get_shape();
                            if (dst_shape == full_mask_shape) {
                                ctx.target_request->set_tensor(mask_iport, graph_mask);
                                state.cached_attention_mask = graph_mask;
                                return;
                            }

                            const std::size_t dst_present_offset = dst_shape[ATTN_KV_DIM] - present_len;
                            copy_mask_segment(dst_present_offset,
                                              full_mask_shape[ATTN_KV_DIM] - present_len,
                                              present_len);
                            copy_mask_segment(0, 0, dst_present_offset);
                            state.cached_attention_mask = dst;
                            return;
                        }
                        if (this_case == pyramid_attention::Selector::Case::PREFILL) {
                            copy_mask_segment(past_len, full_mask_shape[ATTN_KV_DIM] - present_len, present_len);
                            copy_mask_segment(0, 0, past_len);
                            state.cached_attention_mask = dst;
                            return;
                        }
                        OPENVINO_ASSERT(false, "Unsupported pyramid attention case");
                    }
                    return;
                case BehaviorKind::HFA:
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                case BehaviorKind::Pyramid:
                    ctx.legacy_infer();
                    return;
                case BehaviorKind::HFA:
                    if (const auto* hfa_desc = ov::npuw::attn::get_compiled_hfa(
                            get_subgraph_pipeline(ctx, ctx.real_subgraph_idx).context)) {
                        auto& state = get_runtime_state(ctx);
                        auto& io = get_behavior_io(state,
                                                   ctx.subgraph_idx,
                                                   get_param_base(ctx, ctx.real_subgraph_idx),
                                                   hfa_desc->_compiled_final_tile_model->outputs().size());

                        OPENVINO_ASSERT(hfa_desc->is_valid(), "HFA configuration must be valid");
                        const int64_t tile_size = hfa_desc->_tile_size;
                        const int64_t total_kv_length = state.hfa_selector->context_length();
                        const int64_t num_tiles = total_kv_length / tile_size;
                        OPENVINO_ASSERT(total_kv_length % tile_size == 0,
                                        "HFA total KV length must be multiple of tile size for now");

                        const auto& hfa_inputs = io.inputs;
                        const auto& sdpa_info = hfa_desc->_sdpa_attention_info;
                        const auto& sdpa_in = sdpa_info._sdpa_indices;

                        auto past_key_tensor = hfa_inputs.at(sdpa_in.past_key);
                        auto past_value_tensor = hfa_inputs.at(sdpa_in.past_value);
                        auto query_tensor = hfa_inputs.at(sdpa_in.query);
                        auto present_key_tensor = hfa_inputs.at(sdpa_in.present_key);
                        auto attention_mask_tensor = hfa_inputs.at(sdpa_in.attention_mask);
                        auto present_value_tensor = hfa_inputs.at(sdpa_in.present_value);
                        auto& regular_tile_request = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE];
                        auto& final_tile_request = state.hfa_requests.infer_requests[HFARequestSet::FINAL_TILE];
                        auto attention_output_tensor =
                            final_tile_request->get_tensor(hfa_desc->_compiled_final_tile_model->outputs()[0]);
                        const auto& tile_in = sdpa_info._tile_input_indices;
                        const auto& tile_out = sdpa_info._tile_output_indices;

                        ov::SoPtr<ov::ITensor> state_acc, state_max, state_sum;
                        if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                            const auto& current_buffer = state.hfa_runtime_ctx->get_current_state_buffers();
                            state_acc = current_buffer.acc;
                            state_max = current_buffer.max;
                            state_sum = current_buffer.sum;
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.acc],
                                                             state_acc);
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.max],
                                                             state_max);
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.d],
                                                             state_sum);
                        } else {
                            state_acc =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.acc]);
                            state_max =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.max]);
                            state_sum =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.d]);
                            runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc,
                                                                                                       state_max,
                                                                                                       state_sum);
                        }

                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.q],
                                                         query_tensor);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.q],
                                                       query_tensor);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.acc],
                                                         state_acc);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.max],
                                                         state_max);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.d],
                                                         state_sum);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.acc],
                                                       state_acc);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.max],
                                                       state_max);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.d],
                                                       state_sum);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->outputs()[0],
                                                       attention_output_tensor);

                        const uint32_t K_SEQ_DIM = static_cast<uint32_t>(sdpa_info._k_seq_dim);
                        const uint32_t V_SEQ_DIM = static_cast<uint32_t>(sdpa_info._v_seq_dim);
                        constexpr uint32_t MASK_KV_SEQ_DIM = 3;
                        size_t next_available_mask_buffer_idx = 0;

                        auto process_tile = [&](auto& request,
                                                auto& model,
                                                const ov::SoPtr<ov::ITensor>& k_source,
                                                const ov::SoPtr<ov::ITensor>& v_source,
                                                int64_t kv_offset,
                                                int64_t mask_offset,
                                                int64_t tile_length,
                                                bool async = false) {
                            auto k_tile_buffer = request->get_tensor(model->inputs()[tile_in.k]);
                            auto v_tile_buffer = request->get_tensor(model->inputs()[tile_in.v]);
                            auto mask_tile_buffer = request->get_tensor(model->inputs()[tile_in.mask]);

                            if (can_reuse_tensor_zero_copy(k_source,
                                                           k_tile_buffer,
                                                           K_SEQ_DIM,
                                                           kv_offset,
                                                           tile_length)) {
                                request->set_tensor(model->inputs()[tile_in.k], k_source);
                            } else if (hfa_desc->_can_use_tensor_view) {
                                request->set_tensor(model->inputs()[tile_in.k],
                                                    ov::npuw::util::view(k_source, K_SEQ_DIM, kv_offset, tile_length));
                            } else {
                                extract_and_copy_tile(k_source, k_tile_buffer, K_SEQ_DIM, kv_offset, tile_length, "K");
                            }

                            if (can_reuse_tensor_zero_copy(v_source,
                                                           v_tile_buffer,
                                                           V_SEQ_DIM,
                                                           kv_offset,
                                                           tile_length)) {
                                request->set_tensor(model->inputs()[tile_in.v], v_source);
                            } else if (hfa_desc->_can_use_tensor_view) {
                                request->set_tensor(model->inputs()[tile_in.v],
                                                    ov::npuw::util::view(v_source, V_SEQ_DIM, kv_offset, tile_length));
                            } else {
                                extract_and_copy_tile(v_source, v_tile_buffer, V_SEQ_DIM, kv_offset, tile_length, "V");
                            }

                            if (attention_mask_tensor) {
                                if (can_reuse_tensor_zero_copy(attention_mask_tensor,
                                                               mask_tile_buffer,
                                                               MASK_KV_SEQ_DIM,
                                                               mask_offset,
                                                               tile_length)) {
                                    request->set_tensor(model->inputs()[tile_in.mask], attention_mask_tensor);
                                } else if (state.hfa_runtime_ctx.has_value()) {
                                    auto cached_tile =
                                        state.hfa_runtime_ctx->find_cached_mask_tile(attention_mask_tensor,
                                                                                     mask_offset,
                                                                                     tile_length);
                                    if (cached_tile) {
                                        request->set_tensor(model->inputs()[tile_in.mask], cached_tile);
                                    } else {
                                        ov::SoPtr<ov::ITensor> cached_mask_tile =
                                            state.hfa_runtime_ctx->get_mask_tile_buffer(next_available_mask_buffer_idx);
                                        extract_and_copy_tile(attention_mask_tensor,
                                                              cached_mask_tile,
                                                              MASK_KV_SEQ_DIM,
                                                              mask_offset,
                                                              tile_length,
                                                              "Mask");
                                        state.hfa_runtime_ctx->cache_mask_tile(attention_mask_tensor,
                                                                               mask_offset,
                                                                               tile_length,
                                                                               cached_mask_tile);
                                        request->set_tensor(model->inputs()[tile_in.mask], cached_mask_tile);
                                        next_available_mask_buffer_idx++;
                                    }
                                } else {
                                    extract_and_copy_tile(attention_mask_tensor,
                                                          mask_tile_buffer,
                                                          MASK_KV_SEQ_DIM,
                                                          mask_offset,
                                                          tile_length,
                                                          "Mask");
                                }
                            }

                            if (async) {
                                request->start_async();
                                if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                                    state.hfa_runtime_ctx->prepare_next_state_buffers();
                                }
                                request->wait();
                            } else {
                                request->infer();
                            }
                        };

                        int64_t mask_tile_offset = 0;
                        int64_t kv_tile_offset = 0;
                        for (int64_t tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
                            process_tile(regular_tile_request,
                                         hfa_desc->_compiled_tile_model,
                                         past_key_tensor,
                                         past_value_tensor,
                                         kv_tile_offset,
                                         mask_tile_offset,
                                         tile_size);
                            kv_tile_offset += tile_size;
                            mask_tile_offset += tile_size;
                        }

                        if (num_tiles > 0) {
                            const size_t present_seq_length = present_key_tensor->get_shape()[K_SEQ_DIM];
                            const int64_t final_tile_length = static_cast<int64_t>(present_seq_length);
                            OPENVINO_ASSERT(
                                final_tile_length == tile_size,
                                "Final tile must process entire present KV sequence in a single inference. "
                                "This is guaranteed during compilation (tile_size = query_size = present_seq_length).");
                            const int64_t mask_total_length = attention_mask_tensor->get_shape()[MASK_KV_SEQ_DIM];
                            const int64_t final_mask_offset = mask_total_length - final_tile_length;
                            process_tile(final_tile_request,
                                         hfa_desc->_compiled_final_tile_model,
                                         present_key_tensor,
                                         present_value_tensor,
                                         0,
                                         final_mask_offset,
                                         final_tile_length,
                                         true);
                        }

                        if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                            state.hfa_runtime_ctx->switch_buffers();
                        }
                        return;
                    }
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

        private:
            BehaviorKind m_kind;
        };

        return std::make_unique<AttnBehavior>(get_behavior_kind(ctx));
    };
}

}  // namespace

void put_compiled_dynamic(v1::subgraphs::Context& context, CompiledDynamicState state) {
    context.put<CompiledDynamicState>(std::move(state));
}

void put_compiled_pyramid(v1::subgraphs::Context& context, CompiledPyramidState state) {
    context.put<CompiledPyramidState>(std::move(state));
}

void put_compiled_hfa(v1::subgraphs::Context& context, CompiledHFAState state) {
    context.put<CompiledHFAState>(std::move(state));
}

ov::npuw::compiled::Attention* get_compiled_dynamic(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

const ov::npuw::compiled::Attention* get_compiled_dynamic(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

const ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

const ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline) {
    return get_compiled_dynamic(pipeline.context) != nullptr || get_compiled_pyramid(pipeline.context) != nullptr ||
           get_compiled_hfa(pipeline.context) != nullptr;
}

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx) {
    std::optional<ov::npuw::compiled::Attention> dynamic;
    if (const auto* state = get_compiled_dynamic(context)) {
        dynamic = *state;
    }
    stream & dynamic;
    if (stream.input() && dynamic.has_value()) {
        put_compiled_dynamic(context, std::make_shared<ov::npuw::compiled::Attention>(dynamic.value()));
    }

    std::optional<ov::npuw::compiled::PyramidAttention> pyramid;
    if (const auto* state = get_compiled_pyramid(context)) {
        pyramid = *state;
    }
    stream & pyramid;
    if (stream.input() && pyramid.has_value()) {
        put_compiled_pyramid(context, std::make_shared<ov::npuw::compiled::PyramidAttention>(pyramid.value()));
    }

    auto* mutable_pyramid = get_compiled_pyramid(context);
    if (mutable_pyramid != nullptr) {
        size_t num_models = 0;
        if (stream.output()) {
            num_models = mutable_pyramid->_compiled_models.size();
        }
        stream & num_models;

        if (stream.output()) {
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::stringstream ss;
                mutable_pyramid->_compiled_models[i]->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            }
        } else if (num_models > 0) {
            mutable_pyramid->_compiled_models.resize(num_models);
            NPUW_ASSERT(submodel_ctx != nullptr);
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_pyramid->_compiled_models[i] =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
            }
            if (submodel_ctx->compiled_model) {
                mutable_pyramid->_compiled_models[num_models - 1] = submodel_ctx->compiled_model;
                LOG_DEBUG("Reused compiled_model for the last pyramid attention model");
            }
        }
    }

    std::optional<ov::npuw::compiled::HostFlashAttention> hfa;
    if (const auto* state = get_compiled_hfa(context)) {
        hfa = *state;
    }
    stream & hfa;
    if (stream.input() && hfa.has_value()) {
        put_compiled_hfa(context, std::make_shared<ov::npuw::compiled::HostFlashAttention>(hfa.value()));
    }

    auto* mutable_hfa = get_compiled_hfa(context);
    if (mutable_hfa != nullptr) {
        bool has_compiled_model = false;
        if (stream.output()) {
            has_compiled_model = mutable_hfa->_compiled_tile_model != nullptr;
        }
        stream & has_compiled_model;
        if (has_compiled_model) {
            if (stream.output()) {
                std::stringstream ss;
                mutable_hfa->_compiled_tile_model->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            } else {
                NPUW_ASSERT(submodel_ctx != nullptr);
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_hfa->_compiled_tile_model =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
                LOG_DEBUG("Imported compiled tile model for host flash attention");
            }
        }
        if (stream.input()) {
            NPUW_ASSERT(submodel_ctx != nullptr);
            mutable_hfa->_compiled_final_tile_model = submodel_ctx->compiled_model;
            LOG_DEBUG("Set compiled final tile model reference for host flash attention");
        }
    }
}

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             BehaviorKind kind) {
    compiled_pipeline.registration.group = ov::npuw::patterns::attn::SDPA::group_name();
    compiled_pipeline.registration.name = ov::npuw::patterns::attn::SDPA::pattern_name();
    compiled_context.put<BehaviorKind>(kind);
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
    spec.registration = compiled_pipeline.registration;
    spec.context = compiled_context;
    spec.factory = make_runtime_factory();
    spec.handles_function_prologue = true;
    compiled_pipeline.runtime_behavior = std::move(spec);
}

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry) {
    std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> registrations;
    registrations.reserve(3);

    // Matcher-only registrations so the registry handles SDPA/SDPADecomposed isolation
    // (replaces the legacy HNDL_ATTN fallback in snapshot.cpp)
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPA>().scoped());
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPADecomposed>().scoped());

    // Behavior registration: fires for any function tagged "attn" (i.e. from either SDPA
    // or SDPADecomposed pattern).  The at_partition callback checks whether dynamic
    // attention was detected (f._attention set by Partitioner::attention()) and, if so,
    // stashes the compile-time descriptor in the pipeline context for later stages.
    ov::npuw::v1::subgraphs::PatternRegistration attn_behavior;
    attn_behavior.tag = ov::npuw::patterns::attn::SDPA::isolation_tag();
    attn_behavior.partition_stage = [](ov::npuw::Function& f, ov::npuw::v1::subgraphs::Context& ctx) {
        if (f._attention.has_value()) {
            put_compiled_dynamic(ctx, std::make_shared<ov::npuw::compiled::Attention>(f._attention.value(), f._model));
            ctx.put<BehaviorKind>(BehaviorKind::Dynamic);
            return;
        }
        if (f._pyramid_attention.has_value()) {
            put_compiled_pyramid(ctx,
                                 std::make_shared<ov::npuw::compiled::PyramidAttention>(f._pyramid_attention.value()));
            ctx.put<BehaviorKind>(BehaviorKind::Pyramid);
            return;
        }
        if (f._host_flash_attention.has_value()) {
            put_compiled_hfa(ctx,
                             std::make_shared<ov::npuw::compiled::HostFlashAttention>(f._host_flash_attention.value()));
            ctx.put<BehaviorKind>(BehaviorKind::HFA);
        }
    };
    attn_behavior.compile_stage = [](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                                     ov::npuw::v1::subgraphs::Context& compiled_context) {
        const auto* kind = compiled_context.get_if<BehaviorKind>();
        if (kind == nullptr) {
            return;
        }
        attach_runtime_behavior(compiled_pipeline, compiled_context, *kind);
    };
    registrations.emplace_back(registry.add(std::move(attn_behavior)));

    return registrations;
}

}  // namespace attn
}  // namespace npuw
}  // namespace ov
