// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "llm_infer_request.hpp"

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {
std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (int i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout  = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat  = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval  = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> cvt_kvcache_to_fp16(const std::shared_ptr<ov::Model>& model) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (auto tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    for (auto tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    return ppp.build();
}

struct KVAxesPosition {
    uint32_t batch;
    uint32_t seq_len;
};

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const KVAxesPosition& kv_axes_position) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

ov::AnyMap get_baseline_common_config() {
    ov::AnyMap config = {
        { "NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm" },
        { "NPUW_DEVICES", "NPU,CPU" },
        { "NPU_USE_NPUW",  "YES" },
        { "NPUW_FOLD", "YES" },
        { "NPUW_DCOFF_TYPE", "f16" },
        { "NPUW_DCOFF_SCALE", "YES"},
        { "NPUW_WEIGHTS_BANK", "shared" },
        { "NPUW_SLICE_OUT", "YES" },
        { "NPUW_FUNCALL_ASYNC", "YES" }
    };
    return config;
}

} // namespace

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin), orig_model(model) {
    std::cout << "[LOG_DEBUG] LLMCompiledModel::LLMCompiledModel" << std::endl;

    // (2) Expose KV-cache input and output layers from kvcache model
    auto kvcache_model = model->clone();
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);

    kvcache_model = redirect_new_kv_to_output(kvcache_model);
    kvcache_model = cvt_kvcache_to_fp16(kvcache_model);

    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    KVAxesPosition axes{0u, 2u};
    reshape_to_static(prefill_model, 1024, 1024, axes);
    reshape_to_static(kvcache_model, 1u, 1024+128, axes);

    // FIXME: Should be compiled w/o accessing OpenVINO high-level API as we already know it should be NPUW

    auto generate_config = get_baseline_common_config();
    auto prefill_config = get_baseline_common_config();

    kvcache_compiled = std::make_shared<ov::npuw::CompiledModel>(kvcache_model, plugin, generate_config);
    prefill_compiled = std::make_shared<ov::npuw::CompiledModel>(prefill_model, plugin, generate_config);

    std::cout << "[LOG_DEBUG] LLMCompiledModel - done " << std::endl;
}

void ov::npuw::LLMCompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::npuw::LLMCompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::LLMCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::npuw::LLMCompiledModel::get_property(const std::string& name) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_sync_infer_request() const {
    auto* non_const_this = const_cast<ov::npuw::LLMCompiledModel*>(this);  // because of const in API
    return non_const_this->create_llm_infer_request();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_llm_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr);
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::LLMCompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    return std::make_shared<ov::IAsyncInferRequest>(internal_request, get_task_executor(), get_callback_executor());
}
