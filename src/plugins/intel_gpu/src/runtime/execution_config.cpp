// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/search_sorted.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/properties.hpp"


namespace ov::intel_gpu {

namespace {

ov::RTMap get_rt_info(const ov::Model& model) {
    ov::RTMap rt_info;
    if (model.has_rt_info("runtime_options"))
        rt_info = model.get_rt_info<ov::AnyMap>("runtime_options");

    if (model.has_rt_info("__weights_path")) {
        rt_info[ov::weights_path.name()] = model.get_rt_info<ov::Any>("__weights_path");
    }
    return rt_info;
}

bool is_llm(const ov::Model& model) {
    using namespace ov::pass::pattern;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto convert_past = wrap_type<ov::op::v0::Convert>({past});
    auto gather_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past});
    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto gather_convert = wrap_type<ov::op::v0::Convert>({gather_past});
    auto concat_past_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past, gather_past, gather_convert});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_past_input, any_input()});
    auto convert_present = wrap_type<ov::op::v0::Convert>({concat});
    auto present_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{concat, convert_present});
    auto present = wrap_type<ov::op::v6::Assign>({present_input});

    auto kvcache_matcher = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheMatcher");

    for (auto& op : model.get_ordered_ops()) {
        if (kvcache_matcher->match(op) || ov::is_type<ov::op::PagedAttentionExtension>(op)) {
            return true;
        }
    }

    return false;
}

} // namespace

ExecutionConfig::ExecutionConfig() : ov::PluginConfig() { }

ExecutionConfig::ExecutionConfig(const ExecutionConfig& other) : ExecutionConfig() {
    m_user_properties = other.m_user_properties;
    m_is_finalized = other.m_is_finalized;
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }
}

ExecutionConfig& ExecutionConfig::operator=(const ExecutionConfig& other) {
    m_user_properties = other.m_user_properties;
    m_is_finalized = other.m_is_finalized;
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }
    return *this;
}

ExecutionConfig ExecutionConfig::clone() const {
    ExecutionConfig new_config = *this;
    new_config.m_is_finalized = false;
    return new_config;
}

void ExecutionConfig::finalize(cldnn::engine& engine) {
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    PluginConfig::finalize(ctx.get(), nullptr);
}

void ExecutionConfig::apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info, bool is_llm) {
    const auto& info = dynamic_cast<const RemoteContextImpl*>(context)->get_engine().get_device_info();
    if (!info.supports_immad) {
        apply_rt_info_property(ov::hint::kv_cache_precision, rt_info);
    }
    if (!is_llm)
        apply_rt_info_property(ov::hint::activations_scale_factor, rt_info);

    apply_rt_info_property(ov::hint::dynamic_quantization_group_size, rt_info);

    // WEIGHTS_PATH is used for the weightless cache mechanism which is used only with
    // ov::CacheMode::OPTIMIZE_SIZE setting. Not setting WEIGHTS_PATH will result in not
    // using that mechanism.
    if (get_cache_mode() == ov::CacheMode::OPTIMIZE_SIZE) {
        apply_rt_info_property(ov::weights_path, rt_info);
    }
}

void ExecutionConfig::apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) {
    apply_rt_info(context, get_rt_info(model), is_llm(model));

    const auto& ops = model.get_ops();

    std::function<void(std::shared_ptr<Node>)> process_op = [&, this](std::shared_ptr<Node> op) {
        // In the case of dynamic models, because most of the layers are mapped to shape agnostic kernels,
        // smaller # of kernels are built compared to static models.
        // So having smaller batch size is even better for dynamic model as we can do more parallel build.
        if (op->is_dynamic()) {
            m_max_kernels_per_batch = 4;
        }

        // Allow using onednn for models with LSTMSequence op as it's much more performant than existing ocl impl
        if (ov::is_type<ov::op::v5::LSTMSequence>(op)) {
            m_use_onednn = true;
        }

        if (auto multi_subgraph_op = ov::as_type_ptr<op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                for (auto& sub_op : sub_graph->get_ops()) {
                    process_op(sub_op);
                }
            }
        }
    };

    for (const auto& op : ops) {
        process_op(op);
    }

    m_optimize_data = true;
}

void ExecutionConfig::finalize_impl(const IRemoteContext* context) {
    GPU_DEBUG_IF(get_help()) {
        print_help();
        exit(-1);
    }

    const auto& info = dynamic_cast<const RemoteContextImpl*>(context)->get_engine().get_device_info();
    apply_hints(info);
    if (!is_set_by_user(ov::internal::enable_lp_transformations)) {
        m_enable_lp_transformations = info.supports_imad || info.supports_immad;
    }
    if (!is_set_by_user(ov::intel_gpu::use_onednn) && info.supports_immad) {
        m_use_onednn = true;
    }
    if (get_use_onednn()) {
        m_queue_type = QueueTypes::in_order;
    }

    if (!is_set_by_user(ov::hint::kv_cache_precision) || get_kv_cache_precision() == ov::element::dynamic) {
        if (info.supports_immad) {  // MFDNN-11755
            m_kv_cache_precision = get_inference_precision();
        } else {
            // Enable KV-cache compression by default for non-systolic platforms only
            m_kv_cache_precision = ov::element::i8;
        }
    }

    // Enable dynamic quantization by default for non-systolic platforms
    if (!is_set_by_user(ov::hint::dynamic_quantization_group_size) && get_dynamic_quantization_group_size() == 0 && !info.supports_immad) {
        m_dynamic_quantization_group_size = 32;
    }

    if (!get_force_implementations().empty()) {
        m_optimize_data = true;
    }

#ifdef ENABLE_DEBUG_CAPS
    // For now we apply env/config only for build with debug caps, but it can be updated in the future to allow
    // reading release options for any build type
    apply_config_options(context->get_device_name(), get_debug_config());
#endif // ENABLE_DEBUG_CAPS
}

void ExecutionConfig::apply_hints(const cldnn::device_info& info) {
    apply_execution_hints(info);
    apply_performance_hints(info);
    apply_priority_hints(info);
}

void ExecutionConfig::apply_execution_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::execution_mode)) {
        const auto mode = get_execution_mode();
        if (!is_set_by_user(ov::hint::inference_precision)) {
            if (mode == ov::hint::ExecutionMode::ACCURACY) {
                m_inference_precision = ov::element::dynamic;
            } else if (mode == ov::hint::ExecutionMode::PERFORMANCE) {
                if (info.supports_fp16)
                    m_inference_precision = ov::element::f16;
                else
                    m_inference_precision = ov::element::f32;
            }
        }
    }
}

void ExecutionConfig::apply_performance_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::performance_mode)) {
        const auto mode = get_performance_mode();
        if (!is_set_by_user(ov::num_streams)) {
            if (mode == ov::hint::PerformanceMode::LATENCY) {
                m_num_streams = 1;
            } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
                m_num_streams = ov::streams::AUTO;
            }
        }
    }

    if (get_num_streams() == ov::streams::AUTO) {
        int32_t n_streams = std::max<int32_t>(info.num_ccs, 2);
        m_num_streams = n_streams;
    }

    if (get_exclusive_async_requests()) {
        m_num_streams = 1;
    }

    // Allow kernels reuse only for single-stream scenarios
    if (get_enable_kernels_reuse()) {
        if (get_num_streams() != 1) {
            m_enable_kernels_reuse = false;
        }
    }
}

void ExecutionConfig::apply_priority_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::model_priority)) {
        const auto priority = get_model_priority();
        if (!is_set_by_user(ov::intel_gpu::hint::queue_priority)) {
            m_queue_priority = priority;
        }
    }
}

}  // namespace ov::intel_gpu
