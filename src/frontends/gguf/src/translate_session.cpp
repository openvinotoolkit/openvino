// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/constant_folding.hpp>

#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "pass/lower_set_rows_stateless.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {

using namespace ov::op;

namespace {

void add_sliced_mask(TensorMap& tensor_map) {
    // Slice the full attention mask down to the current token window for the attention ops.
    // For static (fixed-shape) models the slice bounds are constants, so this folds to a
    // fixed-size slice -- there is no separate static pass-through branch.
    auto create_sliced_mask = [&](const std::string& mask_name, const std::string& sliced_name) {
        if ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
            std::shared_ptr<ov::Node> mask_sliced =
                std::make_shared<ov::op::v8::Slice>(mask, zero, token_len_per_seq, one, two);
            mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
            mask_sliced->set_friendly_name(sliced_name);
            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced");
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

void add_rope_sin_cos(TensorMap& tensor_map, const RopeConfig& rope_config) {
    // n_dims == 0 means the model uses no RoPE; per_op means each ROPE op builds its own sin/cos
    // (e.g. gemma4 where SWA and global layers differ), so skip the shared table entirely.
    if (tensor_map.find("inp_pos") == tensor_map.end() || rope_config.n_dims == 0 || rope_config.per_op) {
        return;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_config, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap& tensor_map, const RopeConfig& rope_config) {
    add_sliced_mask(tensor_map);
    add_rope_sin_cos(tensor_map, rope_config);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr& input_model,
                                   const std::unordered_map<std::string, CreatorFunction>& translator_map,
                                   const std::vector<DecoderTransformationExtension::Ptr>& transformation_extensions)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_ov_model(nullptr),
      m_transformation_extensions(transformation_extensions) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr& input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto& gguf_model = std::dynamic_pointer_cast<InputModel>(input_model);

    for (const auto& it : gguf_model->get_model_inputs()) {
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            params.push_back(param);
        }
        (*tensor_map)[it.first] = it.second;
    }

    // Weights are not seeded here: a weight is visited as a regular "GGML_OP_NONE" node (a ggml
    // leaf carrying a "data" attribute) in visit_subgraph, and translate_weight writes its
    // dequantized node into the tensor map under the weight name, before the consuming op is
    // visited (the cgraph is topologically ordered).

    auto node_visitor = [&](std::shared_ptr<GgufDecoder> decoder) {
        auto operation_type = decoder->get_op_type();
        if (operation_type == "GGML_OP_NONE") {
            // A GGML_OP_NONE leaf is a weight only if the decoder exposes its raw bytes via the
            // "data" attribute; otherwise it is a model-input leaf (already seeded as a Parameter
            // above) and there is nothing to translate.
            if (!decoder->get_attribute("data").is<ov::Tensor>()) {
                return;
            }
        }

        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(),
                                      "Translation for operation type ",
                                      operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map);
        converted_outputs = it->second(node_context);

        const auto& node_output_names = decoder->get_output_names();
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(),
                                      "Number of ",
                                      operation_type,
                                      " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(),
                                      " and ",
                                      converted_outputs.size(),
                                      " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    // Build the shared LLM scaffolding (sliced attention mask + rope sin/cos table) once, so all
    // layers reuse it. Both builders are self-gating: add_sliced_mask only fires when the mask +
    // token_len_per_seq inputs are present, and add_rope_sin_cos only when inp_pos is present and
    // the model uses a shared rope table (n_dims != 0, not per-op). For a bare op / small cgraph
    // (no rope_config -> default n_dims == 0, no mask/pos inputs) both no-op, and the ROPE/attention
    // translators fall back to building their own -- so there is no separate "naive" mode.
    preprocess(*tensor_map, gguf_model->get_rope_config());
    gguf_model->visit_subgraph(node_visitor);

    for (const auto& name : gguf_model->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ",
                                name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    ov::ParameterVector used_params;
    for (const auto& param : params) {
        if (!param->output(0).get_target_inputs().empty()) {
            used_params.push_back(param);
        }
    }
    resulting_model = std::make_shared<Model>(results, used_params);

    apply_transformations(resulting_model);

    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
    // occurs", doubling memory usage per compile_model call.
    //
    // The bin_offset field serves as a unique key (not a real file offset) — this is
    // the same convention the GPU plugin uses for non-IR models (see
    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
    // import will map multiple constants to the same data.
    //
    // Small constants (< 16 elements) are excluded since they may be introduced by
    // optimization patterns and the overhead is negligible.
    size_t offset = 0;
    for (auto& node : resulting_model->get_ordered_ops()) {
        if (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
            cnst && cnst->get_byte_size() / cnst->get_element_type().size() >= 16) {
            auto& rt_info = cnst->get_rt_info();
            if (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    return resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();

    // Caller-registered transformation extensions run first. A SetRows-lowering extension (e.g.
    // the backend's stateful lowering) consumes the KV-cache SetRows ops here; the built-in stateless
    // lowering below then only fires on the ops left untouched. With no extension registered, the
    // stateless lowering handles every SetRows op -- so a plain convert() yields the
    // llama.cpp-faithful stateless model.
    for (const auto& ext : m_transformation_extensions) {
        ext->register_pass(manager);
    }
    manager.register_pass<pass::LowerSetRowsStateless>();
    manager.register_pass<ov::pass::ConvertConvertLike>();

    manager.run_passes(model);
    return model;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
