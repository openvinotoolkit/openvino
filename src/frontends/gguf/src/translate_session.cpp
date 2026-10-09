// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>

#include "builder/gguf_builder_decoder.hpp"
#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/frontend/gguf/tokenizer_metadata.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "pass/lower_set_rows_stateless.hpp"
#include "pass/prune_orphaned_parameters.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {

using namespace ov::op;

void translate_node(const std::shared_ptr<GgufDecoder>& decoder,
                    std::shared_ptr<TensorMap>& tensor_map,
                    const std::unordered_map<std::string, CreatorFunction>& translators,
                    std::map<std::string, std::string>& storage_aliases) {
    const auto& operation_type = decoder->get_op_type();
    if (operation_type == "GGML_OP_NONE") {
        // A GGML_OP_NONE leaf is a weight if the decoder marks it as one: either the native
        // builder's pre-extracted payload (bool "gguf_weight") or the cgraph decoder's raw
        // bytes ("data"). Otherwise it is a model-input leaf (already seeded as a Parameter
        // above) and there is nothing to translate.
        const bool is_builder_weight =
            decoder->get_attribute("gguf_weight").is<bool>() && decoder->get_attribute("gguf_weight").as<bool>();
        const bool is_cgraph_weight = decoder->get_attribute("data").is<ov::Tensor>();
        if (!is_builder_weight && !is_cgraph_weight) {
            return;
        }
    }

    ov::OutputVector converted_outputs;
    auto it = translators.find(operation_type);
    FRONT_END_OP_CONVERSION_CHECK(it != translators.end(),
                                  "Translation for operation type ",
                                  operation_type,
                                  " is not implemented.");
    NodeContext node_context(decoder, tensor_map);
    try {
        converted_outputs = it->second(node_context);
    } catch (const std::exception& e) {
        // Name the failing node and its operands: translator checks only describe the local condition.
        std::string inputs;
        const auto input_names = decoder->get_input_names();
        for (size_t i = 0; i < input_names.size(); ++i) {
            inputs += i ? ", " : "";
            const auto& name = input_names[i];
            const auto it = tensor_map->find(name);
            inputs += it == tensor_map->end() ? "'" + name + "' (not translated)" : it->second.get_partial_shape().to_string();
        }
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      operation_type,
                                      " '",
                                      decoder->get_op_name(),
                                      "' (inputs: ",
                                      inputs,
                                      "; ggml output shape: ",
                                      decoder->get_output_shape().to_string(),
                                      "): ",
                                      e.what());
    }

    const auto& node_output_names = decoder->get_output_names();
    FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(),
                                  "Number of ",
                                  operation_type,
                                  " outputs greater than number of converted outputs, which are ",
                                  node_output_names.size(),
                                  " and ",
                                  converted_outputs.size(),
                                  " respectively.");

    const auto storage_name = [&](const std::string& name) {
        const auto it = storage_aliases.find(name);
        return it == storage_aliases.end() ? name : it->second;
    };
    // Operations that return their whole destination with a region rewritten update that storage in
    // place: every later read of the storage, and the model output written back to it, must see the
    // new value. SET_ROWS writes rows of input 2, a CPY update ("axis") writes a range of input 1, and a
    // ranged SCALE (start/length inputs) rescales a range of input 0.
    int updated_input = -1;
    if (operation_type == "GGML_OP_SET_ROWS") {
        updated_input = 2;
    } else if (operation_type == "GGML_OP_CPY" && !decoder->get_attribute("axis").empty()) {
        updated_input = 1;
    } else if (operation_type == "GGML_OP_SCALE" && decoder->get_input_size() == 3) {
        updated_input = 0;
    }
    std::string output_storage;
    if (updated_input >= 0) {
        const auto destination = node_context.get_input(updated_input);
        output_storage = storage_name(decoder->get_input_names().at(updated_input));
        const auto previous_root = tensor_map->at(output_storage);
        const ov::Output<Node> updated_root =
            previous_root == destination
                ? converted_outputs.at(0)
                : std::make_shared<v1::Reshape>(converted_outputs.at(0),
                                               std::make_shared<v3::ShapeOf>(previous_root, ov::element::i64),
                                               false)->output(0);
        std::unordered_map<Node*, std::shared_ptr<Node>> rebound_nodes;
        const auto rebind = [&](auto&& self, const ov::Output<Node>& value) -> ov::Output<Node> {
            if (value == previous_root) {
                return updated_root;
            }
            const auto node = value.get_node_shared_ptr();
            const auto found = rebound_nodes.find(node.get());
            if (found != rebound_nodes.end()) {
                return found->second->output(value.get_index());
            }
            auto inputs = node->input_values();
            bool changed = false;
            for (auto& input : inputs) {
                auto replacement = self(self, input);
                changed = changed || replacement != input;
                input = replacement;
            }
            auto rebound = changed ? node->clone_with_new_inputs(inputs) : node;
            if (changed) {
                ov::copy_runtime_info(node, rebound);
            }
            rebound_nodes.emplace(node.get(), rebound);
            return rebound->output(value.get_index());
        };
        TensorMap updates;
        // Source RESHAPE aliases share storage; CONT copies may share an OV node but must not be rebound.
        // Replay their shape expressions on the new value, including ShapeOf, so growing state stays dynamic.
        for (const auto& entry : *tensor_map) {
            if (storage_name(entry.first) == output_storage) {
                updates[entry.first] = entry.second == destination
                                           ? converted_outputs.at(0)
                                           : rebind(rebind, entry.second);
            }
        }
        for (const auto& update : updates) {
            (*tensor_map)[update.first] = update.second;
        }
    } else if (operation_type == "GGML_OP_RESHAPE") {
        output_storage = storage_name(decoder->get_input_names().at(0));
    }
    for (size_t i = 0; i < node_output_names.size(); ++i) {
        const auto& output_name = node_output_names[i];
        if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
            (*tensor_map)[output_name] = converted_outputs[i];
        }
        storage_aliases.erase(output_name);
        if (!output_storage.empty() && output_storage != output_name) {
            storage_aliases[output_name] = output_storage;
        }
    }
}

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
    const auto& gguf_model_decoder = gguf_model->get_model_decoder();

    // Auxiliary input Parameters may only get a consumer from a later normalization pass, after
    // the unused-Parameter pruning below. Track them so pruning never drops one for lack of a
    // consumer at translate time; a pass that ends up not consuming one leaves it dangling, which
    // later constant folding removes.
    std::set<ov::Node*> deferred_use_params;
    // get_model_inputs() already folds in get_model_extra_inputs()'s entries (per the decoder.hpp
    // contract), so the same Parameter can show up in both loops below; skip re-adding it or it
    // ends up registered on the model twice, under two different indices.
    std::set<ov::Node*> seen_params;

    for (const auto& it : gguf_model_decoder->get_model_inputs()) {
        // Not every decoder splits auxiliary inputs into get_model_extra_inputs(): one that still
        // folds them into get_model_inputs() (see the decoder.hpp contract) hands us a mix of
        // Parameters and other node types here, so guard the cast instead of pushing a null
        // Parameter into `params`.
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            params.push_back(param);
            seen_params.insert(param.get());
        }
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : gguf_model_decoder->get_model_extra_inputs()) {
        if (auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            if (seen_params.insert(p.get()).second) {
                params.push_back(p);
            }
            deferred_use_params.insert(p.get());
        }
        (*tensor_map)[it.first] = it.second;
    }

    // Weights are not seeded here: every decoder surfaces them as "GGML_OP_NONE" leaves that
    // translate_weight turns into a compressed subgraph during the walk below, which keeps them
    // lazy (never materialized to f32) and keeps one weight-loading path for both ingest paths.

    std::map<std::string, std::string> storage_aliases;
    auto node_visitor = [&](std::shared_ptr<GgufDecoder> decoder) {
        translate_node(decoder, tensor_map, m_translator_map, storage_aliases);
    };

    // Native builders eagerly translate their nodes; other decoders are walked here.
    const auto builder = std::dynamic_pointer_cast<GgufBuilderDecoder>(gguf_model_decoder);
    if (builder) {
        tensor_map = builder->values();
    } else {
        gguf_model_decoder->visit_subgraph(node_visitor);
    }

    for (const auto& name : gguf_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ",
                                name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    ov::ParameterVector used_params;
    for (const auto& param : params) {
        // Keep a Parameter if it currently feeds something, OR if its consumer is created by a
        // later normalization pass.
        if (!param->output(0).get_target_inputs().empty() || deferred_use_params.count(param.get())) {
            used_params.push_back(param);
        }
    }
    resulting_model = std::make_shared<Model>(results, used_params);

    // M-RoPE models need 4 position sections per token, which changes the position_ids contract a
    // GenAI-facing consumer has to satisfy; record it so AdaptToGenAI can adapt (it runs on the
    // model alone and cannot ask the decoder).
    {
        const auto rc_any = gguf_model_decoder->get_attribute("rope_config");
        if (!rc_any.empty() && rc_any.as<RopeConfig>().is_imrope) {
            resulting_model->get_rt_info()[pass::gguf_imrope_key()] = true;
        }
    }

    // Sliding-window length, when the metadata carries one; AdaptToGenAI needs it to build a
    // correctly windowed self_kq_mask_swa instead of reusing the full causal mask (it runs on the
    // model alone and cannot ask the decoder).
    {
        const auto swa_any = gguf_model_decoder->get_attribute("swa_window_size");
        if (!swa_any.empty() && swa_any.as<int>() > 0) {
            resulting_model->get_rt_info()[pass::gguf_swa_window_key()] = static_cast<int64_t>(swa_any.as<int>());
        }
    }

    // Record the recurrent (overwritten, non-appending) states BEFORE transformations: a caller
    // that registered MakeStateful runs it inside apply_transformations, and unlike a KV cache
    // these carry nothing in the graph that identifies them (see gguf_recurrent_states_key).
    // Flattened to alternating {input, output} because rt_info takes an ov::Any.
    {
        const auto& rs = gguf_model_decoder->get_recurrent_states();
        if (!rs.empty()) {
            std::vector<std::string> flat;
            flat.reserve(rs.size() * 2);
            for (const auto& kv : rs) {
                flat.push_back(kv.first);
                flat.push_back(kv.second);
            }
            resulting_model->get_rt_info()[pass::gguf_recurrent_states_key()] = flat;
        }
    }

    // The architecture and projector metadata select what the GenAI adapters change, including
    // adapters registered as transformation extensions.
    if (builder)
        resulting_model->get_rt_info()["gguf_architecture"] = builder->get_attribute("architecture");
    if (!gguf_model_decoder->get_mmproj_config().empty()) {
        resulting_model->get_rt_info()["gguf_mmproj"] = gguf_model_decoder->get_mmproj_config();
    }

    if (builder) {
        resulting_model = resulting_model->clone();
    }
    resulting_model = apply_transformations(resulting_model);

    // Auxiliary Parameters are kept through normalization in case a transformation extension
    // creates their only consumer; drop any still disconnected afterwards (a plugin requires
    // every declared model input to reach a node in the execution graph).
    std::set<ov::Node*> ordered_nodes;
    for (const auto& node : resulting_model->get_ordered_ops()) {
        ordered_nodes.insert(node.get());
    }
    const auto transformed_params = resulting_model->get_parameters();
    for (const auto& param : transformed_params) {
        if (ordered_nodes.count(param.get()) == 0) {
            resulting_model->remove_parameter(param);
        }
    }

    // Attach GGUF tokenizer metadata (the file's tokenizer.* keys) to the model's rt_info as a
    // non-serializable attribute, so a downstream consumer (OpenVINO GenAI) can build the
    // tokenizer without reopening the .gguf. Empty when the decoder carries no tokenizer config.
    const auto& tok_cfg = gguf_model_decoder->get_tokenizer_config();
    if (!tok_cfg.empty()) {
        resulting_model->get_rt_info()[gguf_tokenizer_metadata_key()] =
            std::make_shared<GGUFTokenizerMetadata>(tok_cfg);
    }

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
            cnst && cnst->get_element_type().size() > 0 &&
            cnst->get_byte_size() / cnst->get_element_type().size() >= 16) {
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

    // Caller-registered transformation extensions run first, which is what makes execution mode a
    // caller concern rather than a frontend one: an extension that lowers SetRows itself -- e.g.
    // ov::frontend::gguf::pass::GGUFMakeStateful, or a backend's own variant -- consumes the KV-cache
    // SetRows ops here, and the built-in stateless lowering below then only fires on the ops left
    // untouched (MoE routing writes and the like). With no extension registered, the stateless
    // lowering handles every SetRows op, so a plain convert() yields the stateless model.
    for (const auto& ext : m_transformation_extensions) {
        ext->register_pass(manager);
    }
    manager.register_pass<pass::LowerSetRowsStateless>();

    manager.register_pass<ov::pass::ConvertConvertLike>();
    // The lowered ConvertLikes are frequently no-ops (k/v already share q's precision). Drop them:
    // a same-type Convert on an SDPA k/v input is invisible to the plugins but breaks
    // StateManagementPattern, which admits no Convert between the KV-cache Concat and SDPA, and so
    // silently disables the PagedAttention backend for every GGUF model.
    manager.register_pass<ov::pass::EliminateConvert>();
    // ggml VIEWs of disjoint ranges of one tensor (fused q/k/v, z/beta/alpha) become Slices; as one
    // VariadicSplit the plugins can run them in place instead of as one copy kernel each.
    manager.register_pass<ov::pass::GroupedSliceToVSplitOptimization>();

    // Fuse a standalone Loop-based Gated Delta Net recurrence here (not in a plugin, which can't
    // remove a Parameter without violating its I/O port-count invariant), only for models with a
    // Loop: GatedDeltaNetFusion also bundles TransposeFuse, a generic cleanup that would otherwise
    // touch every architecture's op count.
    const auto ordered_ops = model->get_ordered_ops();
    const auto has_loop =
        std::any_of(ordered_ops.begin(), ordered_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
            return ov::is_type<ov::op::v5::Loop>(node);
        });
    if (has_loop) {
        auto live_before_gdn = std::make_shared<std::unordered_set<const ov::Node*>>();
        manager.register_pass<pass::SnapshotLiveParameters>(live_before_gdn);
        manager.register_pass<ov::pass::GatedDeltaNetFusion>();
        manager.register_pass<pass::PruneParametersOrphanedSince>(live_before_gdn);
    }

    manager.run_passes(model);
    return model;
}

}  // namespace ov::frontend::gguf
