// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/frontend.hpp"

#include "graph_iterator_meta.hpp"
#include "graph_iterator_proto.hpp"
#include "graph_iterator_proto_txt.hpp"
#include "graph_iterator_saved_model.hpp"
#include "helper_ops/internal_operation.hpp"
#include "helper_transforms/const_to_result_remover.hpp"
#include "helper_transforms/embedding_segments_feature_fusing.hpp"
#include "helper_transforms/saved_model_unused_remover.hpp"
#include "helper_transforms/tensor_array_v3_replacer.hpp"
#include "helper_transforms/tensor_list_ops_resolver.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "tf_framework_node.hpp"
#include "transformations/common_optimizations/eliminate_loop_inputs_outputs.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/switch_merge_resolve.hpp"
#include "transformations/transpose_sinking/ts_general.hpp"
#include "transformations/uninitialized_variable_resolve.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow;

namespace {
void update_failures_unsupported_ops(const std::string& op_type,
                                     const ov::op::util::FrameworkNodeAttrs& fw_node_attrs,
                                     std::set<std::string>& unsupported_operations,
                                     std::unordered_map<std::string, std::string>& failures) {
    // if this operation is encountered among unsupported operations
    // or conversion failures, skip it
    if (failures.count(op_type) > 0 || unsupported_operations.count(op_type) > 0) {
        return;
    }
    if (fw_node_attrs.find(FrameworkNode::failed_conversion_key) != fw_node_attrs.end()) {
        // save only the first encountered failure that is more improtant for developer
        // that means the translator is found but the conversion is failed
        failures[op_type] = fw_node_attrs.at(FrameworkNode::failed_conversion_key);
    } else {
        // found new unsupported operation
        unsupported_operations.insert(op_type);
    }
}

void get_unsupported_operations_and_failures(const std::shared_ptr<Model>& model,
                                             std::set<std::string>& unsupported_operations,
                                             std::unordered_map<std::string, std::string>& failures) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto& internal_op = ov::as_type_ptr<InternalOperation>(node)) {
            // handle internal operations separately
            // which can have elaborated reason of unconverted operation
            // like Const of string type
            auto op_type = internal_op->get_no_conversion_reason();
            if (unsupported_operations.count(op_type) > 0) {
                continue;
            }
            unsupported_operations.insert(op_type);
        } else if (const auto& variable = ov::as_type_ptr<Variable>(node)) {
            auto op_type = variable->get_decoder()->get_op_type();
            auto op_name = variable->get_name();
            failures[op_type] = "Variable or resource `" + op_name + "` is not initialized, model is inconsistent";
        } else if (const auto& fw_node = ov::as_type_ptr<FrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            auto fw_node_attrs = fw_node->get_attrs();
            update_failures_unsupported_ops(op_type, fw_node_attrs, unsupported_operations, failures);
        } else if (const auto& fw_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(node)) {
            // handle auxiliary operations from common frontend like ComplexTypeMark
            auto op_type = std::string(fw_node->get_type_name());
            auto fw_node_attrs = fw_node->get_attrs();
            update_failures_unsupported_ops(op_type, fw_node_attrs, unsupported_operations, failures);
        }
        if (const auto& fw_node = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            int subgraphs_size = static_cast<int>(fw_node->get_internal_subgraphs_size());
            for (int i = 0; i < subgraphs_size; ++i) {
                get_unsupported_operations_and_failures(fw_node->get_function(i), unsupported_operations, failures);
            }
        }
    }
}

void translate_framework_node(const std::shared_ptr<FrameworkNode>& node,
                              const TranslatorDictionaryType& op_translators) {
    auto type = node->get_op_type();

    const auto& TRANSLATE_OP_MAP = op_translators;
    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");

    ov::OutputVector ov_inputs = node->input_values();
    NodeContext node_ctx(node->get_decoder(), ov_inputs);
    auto new_node_outputs = translator_it->second(node_ctx);

    auto new_output = new_node_outputs.begin();
    auto old_outputs = node->outputs();
    auto old_output = old_outputs.begin();

    for (; new_output != new_node_outputs.end() && old_output != old_outputs.end(); ++old_output, ++new_output) {
        old_output->replace(new_output->port);
    }
}
}  // namespace

FrontEnd::FrontEnd() : m_op_translators(tensorflow::op::get_supported_ops()) {}

/// \brief Check if FrontEndTensorflow can recognize model from given parts
bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;

    // For TF1 models it can be a case of two input variants: input model and v1 checkpoints
    if (variants.size() != 1 + extra_variants_num)
        return false;

    // to figure out if the model with v1 checkpoints is supported,
    // it is sufficient to check only the input model format
    // avoid parsing of checkpoints here
    if (variants[0].is<std::string>()) {
        std::string model_path = variants[0].as<std::string>();
        if (GraphIteratorProto::is_supported(model_path)) {
            // handle binary protobuf format
            // for automatic deduction of the frontend to convert the model
            // we have more strict rule that is to have `.pb` extension in the path
            return true;
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            return true;
        } else if (GraphIteratorMeta::is_supported(model_path)) {
            return true;
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // handle text protobuf format
            return true;
        }
    } else if (variants[0].is<std::vector<std::string>>() && variants[0].as<std::vector<std::string>>().size() == 2) {
        // here, we assume to get the input model path and checkpoints directory
        auto paths = variants[0].as<std::vector<std::string>>();
        auto model_path = paths[0];
        auto checkpoints_dir = paths[1];
        if (GraphIteratorProto::is_supported(model_path)) {
            // binary protobuf format with checkpoints
            return true;
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // text protobuf format with checkpoints
            return true;
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            // saved model format with tagged metagraphs
            return true;
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring model_path = variants[0].as<std::wstring>();
        if (GraphIteratorProto::is_supported(model_path)) {
            // handle binary protobuf format with a path in Unicode
            // for automatic deduction of the frontend to convert the model
            // we have more strict rule that is to have `.pb` extension in the path
            return true;
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            return true;
        } else if (GraphIteratorMeta::is_supported(model_path)) {
            return true;
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // handle text protobuf format
            return true;
        }
    } else if (variants[0].is<std::vector<std::wstring>>() && variants[0].as<std::vector<std::wstring>>().size() == 2) {
        // here, we assume to get the input model path and checkpoints directory
        auto paths = variants[0].as<std::vector<std::wstring>>();
        auto model_path = ov::util::wstring_to_string(paths[0]);
        auto checkpoints_dir = ov::util::wstring_to_string(paths[1]);
        if (GraphIteratorProto::is_supported(model_path)) {
            // binary protobuf format with checkpoints
            return true;
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // text protobuf format with checkpoints
            return true;
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            // saved model format with tagged metagraphs
            return true;
        }
    }
#endif
    else if (variants[0].is<GraphIterator::Ptr>()) {
        // this is used for OpenVINO with TensorFlow Integration
        return true;
    }
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    // Enable mmap by default
    bool mmap_enabled = variants[variants.size() - 1].is<bool>() ? variants[variants.size() - 1].as<bool>() : true;

    // For TF1 models it can be a case of two input variants: input model and v1 checkpoints
    FRONT_END_GENERAL_CHECK(
        variants.size() == 1 + extra_variants_num,
        "[TensorFlow Frontend] Internal error or inconsistent input model: the frontend supports "
        "frozen formats (.pb and .pbtxt), SavedModel and MetaGraph (.meta) formats, and v1 checkpoints.");

    if (variants[0].is<std::string>()) {
        auto model_path = variants[0].as<std::string>();
        if (GraphIteratorProto::is_supported(model_path)) {
            // handle binary protobuf format
            return std::make_shared<InputModel>(std::make_shared<GraphIteratorProto>(model_path), m_telemetry);
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            std::shared_ptr<GraphIteratorSavedModel> graph_iterator;
            graph_iterator = std::make_shared<GraphIteratorSavedModel>(model_path, std::string("serve"), mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_saved_model_input_names(),
                                                graph_iterator->get_saved_model_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        } else if (GraphIteratorMeta::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorMeta>(model_path, mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_metagraph_input_names(),
                                                graph_iterator->get_metagraph_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // handle text protobuf format
            return std::make_shared<InputModel>(std::make_shared<GraphIteratorProtoTxt>(model_path), m_telemetry);
        }
    } else if (variants[0].is<std::vector<std::string>>()) {
        // here, we assume to get the input model path and checkpoints directory
        auto paths = variants[0].as<std::vector<std::string>>();
        FRONT_END_GENERAL_CHECK(
            paths.size() == 2,
            "[TensorFlow Frontend] Internal error or inconsistent input model: the frontend supports "
            "frozen formats (.pb and .pbtxt), SavedModel and MetaGraph (.meta) formats, and v1 checkpoints.");
        auto model_path = paths[0];
        auto checkpoints_dir = paths[1];
        if (GraphIteratorProto::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorProto>(model_path, checkpoints_dir);
            // handle binary protobuf format with checkpoints
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                HashTableKeysValuesMap{},
                                                HashTableKeysValuesMap{},
                                                graph_iterator->get_checkpoint_v1_reader(),
                                                false);
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorProtoTxt>(model_path, checkpoints_dir);
            // handle text protobuf format with checkpoints
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                HashTableKeysValuesMap{},
                                                HashTableKeysValuesMap{},
                                                graph_iterator->get_checkpoint_v1_reader(),
                                                false);
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            auto saved_model_tags = paths[1];
            std::shared_ptr<GraphIteratorSavedModel> graph_iterator;
            graph_iterator = std::make_shared<GraphIteratorSavedModel>(model_path, saved_model_tags, mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_saved_model_input_names(),
                                                graph_iterator->get_saved_model_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring model_path = variants[0].as<std::wstring>();
        if (GraphIteratorProto::is_supported(model_path)) {
            // handle binary protobuf format with a path in Unicode
            return std::make_shared<InputModel>(std::make_shared<GraphIteratorProto>(model_path), m_telemetry);
        } else if (GraphIteratorSavedModel::is_supported(model_path)) {
            std::shared_ptr<GraphIteratorSavedModel> graph_iterator;
            graph_iterator = std::make_shared<GraphIteratorSavedModel>(model_path,
                                                                       std::string(META_GRAPH_DEFAULT_TAG),
                                                                       mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_saved_model_input_names(),
                                                graph_iterator->get_saved_model_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        } else if (GraphIteratorMeta::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorMeta>(model_path, mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_metagraph_input_names(),
                                                graph_iterator->get_metagraph_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            // handle text protobuf format with a path in Unicode
            return std::make_shared<InputModel>(std::make_shared<GraphIteratorProtoTxt>(model_path), m_telemetry);
        }
    } else if (variants[0].is<std::vector<std::wstring>>()) {
        // here, we assume to get the input model path and checkpoints directory
        auto paths = variants[0].as<std::vector<std::wstring>>();
        FRONT_END_GENERAL_CHECK(
            paths.size() == 2,
            "[TensorFlow Frontend] Internal error or inconsistent input model: the frontend supports "
            "frozen formats (.pb and .pbtxt), SavedModel and MetaGraph (.meta) formats, and v1 checkpoints.");
        auto model_path = ov::util::wstring_to_string(paths[0]);
        auto checkpoints_dir = ov::util::wstring_to_string(paths[1]);
        if (GraphIteratorProto::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorProto>(model_path, checkpoints_dir);
            // handle binary protobuf format with checkpoints
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                HashTableKeysValuesMap{},
                                                HashTableKeysValuesMap{},
                                                graph_iterator->get_checkpoint_v1_reader(),
                                                false);
        } else if (GraphIteratorProtoTxt::is_supported(model_path)) {
            auto graph_iterator = std::make_shared<GraphIteratorProtoTxt>(model_path, checkpoints_dir);
            // handle text protobuf format with checkpoints
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                HashTableKeysValuesMap{},
                                                HashTableKeysValuesMap{},
                                                graph_iterator->get_checkpoint_v1_reader(),
                                                false);
        }
        auto saved_model_tags = ov::util::wstring_to_string(paths[1]);
        if (GraphIteratorSavedModel::is_supported(model_path)) {
            std::shared_ptr<GraphIteratorSavedModel> graph_iterator;
            graph_iterator = std::make_shared<GraphIteratorSavedModel>(model_path, saved_model_tags, mmap_enabled);
            return std::make_shared<InputModel>(graph_iterator,
                                                m_telemetry,
                                                graph_iterator->get_variables_index(),
                                                graph_iterator->get_saved_model_input_names(),
                                                graph_iterator->get_saved_model_output_names(),
                                                graph_iterator->get_hash_table_keys_map(),
                                                graph_iterator->get_hash_table_values_map(),
                                                nullptr,
                                                true);
        }
    }
#endif
    else if (variants[0].is<GraphIterator::Ptr>()) {
        // this is used for OpenVINO with TensorFlow Integration
        auto graph_iterator = variants[0].as<GraphIterator::Ptr>();
        std::shared_ptr<std::map<std::string, std::string>> input_names_map = nullptr;
        std::shared_ptr<std::map<std::string, std::string>> output_names_map = nullptr;
        if (graph_iterator->get_input_names_map().size() > 0) {
            input_names_map =
                std::make_shared<std::map<std::string, std::string>>(graph_iterator->get_input_names_map());
        }
        if (graph_iterator->get_output_names_map().size() > 0) {
            output_names_map =
                std::make_shared<std::map<std::string, std::string>>(graph_iterator->get_output_names_map());
        }
        return std::make_shared<InputModel>(graph_iterator,
                                            m_telemetry,
                                            nullptr,
                                            input_names_map,
                                            output_names_map,
                                            HashTableKeysValuesMap{},
                                            HashTableKeysValuesMap{},
                                            nullptr,
                                            false);
    }

    FRONT_END_GENERAL_CHECK(false,
                            "[TensorFlow Frontend] Internal error or inconsistent input model: the frontend supports "
                            "frozen formats (.pb and .pbtxt), SavedModel and MetaGraph (.meta), and v1 checkpoints.");

    return nullptr;
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    auto f = convert_partially(model);

    std::unordered_map<std::string, std::string> failures;
    std::set<std::string> unsupported_operations;
    get_unsupported_operations_and_failures(f, unsupported_operations, failures);

    std::stringstream exception_message;
    for (const auto& failure : failures) {
        auto exception_str = "[TensorFlow Frontend] Internal error, conversion is failed for " + failure.first +
                             " operation with a message:\n" + failure.second + "\n";
        exception_message << exception_str;
        if (m_telemetry) {
            m_telemetry->send_event("error_info",
                                    ov::util::filter_lines_by_prefix(exception_str, "[TensorFlow Frontend] "));
        }
    }

    if (m_telemetry) {
        for (const auto& unsupported_operation : unsupported_operations) {
            m_telemetry->send_event("error_cause", "tf_" + unsupported_operation);
        }
    }
    if (unsupported_operations.size() > 0) {
        exception_message << "[TensorFlow Frontend] Internal error, no translator found for operation(s): ";
        size_t counter = 0;
        size_t tokenizer_counter = 0;
        std::string unsupported_ops_from_tokenizers;
        const auto& all_tokenizer_ops = ov::frontend::tensorflow::op::get_supported_ops_via_tokenizers();
        for (const auto& unsupported_operation : unsupported_operations) {
            if (counter > 0) {
                exception_message << ", ";
            }
            exception_message << unsupported_operation;
            ++counter;

            // collect a list of unconverted operations for which openvino-tokenizers provides conversion extensions
            if (std::find(all_tokenizer_ops.begin(), all_tokenizer_ops.end(), unsupported_operation) !=
                all_tokenizer_ops.end()) {
                if (tokenizer_counter > 0) {
                    unsupported_ops_from_tokenizers += ", ";
                }
                unsupported_ops_from_tokenizers += unsupported_operation;
                ++tokenizer_counter;
            }
        }
        exception_message
            << "\nTo facilitate the conversion of unsupported operations, refer to Frontend Extension "
               "documentation: "
               "https://docs.openvino.ai/latest/openvino_docs_Extensibility_UG_Frontend_Extensions.html \n";

        // recommend to use openvino-tokenizers if some unconverted operations from tokenizers are met
        if (unsupported_ops_from_tokenizers.size() > 0) {
            exception_message << "\nEncountered unconverted operation(s) for which openvino-tokenizers package "
                                 "provides conversion extension(s): "
                              << unsupported_ops_from_tokenizers
                              << ". Install OpenVINO Tokenizers, refer to the documentation: "
                                 "https://docs.openvino.ai/2025/openvino-workflow-generative/ov-tokenizers.html \n";
        }
    }

    bool is_conversion_successful = ((unsupported_operations.size() == 0) && (failures.size() == 0));
    FRONT_END_OP_CONVERSION_CHECK(is_conversion_successful, exception_message.str());

    return f;
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager("Frontend:TF:convert_partially");
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    // create a shared pointer to the cloned dictionary of translators
    auto translator_map = std::make_shared<TranslatorDictionaryType>(m_op_translators);

    std::shared_ptr<ov::Model> f;
    TranslateSession translate_session(model, translator_map, "TensorFlow_Frontend_IR");
    try {
        f = translate_session.get_converted_model();
    } catch (const std::exception& e) {
        if (m_telemetry) {
            auto filtered_message = ov::util::filter_lines_by_prefix(e.what(), "[TensorFlow Frontend] ");
            if (filtered_message.size() > 0) {
                m_telemetry->send_event("error_info", filtered_message);
            }
        }
        throw;
    }
    normalize(f);

    return f;
}

std::shared_ptr<ov::Model> FrontEnd::decode(const ov::frontend::InputModel::Ptr& model) const {
    auto translator_map = std::make_shared<TranslatorDictionaryType>();

    const std::set<std::string> required_types{"Placeholder", "NoOp"};
    for (const auto& name : required_types) {
        translator_map->emplace(name, m_op_translators.at(name));
    }

    std::shared_ptr<ov::Model> f;
    TranslateSession translate_session(model, translator_map, "TensorFlow_Frontend_IR");
    try {
        f = translate_session.get_converted_model();
    } catch (const std::exception& e) {
        if (m_telemetry) {
            auto filtered_message = ov::util::filter_lines_by_prefix(e.what(), "[TensorFlow Frontend] ");
            if (filtered_message.size() > 0) {
                m_telemetry->send_event("error_info", filtered_message);
            }
        }
        throw;
    }

    return f;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<FrameworkNode>(node)) {
            translate_framework_node(ov::as_type_ptr<FrameworkNode>(node), m_op_translators);
        }
    }
    for (const auto& result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }

    normalize(partiallyConverted);
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    ov::pass::Manager manager("Frontend:TF:normalize");

    // Mark quantized and f16/bf16 compressed constants to prevent CF for them,
    // so that not extra memory is used for intermediate decompressed constants.
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
    manager.register_pass<pass::SavedModelUnusedRemover>();
    manager.register_pass<pass::UninitializedVariableResolver>();
    manager.register_pass<pass::EmbeddingSegmentSingleFeatureFusion>();
    manager.register_pass<pass::TensorArrayV3Replacer>();
    manager.register_pass<pass::ConstToResultRemover>();
    manager.register_pass<pass::SwitchMergeResolver>();

    // apply EliminateLoopInputsOutputs to avoid extra Results
    // that output the same value as receiving on input
    // it is needed for applying TensorListInLoopOptimization
    manager.register_pass<ov::pass::EliminateLoopInputsOutputs>();
    manager.register_pass<pass::TensorListReplacer>();
    manager.register_pass<pass::TensorListInLoopOptimization>();
    manager.register_pass<pass::TensorListSetItemReplacer>();
    manager.register_pass<pass::TensorListPushBackReplacer>();
    manager.register_pass<pass::TensorListGetItemReplacer>();

    manager.register_pass<ov::pass::UnrollIf>();
    manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
    manager.register_pass<ov::pass::ReverseShapeAndTypeInfer>();
    manager.register_pass<ov::pass::ResolveNameCollisions>(true);
    manager.run_passes(model);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_transformation_extensions.push_back(transformation);
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (auto common_conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(common_conv_ext);
        if (common_conv_ext->get_converter()) {
            m_op_translators[common_conv_ext->get_op_type()] =
                ov::frontend::tensorflow::CreatorFunctionIndexed([=](const tensorflow::NodeContext& context) {
                    return common_conv_ext->get_converter()(context);
                });
        } else if (common_conv_ext->get_converter_named_and_indexed()) {
            m_op_translators[common_conv_ext->get_op_type()] =
                ov::frontend::tensorflow::CreatorFunctionNamedAndIndexed([=](const tensorflow::NodeContext& context) {
                    return common_conv_ext->get_converter_named_and_indexed()(context);
                });
        }
        // Ignore other types of extensions in particular CreatorFunctionNamed which cannot be used with tensorflow
        // frontend
    } else if (const auto& tensorflow_conv_ext =
                   ov::as_type_ptr<ov::frontend::tensorflow::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(tensorflow_conv_ext);
        m_op_translators[tensorflow_conv_ext->get_op_type()] = tensorflow_conv_ext->get_converter();
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}
