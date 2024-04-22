// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/frontend.hpp"

#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/pytorch/extension/conversion.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "transformations/op_conversions/convert_convertpromotetypes.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transforms/append_list_unpack_replacer.hpp"
#include "transforms/aten_cat_replacer.hpp"
#include "transforms/aten_getitem_replacer.hpp"
#include "transforms/aten_index_put_replacer.hpp"
#include "transforms/aten_index_replacer.hpp"
#include "transforms/aten_stack_list_construct_replacer.hpp"
#include "transforms/dict_resolver.hpp"
#include "transforms/einsum_list_construct.hpp"
#include "transforms/index_loop_getitem_replacer.hpp"
#include "transforms/irfftn_complex_replacer.hpp"
#include "transforms/listconstruct_replacer.hpp"
#include "transforms/min_max_prim_list_construct_replacer.hpp"
#include "transforms/prim_list_construct_pad.hpp"
#include "transforms/prim_list_tuple_construct_replacer.hpp"
#include "transforms/prim_list_unpack_replacer.hpp"
#include "transforms/prim_unpack_parameter_replacer.hpp"
#include "transforms/quantized_node_remover.hpp"
#include "transforms/remove_packing_ops.hpp"
#include "transforms/reverseprop_resolver.hpp"
#include "transforms/rfftn_complex_replacer.hpp"
#include "transforms/softmax_reshape_elimination.hpp"
#include "transforms/string_equality_replacer.hpp"
#include "transforms/tuple_unpack_replacer.hpp"
#include "transforms/u4_block_repack.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

namespace {
std::map<std::string, std::string> get_unconverted_types_from_model(const std::shared_ptr<Model>& model) {
    std::map<std::string, std::string> unconverted_ops_types;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<PtFrameworkNode>(node)) {
            const auto& attrs = fw_node->get_attrs();
            FRONT_END_GENERAL_CHECK(attrs.find(PtFrameworkNode::op_type_key) != attrs.end(),
                                    "FrameworkNode attributes do not contain operation type.");
            std::string exception_msg;
            if (attrs.find(PtFrameworkNode::failed_conversion_key) != attrs.end()) {
                exception_msg = attrs.at(PtFrameworkNode::failed_conversion_key);
            }
            if (!unconverted_ops_types.count(attrs.at(PtFrameworkNode::op_type_key))) {
                unconverted_ops_types[attrs.at(PtFrameworkNode::op_type_key)] = exception_msg;
            }
        }
        if (const auto& fw_node = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < fw_node->get_internal_subgraphs_size(); i++) {
                const auto& internal_types = get_unconverted_types_from_model(fw_node->get_function(i));
                unconverted_ops_types.insert(internal_types.begin(), internal_types.end());
            }
        }
    }
    return unconverted_ops_types;
}

std::string pack_detailed_failure_report(const std::map<std::string, std::string>& unconverted_ops,
                                         const std::string& additional_error = "") {
    std::stringstream error_msg;
    std::stringstream unconverted_ops_msg;
    std::stringstream failed_ops_msg;
    std::stringstream failed_ops_short;
    error_msg << "Model wasn't fully converted.";
    unconverted_ops_msg << "-- No conversion rule found for operations: ";
    failed_ops_msg << " Failed operations detailed log:";
    failed_ops_short << "-- Conversion is failed for: ";
    bool at_least_one = false;
    bool at_least_one_except = false;
    for (auto&& op : unconverted_ops) {
        if (op.second.empty()) {
            if (at_least_one)
                unconverted_ops_msg << ", ";
            unconverted_ops_msg << op.first;
            at_least_one = true;
        } else {
            if (at_least_one_except)
                failed_ops_short << ", ";
            failed_ops_short << op.first;
            failed_ops_msg << "\n-- " << op.first << " with a message:\n" << op.second;
            at_least_one_except = true;
        }
    }
    if (at_least_one_except)
        error_msg << failed_ops_msg.str();
    error_msg << "\nSummary:" << additional_error;
    if (at_least_one)
        error_msg << '\n' << unconverted_ops_msg.str();
    if (at_least_one_except)
        error_msg << '\n' << failed_ops_short.str();
    return error_msg.str();
}
}  // namespace

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    FRONT_END_GENERAL_CHECK(std::dynamic_pointer_cast<pytorch::InputModel>(model), "Invalid input model");
    std::map<std::string, CreatorFunction> supported_ops = get_supported_ops(model);
    std::shared_ptr<Model> converted_model;
    {
        TranslateSession translate_session(model, supported_ops, m_telemetry);
        converted_model = translate_session.get_converted_model();
    }

    std::string norm_err;
    try {
        normalize(converted_model);
    } catch (const std::exception& e) {
        norm_err = "\n-- normalize step failed with: " + std::string(e.what());
    }

    const auto& unconverted_ops = get_unconverted_types_from_model(converted_model);
    for (auto&& op : unconverted_ops) {
        if (m_telemetry) {
            m_telemetry->send_event("error_cause", "pytorch_" + op.first);
        }
    }
    bool is_conversion_successful = unconverted_ops.size() == 0 && norm_err.empty();
    FRONT_END_OP_CONVERSION_CHECK(is_conversion_successful, pack_detailed_failure_report(unconverted_ops, norm_err));
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<Model>& partiallyConverted) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    FRONT_END_GENERAL_CHECK(std::dynamic_pointer_cast<pytorch::InputModel>(model), "Invalid input model");
    std::map<std::string, CreatorFunction> supported_ops = get_supported_ops(model);
    std::shared_ptr<Model> partial_model;
    {
        TranslateSession translate_session(model, supported_ops, m_telemetry);
        partial_model = translate_session.get_converted_model();
    }
    try {
        normalize(partial_model);
    } catch (...) {
        // normalize can fail on transformation, but the model is still valid. We can return such model.
        // If model can be validated we suppress normalize exception.
        partial_model->validate_nodes_and_infer_types();
    }
    return partial_model;
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    ov::pass::Manager manager;

    // the following 2 transformations are needed for keypoint detectron2 models to work.
    // AtenIndexToSelect will be called twice
    manager.register_pass<ov::pass::ConvertConvertLike>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenIndexToSelect>();

    // Mark quantized and f16/bf16 compressed constants to prevent CF for them,
    // so that not extra memory is used for intermediate decompressed constants.
    manager.register_pass<ov::pass::MarkDequantizationSubgraph>(
        element::TypeVector{element::u8, element::i8, element::u4, element::i4});
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
    manager.register_pass<ov::pass::ConstantFolding>();

    manager.register_pass<ov::pass::ConvertConvertPromoteTypes>();
    manager.register_pass<ov::pass::PushConstantToSubgraph>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.register_pass<ov::frontend::pytorch::pass::TupleUnpackInBodyReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenCatToConcat>();
    manager.register_pass<ov::frontend::pytorch::pass::AppendListUnpackReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenStackListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimListUnpackReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenGetItemReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::ListConstructReplacer>();
    // TODO: remove AtenIndexToSelect when problem with  dynamic input rank is gone.
    manager.register_pass<ov::frontend::pytorch::pass::AtenIndexToSelect>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenIndexPutReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimListConstructPadReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenEinsumListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::MinMaxPrimListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::StringEqualityReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::DecomposeUnpackParameters>();
    manager.register_pass<ov::frontend::pytorch::pass::RFFTNComplexReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::IRFFTNComplexReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimTupleUnpackReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::DecomposeListTupleResults>();
    manager.register_pass<ov::frontend::pytorch::pass::DictParameterResolver>();
    manager.register_pass<ov::frontend::pytorch::pass::DictResultResolver>();
    manager.register_pass<ov::frontend::pytorch::pass::IndexLoopGetitemReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::QuantizedNodeRemover>();
    manager.register_pass<ov::frontend::pytorch::pass::SoftmaxReshapeElimination>();
    manager.register_pass<ov::frontend::pytorch::pass::U4BlockRepack>();
    manager.register_pass<ov::frontend::pytorch::pass::ReversepropResolver>();
    manager.register_pass<ov::frontend::pytorch::pass::MovePackThroughLstm>();
    manager.register_pass<ov::frontend::pytorch::pass::RemovePackingOps>();
    manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    manager.register_pass<ov::pass::ReverseShapeAndTypeInfer>();
    manager.register_pass<ov::pass::ResolveNameCollisions>(true);
    manager.run_passes(model);

    // Usually if nn.Module.forward is given as a source model for conversion, there is the first Parameter
    // that represents original `self` argument in forward(self, ...). `self` shouldn't play any role in model
    // inference if model is completely frozen and all methods are inlined. So we check if it doesn't have any
    // consumers in the finally converted model and remove this parameter. This parameter should have index 0.
    if (model->get_parameters().size() > 0) {
        auto self = model->get_parameters()[0];
        if (self->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            OPENVINO_DEBUG << "[ WARNING ] Removing parameter[0] in converted Pytorch model, because it is never used "
                              "and treated as `self`\n";
            model->remove_parameter(self);
        } else {
            OPENVINO_DEBUG << "[ WARNING ] Couldn't remove parameter[0] in converted PyTorch model\n";
        }
    }
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(conv_ext);
        m_op_extension_translators[conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return conv_ext->get_converter()(context);
        };
    } else if (auto conv_ext = std::dynamic_pointer_cast<ov::frontend::pytorch::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(conv_ext);
        m_op_extension_translators[conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return conv_ext->get_converter()(context);
        };
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (const auto& telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    // Currently PyTorch FrontEnd only support TorchDecoder as input
    if (variants.size() != 1 + extra_variants_num || !variants[0].is<std::shared_ptr<IDecoder>>())
        return false;
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    return decoder && std::dynamic_pointer_cast<TorchDecoder>(decoder);
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    FRONT_END_GENERAL_CHECK(variants.size() == 1 + extra_variants_num,
                            "PyTorch Frontend supports exactly one parameter in model representation, got ",
                            std::to_string(variants.size()),
                            " instead.");
    FRONT_END_GENERAL_CHECK(variants[0].is<std::shared_ptr<IDecoder>>(),
                            "PyTorch Frontend doesn't support provided model type. Please provide supported model "
                            "object using Python API.");
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    auto tdecoder = std::dynamic_pointer_cast<TorchDecoder>(decoder);
    FRONT_END_GENERAL_CHECK(tdecoder, "Couldn't cast ov::Any to TorchDecoder");
    return std::make_shared<pytorch::InputModel>(tdecoder);
}

std::map<std::string, CreatorFunction> FrontEnd::get_supported_ops(const ov::frontend::InputModel::Ptr& model) const {
    std::map<std::string, CreatorFunction> supported_ops;
    if (std::dynamic_pointer_cast<pytorch::InputModel>(model)->decoder_type_name() == "fx")
        supported_ops = get_supported_ops_fx();
    else
        supported_ops = get_supported_ops_ts();
    for (auto i = m_op_extension_translators.begin(); i != m_op_extension_translators.end(); i++)
        supported_ops[i->first] = i->second;
    return supported_ops;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
