// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/frontend.hpp"

#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transforms.hpp"
#include "transforms/append_list_unpack_replacer.hpp"
#include "transforms/aten_cat_replacer.hpp"
#include "transforms/aten_getitem_replacer.hpp"
#include "transforms/aten_index_replacer.hpp"
#include "transforms/aten_stack_list_construct_replacer.hpp"
#include "transforms/einsum_list_construct.hpp"
#include "transforms/listconstruct_replacer.hpp"
#include "transforms/min_max_prim_list_construct_replacer.hpp"
#include "transforms/prim_list_construct_pad.hpp"
#include "transforms/prim_list_tuple_construct_replacer.hpp"
#include "transforms/prim_list_unpack_replacer.hpp"
#include "transforms/string_equality_replacer.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

namespace {
std::set<std::string> get_unconverted_types_from_model(const std::shared_ptr<Model>& model) {
    std::set<std::string> unconverted_ops_types;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<PtFrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            unconverted_ops_types.insert(op_type);
        }
        if (const auto& fw_node = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < fw_node->get_internal_subgraphs_size(); i++) {
                auto internal_types = get_unconverted_types_from_model(fw_node->get_function(i));
                unconverted_ops_types.insert(internal_types.begin(), internal_types.end());
            }
        }
    }
    return unconverted_ops_types;
}
}  // namespace

FrontEnd::FrontEnd() : m_op_translators(get_supported_ops()) {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto converted_model = convert_partially(model);
    normalize(converted_model);
    std::set<std::string> unconverted_ops_types = get_unconverted_types_from_model(converted_model);
    std::stringstream ops_str;
    for (auto&& op_type : unconverted_ops_types) {
        ops_str << op_type << '\n';
    }
    FRONT_END_OP_CONVERSION_CHECK(unconverted_ops_types.size() == 0,
                                  "Model wasn't fully converted. Unconverted operation types:\n" + ops_str.str());
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<Model>& partiallyConverted) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    FRONT_END_GENERAL_CHECK(std::dynamic_pointer_cast<pytorch::InputModel>(model), "Invalid input model");
    try {
        TranslateSession translate_session(model, m_op_translators);
        return translate_session.get_converted_model();
    } catch (const std::runtime_error& e) {
        std::cerr << "[ ERROR ] Unexpected error while converting pytorch model: " << e.what() << '\n';
        std::cerr << "Rethrowing. Misleading error message from pybind11 may come next. TODO.";
        throw;
    }
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    ov::pass::Manager manager;

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::PushConstantToSubgraph>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenCatToConcat>();
    manager.register_pass<ov::frontend::pytorch::pass::AppendListUnpackReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenStackListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimListUnpackReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenGetItemReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::ListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenIndexToSelect>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimListConstructPadReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::AtenEinsumListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::MinMaxPrimListConstructReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::StringEqualityReplacer>();
    manager.register_pass<ov::frontend::pytorch::pass::DecomposeListTupleResults>();
    manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    manager.register_pass<ov::pass::ReverseShapeAndTypeInfer>();

    manager.run_passes(model);

    apply_pytorch_conversion_transforms(model);

    // Usually if nn.Module.forward is given as a source model for conversion, there is the first Parameter
    // that represents original `self` argument in forward(self, ...). `self` shouldn't play any role in model
    // inference if model is completelly frozed and all methods are inlined. So we check if it doesn't have any
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
    // Extension loading mechanism is not implemented, any extensions will be ignored
    // see CVS-98766 for tracking progress
    return;
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Currently PyTorch FrontEnd only support TorchDecoder as input
    if (variants.size() != 1 || !variants[0].is<std::shared_ptr<IDecoder>>())
        return false;
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    return decoder && std::dynamic_pointer_cast<TorchDecoder>(decoder);
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(variants.size() == 1,
                            "PyTorch Frontend supports exactly one parameter in model representation, got ",
                            std::to_string(variants.size()),
                            " instead.");
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    auto tdecoder = std::dynamic_pointer_cast<TorchDecoder>(decoder);
    FRONT_END_GENERAL_CHECK(tdecoder, "Couldn't cast ov::Any to TorchDecoder");
    return std::make_shared<pytorch::InputModel>(tdecoder);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
