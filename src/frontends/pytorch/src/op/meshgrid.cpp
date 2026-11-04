// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/internal_op.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/pass/node_registry.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_meshgrid(const NodeContext& context) {
    // aten::meshgrid(Tensor[] tensors, str? indexing) in TorchScript, indexing is kwarg-only in FX.
    num_inputs_check(context, 1, 2);
    std::string indexing = "ij";
    if (!context.input_is_none(1)) {
        indexing = context.const_input<std::string>(1);
    } else if (context.has_attribute("indexing")) {
        indexing = context.get_attribute<std::string>("indexing");
    }
    PYTORCH_OP_CONVERSION_CHECK(indexing == "ij" || indexing == "xy",
                                "aten::meshgrid: unsupported indexing mode: ",
                                indexing);
    if (const auto list = ov::as_type_ptr<SequenceMark>(context.get_input(0).get_node_shared_ptr())) {
        ov::pass::NodeRegistry rg;
        const auto outputs = build_meshgrid(rg, list->get_sequence(), indexing);
        context.mark_nodes(rg.get());
        return {context.mark_node(make_list_construct(outputs))};
    }

    // Unresolved list input: defer to PrimListUnpackReplacer.
    const bool is_fx = context.get_op_type() != "aten::meshgrid";
    std::shared_ptr<PtFrameworkNode> meshgrid;
    if (is_fx) {
        meshgrid = std::make_shared<PtFrameworkNode>(std::make_shared<InternalOpDecoder>("aten::meshgrid", 1),
                                                     OutputVector{context.get_input(0)});
    } else {
        meshgrid =
            std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(), context.get_output_size());
    }
    auto attrs = meshgrid->get_attrs();
    attrs["indexing"] = indexing;
    meshgrid->set_attrs(attrs);
    context.mark_node(meshgrid);
    if (!is_fx) {
        return meshgrid->outputs();
    }
    // FX has no prim::ListUnpack, so build the TorchScript pattern resolved by PrimListUnpackReplacer.
    const auto count = context.get_decoder()->output_list_size();
    auto unpack = context.mark_node(
        std::make_shared<PtFrameworkNode>(std::make_shared<InternalOpDecoder>("prim::ListUnpack", count),
                                          meshgrid->outputs()));
    return {context.mark_node(make_list_construct(unpack->outputs()))};
};

}  // namespace ov::frontend::pytorch::op
