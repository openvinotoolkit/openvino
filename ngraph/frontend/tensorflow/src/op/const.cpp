// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateConstOp(const NodeContext& node) {
    auto dt = node.get_attribute<ngraph::element::Type>("dtype");
    Output<Node> ng_node;

    // For some reason the following do not work (no specialization of
    // tensorflow::checkpoint::SavedTypeTraits...)
    // case DataType::DT_UINT32:
    //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, element::u32,
    //   &ng_node));
    //   break;
    // case DataType::DT_UINT64:
    //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, element::u64,
    //   &ng_node));
    //   break;
    try {
        const auto& func_param = Builder::TF_NGRAPH_CONST_MAP().at(dt);
        TF_RETURN_IF_ERROR(func_param.first(node, func_param.second, ng_node));
    } catch (const std::out_of_range&) {
        throw errors::Unimplemented("Failed to translate Constant with target ngraph type:" + dt.get_type_name());
    }

    return {ng_node};
}
}
}