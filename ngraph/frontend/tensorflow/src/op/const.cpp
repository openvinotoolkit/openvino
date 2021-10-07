// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

namespace {
using ConstMap =
    std::map<ngraph::element::Type,
             std::pair<std::function<Status(const NodeContext&, ngraph::element::Type, ngraph::Output<ngraph::Node>&)>,
                       const ngraph::element::Type>>;

const ConstMap& TF_NGRAPH_CONST_MAP() {
    static const ConstMap the_map = {
        {ngraph::element::f32, make_pair(MakeConstOp<float>, ngraph::element::f32)},
        {ngraph::element::f64, make_pair(MakeConstOp<double>, ngraph::element::f64)},
        {ngraph::element::i8, make_pair(MakeConstOp<int8_t>, ngraph::element::i8)},
        {ngraph::element::i16, make_pair(MakeConstOp<int16_t>, ngraph::element::i16)},
#if 0
      {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ngraph::element::i8)},
      {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ngraph::element::u8)},
      {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ngraph::element::u16)},
#endif
        {ngraph::element::i32, make_pair(MakeConstOp<int32_t>, ngraph::element::i32)},
        {ngraph::element::i64, make_pair(MakeConstOp<int64_t>, ngraph::element::i64)},
        {ngraph::element::u8, make_pair(MakeConstOp<uint8_t>, ngraph::element::u8)},
        {ngraph::element::u16, make_pair(MakeConstOp<uint16_t>, ngraph::element::u16)},
        {ngraph::element::boolean, make_pair(MakeConstOp<bool, char>, ngraph::element::boolean)}
    };
    return the_map;
}
}  // namespace

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
        const auto& func_param = TF_NGRAPH_CONST_MAP().at(dt);
        TF_RETURN_IF_ERROR(func_param.first(node, func_param.second, ng_node));
    } catch (const std::out_of_range&) {
        throw errors::Unimplemented("Failed to translate Constant with target ngraph type:" + dt.get_type_name());
    }

    return {ng_node};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
