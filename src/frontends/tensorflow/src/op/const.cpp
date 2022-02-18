// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

namespace {
using ConstMap = std::map<ov::element::Type,
                          std::pair<std::function<void(const NodeContext&, ov::element::Type, ov::Output<ov::Node>&)>,
                                    const ov::element::Type>>;

const ConstMap& TF_OPENVINO_CONST_MAP() {
    static const ConstMap the_map = {
        {ov::element::f32, make_pair(make_const_op<float>, ov::element::f32)},
        {ov::element::f64, make_pair(make_const_op<double>, ov::element::f64)},
        {ov::element::i8, make_pair(make_const_op<int8_t>, ov::element::i8)},
        {ov::element::i16, make_pair(make_const_op<int16_t>, ov::element::i16)},
#if 0
      {DataType::DT_QINT8, make_pair(make_const_op<qint8>, ov::element::i8)},
      {DataType::DT_QUINT8, make_pair(make_const_op<quint8>, ov::element::u8)},
      {DataType::DT_QUINT16, make_pair(make_const_op<quint16>, ov::element::u16)},
#endif
        {ov::element::i32, make_pair(make_const_op<int32_t>, ov::element::i32)},
        {ov::element::i64, make_pair(make_const_op<int64_t>, ov::element::i64)},
        {ov::element::u8, make_pair(make_const_op<uint8_t>, ov::element::u8)},
        {ov::element::u16, make_pair(make_const_op<uint16_t>, ov::element::u16)},
        {ov::element::boolean, make_pair(make_const_op<bool, char>, ov::element::boolean)}
    };
    return the_map;
}
}  // namespace

OutputVector translate_const_op(const NodeContext& node) {
    auto dt = node.get_attribute<ov::element::Type>("dtype");
    Output<Node> res;

    // TODO: fix DT_UINT32 and DT_UINT64 support
    // no specialization of tensorflow::checkpoint::SavedTypeTraits...)
    try {
        const auto& func_param = TF_OPENVINO_CONST_MAP().at(dt);
        func_param.first(node, func_param.second, res);
    } catch (const std::out_of_range&) {
        TENSORFLOW_OP_VALIDATION(node, false, "Failed to translate Constant with target OV type:" + dt.get_type_name());
    }
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov