// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "graph_iterator_saved_model.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "input_model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "tensor_bundle.pb.h"

using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

// Reading variable from shard file
template <typename T>
static std::shared_ptr<ov::Node> read_variable(std::shared_ptr<SavedModelVariablesIndex> var_index,
                                               const ::tensorflow::BundleEntryProto& entry,
                                               const NodeContext& node) {
    auto ov_type = node.get_attribute<element::Type>("dtype");
    std::vector<T> var_data;
    auto shape = node.get_attribute<::ov::PartialShape>("shape").get_shape();
    size_t size = 1;
    for (int i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    var_data.resize(size);
    TENSORFLOW_OP_VALIDATION(node,
                             size == (entry.size() / sizeof(T)),
                             "[TensorFlow Frontend] Internal error: Available data size isn't equal to calculated.");
    auto fs = var_index->get_data_file(entry.shard_id());
    if (!fs.get()) {
        TENSORFLOW_OP_VALIDATION(node, var_index, "[TensorFlow Frontend] Internal error: Cannot get shard file.");
    }
    fs->seekg(entry.offset(), std::ios::beg);
    fs->read(reinterpret_cast<char*>(var_data.data()), entry.size());
    return std::make_shared<Constant>(ov_type, shape, var_data);
}

OutputVector translate_varhandle_op(const NodeContext& node) {
    auto translate_session = node.get_translate_session();
    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    auto model = reinterpret_cast<ov::frontend::tensorflow::InputModel*>(translate_session->get_input_model().get());
    auto var_index = model->get_variables_index();
    auto ov_type = node.get_attribute<element::Type>("dtype");
    std::shared_ptr<Node> const_node;
    if (ov_type == element::undefined) {
        const_node = std::make_shared<UnsupportedConstant>();
    } else {
        // Getting variable description from variables index
        char* entry_data = nullptr;
        size_t entry_size = 0;
        auto var_name = node.get_name();
        bool result = var_index->get_mapped_variable(var_name, &entry_data, &entry_size);

        if (!result) {
            // For debug purposes
            TENSORFLOW_OP_VALIDATION(node,
                                     result,
                                     "[TensorFlow Frontend] Internal error: Cannot get mapped variable data.");
        }

        ::tensorflow::BundleEntryProto entry;
        TENSORFLOW_OP_VALIDATION(node,
                                 entry.ParseFromArray(entry_data, static_cast<int>(entry_size)),
                                 "[TensorFlow Frontend] Internal error: Cannot get read bundle entry.");
        switch (ov_type) {
        case ov::element::u8:
            const_node = read_variable<uint8_t>(var_index, entry, node);
            break;
        case ov::element::i8:
            const_node = read_variable<int8_t>(var_index, entry, node);
            break;
        case ov::element::i16:
            const_node = read_variable<int16_t>(var_index, entry, node);
            break;
        case ov::element::i32:
            const_node = read_variable<int32_t>(var_index, entry, node);
            break;
        case ov::element::i64:
            const_node = read_variable<int64_t>(var_index, entry, node);
            break;
        case ov::element::f16:
            const_node = read_variable<float16>(var_index, entry, node);
            break;
        case ov::element::f32:
            const_node = read_variable<float>(var_index, entry, node);
            break;
        case ov::element::f64:
            const_node = read_variable<double>(var_index, entry, node);
            break;
        case ov::element::bf16:
            const_node = read_variable<bfloat16>(var_index, entry, node);
            break;
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    }
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_varisinitialized_op(const NodeContext& node) {
    std::shared_ptr<Node> const_node = std::make_shared<Constant>(::ov::element::boolean, Shape{}, true);
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_assignvariable_op(const NodeContext& node) {
    return {};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
