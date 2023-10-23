// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "graph_iterator_saved_model.hpp"
#include "helper_ops/string_constant.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "input_model.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "ov_tensorflow/tensor_bundle.pb.h"

using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

// Reading variable from shard file
template <typename T>
static std::shared_ptr<ov::Node> read_variable(std::shared_ptr<VariablesIndex> var_index,
                                               const ov::element::Type ov_type,
                                               const ov::Shape shape,
                                               const ::tensorflow::BundleEntryProto& entry,
                                               const NodeContext& node) {
    google::protobuf::int64 size = 1;
    for (uint64_t i = 0; i < shape.size(); ++i) {
        size *= static_cast<google::protobuf::int64>(shape[i]);
    }
    TENSORFLOW_OP_VALIDATION(node,
                             size == static_cast<google::protobuf::int64>(entry.size() / sizeof(T)),
                             "[TensorFlow Frontend] Internal error: Available data size isn't equal to calculated.");
    if (var_index->is_mmap_enabled()) {
        auto mapped_memory = var_index->get_data_mmap(entry.shard_id());
        if (!mapped_memory.get()) {
            TENSORFLOW_OP_VALIDATION(node, var_index, "[TensorFlow Frontend] Internal error: Cannot get shard file.");
        }
        TENSORFLOW_OP_VALIDATION(
            node,
            static_cast<int64_t>(mapped_memory->size()) >= entry.offset() + entry.size(),
            "[TensorFlow Frontend] Internal error: Variable entry size is out of bounds of mapped memory size.");
        return std::make_shared<Constant>(
            ov_type,
            shape,
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mapped_memory->data() + entry.offset(),
                                                                              entry.size(),
                                                                              mapped_memory));
    } else {
        std::vector<T> var_data;
        var_data.resize(size);
        auto fs = var_index->get_data_file(entry.shard_id());
        if (!fs.get()) {
            TENSORFLOW_OP_VALIDATION(node, var_index, "[TensorFlow Frontend] Internal error: Cannot get shard file.");
        }
        fs->seekg(entry.offset(), std::ios::beg);
        fs->read(reinterpret_cast<char*>(var_data.data()), entry.size());
        return std::make_shared<Constant>(ov_type, shape, var_data);
    }
}

OutputVector translate_varhandle_op(const NodeContext& node) {
    default_op_checks(node, 0, {"VarHandleOp", "VariableV2"});
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
    } else if (var_index.get() == nullptr) {
        auto shape = node.get_attribute<::ov::PartialShape>("shape").get_shape();
        const_node = std::make_shared<Parameter>(ov_type, shape);
    } else {
        // Getting variable description from variables index
        const char* entry_data = nullptr;
        size_t entry_size = 0;
        auto var_name = node.get_name();
        auto shape = node.get_attribute<::ov::PartialShape>("shape").get_shape();
        bool result = var_index->get_mapped_variable(var_name, &entry_data, &entry_size);

        if (!result) {
            result = var_index->get_variable(var_name, &entry_data, &entry_size);
        }

        TENSORFLOW_OP_VALIDATION(node, result, "[TensorFlow Frontend] Internal error: Cannot find requested variable.");

        ::tensorflow::BundleEntryProto entry;
        TENSORFLOW_OP_VALIDATION(node,
                                 entry.ParseFromArray(entry_data, static_cast<int>(entry_size)),
                                 "[TensorFlow Frontend] Internal error: Cannot get read bundle entry.");

        switch (ov_type) {
        case ov::element::u8:
            const_node = read_variable<uint8_t>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::i8:
            const_node = read_variable<int8_t>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::i16:
            const_node = read_variable<int16_t>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::i32:
            const_node = read_variable<int32_t>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::i64:
            const_node = read_variable<int64_t>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::f16:
            const_node = read_variable<float16>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::f32:
            const_node = read_variable<float>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::f64:
            const_node = read_variable<double>(var_index, ov_type, shape, entry, node);
            break;
        case ov::element::bf16:
            const_node = read_variable<bfloat16>(var_index, ov_type, shape, entry, node);
            break;
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    }
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_varisinitialized_op(const NodeContext& node) {
    auto const_node = std::make_shared<Constant>(::ov::element::boolean, Shape{}, true);
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_readvariable_op(const NodeContext& node) {
    default_op_checks(node, 1, {"ReadVariableOp", "Assign"});

    // Documentation says it should return only one tensor with dtype, but
    // _output_shapes is a vector of shapes and it means it may have multiple outputs
    // https://www.tensorflow.org/api_docs/python/tf/raw_ops/ReadVariableOp
    auto tmp_output_shapes = node.get_attribute_as_any("_output_shapes");

    if (tmp_output_shapes.empty() || !tmp_output_shapes.is<std::vector<::ov::PartialShape>>()) {
        return {node.get_input(0).get_node_shared_ptr()};
    }

    auto output_shapes = tmp_output_shapes.as<std::vector<::ov::PartialShape>>();

    OutputVector outs = {};

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        std::shared_ptr<ov::Node> output_node;
        if (node.get_input(0).get_partial_shape().is_static() &&
            output_shapes[i].get_shape() != node.get_input(0).get_shape()) {
            auto reshape_shape = make_shared<Constant>(ov::element::i32, output_shapes[i].get_shape());
            output_node = make_shared<Reshape>(node.get_input(0), reshape_shape, false);
        } else {
            output_node = node.get_input(0).get_node_shared_ptr();
        }
        if (i == 0) {
            set_out_name(node.get_name(), output_node);
            set_out_name(node.get_name() + ":" + "0", output_node);
        } else {
            set_node_name(node.get_name() + ":" + std::to_string(i), output_node);
        }
        outs.push_back(output_node);
    }
    return outs;
}

OutputVector translate_assignvariable_op(const NodeContext& node) {
    default_op_checks(node, 2, {"AssignVariableOp"});
    auto assignvariableop_node = std::make_shared<UnsupportedConstant>();
    set_node_name(node.get_name(), assignvariableop_node);
    return {assignvariableop_node};
}

OutputVector translate_restorev2_op(const NodeContext& node) {
    default_op_checks(node, 3, {"RestoreV2"});
    auto translate_session = node.get_translate_session();
    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    auto model = reinterpret_cast<ov::frontend::tensorflow::InputModel*>(translate_session->get_input_model().get());
    auto var_index = model->get_variables_index();
    auto tensor_names =
        reinterpret_cast<StringConstant*>(node.get_input(1).get_node())->get_data().as<std::vector<std::string>>();
    auto tensor_types = node.get_attribute<std::vector<ov::element::Type>>("dtypes");

    OutputVector outs = {};

    for (size_t i = 0; i < tensor_names.size(); ++i) {
        auto const_node = std::make_shared<UnsupportedConstant>();
        if (i == 0)
            set_node_name(node.get_name(), const_node);
        else
            set_node_name(node.get_name() + ":" + std::to_string(i), const_node);
        outs.push_back(const_node);
    }

    return outs;
}

OutputVector translate_staticregexfullmatch_op(const NodeContext& node) {
    default_op_checks(node, 1, {"StaticRegexFullMatch"});
    // auto pattern = node.get_attribute_as_any("pattern").as<std::string>();
    auto const_node = std::make_shared<Constant>(ov::element::boolean, ov::Shape{}, true);
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_stringjoin_op(const NodeContext& node) {
    default_op_checks(node, 1, {"StringJoin"});
    auto const_node = std::make_shared<UnsupportedConstant>();
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

OutputVector translate_mergev2checkpoint_op(const NodeContext& node) {
    default_op_checks(node, 1, {"MergeV2Checkpoint"});
    auto const_node = std::make_shared<UnsupportedConstant>();
    set_node_name(node.get_name(), const_node);
    return {const_node};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
