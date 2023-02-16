// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/special_types.hpp"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

namespace {
const std::map<::tensorflow::DataType, ov::element::Type>& TYPE_MAP() {
    static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};
    return type_map;
}

template <typename T>
void extract_tensor_content(const std::string& tensor_content, ov::Tensor* values) {
    const auto tensor_content_size = tensor_content.size();
    FRONT_END_GENERAL_CHECK(tensor_content_size % sizeof(T) == 0,
                            "Size of tensor_content (",
                            tensor_content_size,
                            ") is not a multiple of ",
                            sizeof(T));

    const T* tensor_values = reinterpret_cast<const T*>(tensor_content.data());
    FRONT_END_GENERAL_CHECK(values->get_size() == tensor_content_size / sizeof(T),
                            "Size of tensor is not equal to tensor_content size.");
    std::copy(tensor_values, tensor_values + tensor_content_size / sizeof(T), values->data<T>());
}

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)  // possible loss of data
#    pragma warning(disable : 4267)  // possible loss of data
#endif
template <typename T>
void extract_compressed_tensor_content(const ::tensorflow::TensorProto& tensor_proto,
                                       int64_t val_size,
                                       ov::Tensor* values) {
    auto val_lastsaved = static_cast<T>(0);
    auto values_data = values->data<T>();
    for (size_t i = 0; i < values->get_size(); i++) {
        if (val_size == 0) {
            values_data[i] = static_cast<T>(0);
        } else if (static_cast<int64_t>(i) < val_size) {
            auto val_i = static_cast<T>(0);
            switch (values->get_element_type()) {
            // TODO: there are more element types to support here
            case ov::element::boolean:
                val_i = tensor_proto.bool_val()[i];
                break;
            case ov::element::i32:
                val_i = tensor_proto.int_val()[i];
                break;
            case ov::element::i64:
                val_i = tensor_proto.int64_val()[i];
                break;
            case ov::element::f16:
                val_i = float16::from_bits(tensor_proto.half_val()[i]);
                break;
            case ov::element::f32:
                val_i = tensor_proto.float_val()[i];
                break;
            case ov::element::f64:
                val_i = tensor_proto.double_val()[i];
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + values->get_element_type().get_type_name());
            }
            values_data[i] = val_i;
            val_lastsaved = val_i;
        } else {
            values_data[i] = val_lastsaved;
        }
    }
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace

ov::Any DecoderProto::get_attribute(const std::string& name) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    switch (attrs[0].value_case()) {
    case ::tensorflow::AttrValue::ValueCase::kB:
        return attrs[0].b();
    case ::tensorflow::AttrValue::ValueCase::kF:
        return attrs[0].f();
    case ::tensorflow::AttrValue::ValueCase::kS:
        return attrs[0].s();
    case ::tensorflow::AttrValue::ValueCase::kI:
        return attrs[0].i();
    case ::tensorflow::AttrValue::ValueCase::kShape: {
        const auto& tf_shape = attrs[0].shape();
        if (tf_shape.unknown_rank()) {
            return ov::PartialShape::dynamic();
        }
        auto shape_rank = tf_shape.dim_size();
        std::vector<ov::Dimension> dims(shape_rank);
        for (int i = 0; i < shape_rank; ++i) {
            dims[i] = static_cast<ov::Dimension::value_type>(tf_shape.dim(i).size());
        }
        return ov::PartialShape(dims);
    }

    case ::tensorflow::AttrValue::ValueCase::kType: {
        if (TYPE_MAP().count(attrs[0].type())) {
            return TYPE_MAP().at(attrs[0].type());
        } else {
            // for all unsupported types return undefined type
            return ov::element::undefined;
        }
    }

    case ::tensorflow::AttrValue::ValueCase::kList: {
        const auto& list = attrs[0].list();
        if (list.i_size())
            return std::vector<int64_t>(list.i().begin(), list.i().end());

        if (list.f_size())
            return std::vector<float>(list.f().begin(), list.f().end());

        if (list.s_size())
            return std::vector<std::string>(list.s().begin(), list.s().end());

        if (list.b_size())
            return std::vector<bool>(list.b().begin(), list.b().end());

        if (list.shape_size()) {
            auto shapes_size = list.shape_size();
            std::vector<ov::PartialShape> res(shapes_size);
            for (int shape_ind = 0; shape_ind < shapes_size; ++shape_ind) {
                auto shape = list.shape(shape_ind);
                if (shape.unknown_rank()) {
                    res[shape_ind] = ov::PartialShape::dynamic();
                } else {
                    auto shape_rank = shape.dim_size();
                    std::vector<ov::Dimension> dims(shape_rank);
                    for (int dim_ind = 0; dim_ind < shape_rank; ++dim_ind) {
                        dims[dim_ind] = static_cast<ov::Dimension::value_type>(shape.dim(dim_ind).size());
                    }
                    res[shape_ind] = dims;
                }
            }
            return res;
        }

        if (list.type_size()) {
            std::vector<ov::element::Type> res;
            for (int idx = 0; idx < list.type_size(); ++idx) {
                res.emplace_back(TYPE_MAP().at(list.type(idx)));
            }
            return res;
        }

        if (list.tensor_size() || list.func_size())
            FRONT_END_GENERAL_CHECK(
                false,
                "Conversion from Tensorflow to OpenVINO data type failed: List of tensors/functions type for '",
                name,
                "' attribute is not supported.");

        // If we got to this point it must mean we have empty list attribute
        return EmptyList();
    }

    case ::tensorflow::AttrValue::ValueCase::kTensor: {
        const auto& tensor_proto = attrs[0].tensor();
        const auto& tf_shape = tensor_proto.tensor_shape();
        ov::PartialShape pshape;
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            pshape.push_back(tf_shape.dim(i).size());
        }
        FRONT_END_GENERAL_CHECK(pshape.is_static(), "Dynamic shapes are not supported for Tensor attribute.");
        const auto& tf_type = tensor_proto.dtype();
        FRONT_END_GENERAL_CHECK(
            TYPE_MAP().count(tf_type),
            "Encountered unknown element type " + DataType_Name(tf_type) + " on an empty tensor_proto");
        auto ov_type = TYPE_MAP().at(tf_type);
        ov::Tensor res(ov_type, pshape.get_shape());
        auto tensor_content = tensor_proto.tensor_content();
        if (!tensor_content.empty() && tensor_proto.has_tensor_shape()) {
            switch (ov_type) {
            case ov::element::u8:
                extract_tensor_content<uint8_t>(tensor_content, &res);
                break;
            case ov::element::i8:
                extract_tensor_content<int8_t>(tensor_content, &res);
                break;
            case ov::element::i16:
                extract_tensor_content<int16_t>(tensor_content, &res);
                break;
            case ov::element::i32:
                extract_tensor_content<int32_t>(tensor_content, &res);
                break;
            case ov::element::i64:
                extract_tensor_content<int64_t>(tensor_content, &res);
                break;
            case ov::element::f16:
                extract_tensor_content<float16>(tensor_content, &res);
                break;
            case ov::element::f32:
                extract_tensor_content<float>(tensor_content, &res);
                break;
            case ov::element::f64:
                extract_tensor_content<double>(tensor_content, &res);
                break;
            case ov::element::bf16:
                extract_tensor_content<bfloat16>(tensor_content, &res);
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
            }
        } else {
            int64_t val_size = 0;
            switch (ov_type) {
            case ov::element::boolean:
                val_size = tensor_proto.bool_val_size();
                extract_compressed_tensor_content<bool>(tensor_proto, val_size, &res);
                break;
            case ov::element::i32:
                val_size = tensor_proto.int_val_size();
                extract_compressed_tensor_content<int32_t>(tensor_proto, val_size, &res);
                break;
            case ov::element::i64:
                val_size = tensor_proto.int64_val_size();
                extract_compressed_tensor_content<int64_t>(tensor_proto, val_size, &res);
                break;
            case ov::element::f16:
                val_size = tensor_proto.half_val_size();
                extract_compressed_tensor_content<float16>(tensor_proto, val_size, &res);
                break;
            case ov::element::f32:
                val_size = tensor_proto.float_val_size();
                extract_compressed_tensor_content<float>(tensor_proto, val_size, &res);
                break;
            case ov::element::f64:
                val_size = tensor_proto.double_val_size();
                extract_compressed_tensor_content<double>(tensor_proto, val_size, &res);
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
            }
        }
        return res;
    }
    case ::tensorflow::AttrValue::ValueCase::kPlaceholder:
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: Placeholder type for '",
                                name,
                                "' attribute is not supported.");
    case ::tensorflow::AttrValue::ValueCase::kFunc:
        // attrs[0].func() returns NameAttrList object from which
        // we retrieve the function name
        // Further, InputModel object is created for FunctionDef with this name
        // and is converted to ov::Model object.
        return attrs[0].func().name();
    default:
        FRONT_END_GENERAL_CHECK(false, "Conversion from Tensorflow to OpenVINO data type failed.");
    }
}

size_t DecoderProto::get_input_size() const {
    return m_node_def->input_size();
}

void DecoderProto::get_input_node(size_t input_port_idx,
                                  std::string& producer_name,
                                  size_t& producer_output_port_index) const {
    // Body graph nodes may have two colons `:`, for example,
    // producer_name:z:2 means that producer operation name is `producer_name`
    // and output port is 2
    std::string producer_port_name = m_node_def->input(static_cast<int>(input_port_idx));
    auto first_colon = producer_port_name.find_first_of(":");
    auto last_colon = producer_port_name.find_last_of(":");
    if (first_colon != std::string::npos && last_colon != std::string::npos) {
        producer_name = producer_port_name.substr(0, first_colon);
        auto port_id = producer_port_name.substr(last_colon + 1);
        FRONT_END_GENERAL_CHECK(!port_id.empty() && std::all_of(port_id.begin(), port_id.end(), ::isdigit),
                                "Port id is not specified or not a number. Value: ",
                                port_id);
        producer_output_port_index = std::stoi(port_id);
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

const std::string& DecoderProto::get_op_type() const {
    return m_node_def->op();
}

const std::string& DecoderProto::get_op_name() const {
    return m_node_def->name();
}

std::vector<::tensorflow::AttrValue> DecoderProto::decode_attribute_helper(const std::string& name) const {
    auto attr_map = m_node_def->attr();
    if (attr_map.contains(name)) {
        auto value = m_node_def->attr().at(name);
        return {value};
    } else {
        return {};
    }
}
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
