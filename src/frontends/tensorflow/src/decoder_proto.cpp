// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "openvino/frontend/tensorflow/node_context.hpp"

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
}  // namespace

ov::Any DecoderProto::get_native_attribute(const std::string& name) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    switch (attrs[0].value_case()) {
    case ::tensorflow::AttrValue::ValueCase::kTensor:
        return attrs[0].tensor();
    case ::tensorflow::AttrValue::ValueCase::kType:
        return attrs[0].type();
    default:
        FRONT_END_GENERAL_CHECK(false, "DataType is not covered.");
    }
}

template <typename T>
void extract_tensor_content(const std::string& tensor_content, ov::Tensor* values) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
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
        std::vector<ov::Dimension> dims;
        const auto& tf_shape = attrs[0].shape();
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.emplace_back(tf_shape.dim(i).size());
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
            std::vector<ov::PartialShape> res;
            for (const auto& it : list.shape()) {
                std::vector<ov::Dimension> dims;
                for (int i = 0; i < it.dim_size(); i++) {
                    dims.emplace_back(it.dim(i).size());
                }
                res.emplace_back(dims);
            }
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

        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: List type for '",
                                name,
                                "' attribute is not supported.");
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
        auto ov_type = TYPE_MAP().at(tf_type);
        ov::Tensor res(ov_type, pshape.get_shape());
        auto tensor_content = tensor_proto.tensor_content();
        if (!tensor_content.empty() && tensor_proto.has_tensor_shape()) {
            switch (ov_type) {
            // TODO: there are more element types to support here
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
            return res;
        }
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: Tensor type for '",
                                name,
                                "' attribute is not supported.");
    }
    case ::tensorflow::AttrValue::ValueCase::kPlaceholder:
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: Placeholder type for '",
                                name,
                                "' attribute is not supported.");
    case ::tensorflow::AttrValue::ValueCase::kFunc:
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: Function type for '",
                                name,
                                "' attribute is not supported.");
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
    // TODO: handle body graph nodes with a couple of columns
    std::string producer_port_name = m_node_def->input(input_port_idx);
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != std::string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
        auto port_id = producer_port_name.substr(delim_pos + 1);
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
