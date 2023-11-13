// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/special_types.hpp"
#include "ov_tensorflow/attr_value.pb.h"
#include "ov_tensorflow/node_def.pb.h"
#include "ov_tensorflow/types.pb.h"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

namespace {

template <
        typename T,
        typename std::enable_if<!std::is_same<T, std::string>::value, bool>::type = true>
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


template <
        typename T,
        typename std::enable_if<std::is_same<T, std::string>::value, bool>::type = true>
void extract_tensor_content(const std::string& tensor_content, ov::Tensor* values) {
    // FIXME: Works for scalars only

    std::cerr << "[ TF FE STR ] 1 Trying to decode this '" << tensor_content << "' as a string scalar value to initialize a tensor\n";
    *values->data<std::string>() = tensor_content;
}

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)  // possible loss of data
#    pragma warning(disable : 4267)  // possible loss of data
#endif
template <
        typename T,
        typename std::enable_if<!std::is_same<T, std::string>::value, bool>::type = true>
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

// FIXME: make it more specialized by avoiding mentioning T because it is always std::string
template <
        typename T,
        typename std::enable_if<std::is_same<T, std::string>::value, bool>::type = true>
void extract_compressed_tensor_content(const ::tensorflow::TensorProto& tensor_proto,
                                       int64_t val_size,
                                       ov::Tensor* values) {
    //std::cerr << "[ TF FE STR ] 2 Trying to decode tensor proto to string tensor\n";
    auto val_lastsaved = std::string();
    //std::cerr << ">>>> 1\n";
    auto values_data = values->data<T>();
    //std::cerr << ">>>> 2\n";
    for (auto i = 0; i < values->get_size(); i++) {
        //std::cerr << ">>>> 3\n";
        if (val_size == 0) {
            values_data[i] = std::string();
        } else if (i < val_size) {
            auto val_i = std::string();
            switch (values->get_element_type()) {
            case ov::element::string:
                //std::cerr << "[ TF FE STR ] Trying to decode this '" << tensor_proto.string_val()[i] <<  "' as one of string element values to initialize a tensor\n";
                val_i = tensor_proto.string_val()[i];
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + values->get_element_type().get_type_name());
            }
            //std::cerr << ">>>> 4\n";
            values_data[i] = val_i;
            //std::cerr << ">>>> 5\n";

            val_lastsaved = val_i;
            //std::cerr << ">>>> 6\n";
        } else {
            //std::cerr << ">>>> 7\n";
            values_data[i] = val_lastsaved;
            //std::cerr << ">>>> 8\n";
        }
    }
}
}  // namespace

ov::Any DecoderProto::get_attribute(const std::string& name) const {
    //std::cerr << "[ INFO ] Trying to decode attribute " << name << "\n";
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
        //std::cerr << "kS\n";
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
        auto atype = attrs[0].type();
        if (atype == ::tensorflow::DT_COMPLEX64) {
            return ov::Any("DT_COMPLEX64");
        } else if (atype == ::tensorflow::DT_COMPLEX128) {
            return ov::Any("DT_COMPLEX128");
        } else {
            return get_ov_type(atype);
        }
    }

    case ::tensorflow::AttrValue::ValueCase::kList: {
        //std::cerr << "kList\n";
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
                if (list.type(idx) != ::tensorflow::DataType::DT_STRING) {
                    res.emplace_back(get_ov_type(list.type(idx)));
                } else {
                    res.emplace_back(ov::element::dynamic);
                }
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
        return unpack_tensor_proto(attrs[0].tensor());
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

void parse_producer_name(const std::string& producer_port_name,
                         std::string& producer_name,
                         std::string& producer_output_port_name,
                         size_t& producer_output_port_index) {
    // Body graph nodes may have two colons `:` input names, for example,
    // `TopKV2Name:indices:0` means that producer operation name is `TopKV2Name`
    // the middle name is output port name of the producer `indices` that means
    // the second output port of TopKV2 is used.
    // The first output port of TopKV2 is described as `TopKV2Name:values:0`
    auto first_colon = producer_port_name.find_first_of(":");
    auto last_colon = producer_port_name.find_last_of(":");
    if (first_colon != std::string::npos && first_colon < last_colon) {
        // we have at least two colons producer_name:output_port_name:port_idx
        producer_name = producer_port_name.substr(0, first_colon);
        auto port_id = producer_port_name.substr(last_colon + 1);
        auto port_name = producer_port_name.substr(first_colon + 1, last_colon - first_colon - 1);
        FRONT_END_GENERAL_CHECK(!port_id.empty() && std::all_of(port_id.begin(), port_id.end(), ::isdigit),
                                "Port id is not specified or not a number. Value: ",
                                port_id);
        producer_output_port_index = std::stoi(port_id);
        producer_output_port_name = port_name;
        return;
    } else if (first_colon != std::string::npos) {
        // just one colon case
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

void DecoderProto::get_input_node(size_t input_port_idx,
                                  std::string& producer_name,
                                  std::string& producer_output_port_name,
                                  size_t& producer_output_port_index) const {
    const std::string producer_port_name = m_node_def->input(static_cast<int>(input_port_idx));
    parse_producer_name(producer_port_name, producer_name, producer_output_port_name, producer_output_port_index);
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
