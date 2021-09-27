// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "node_context.hpp"

namespace ngraph {
namespace frontend {
std::map<::tensorflow::DataType, ngraph::element::Type> TYPE_MAP{
    {::tensorflow::DataType::DT_BOOL, ngraph::element::boolean},
    {::tensorflow::DataType::DT_INT16, ngraph::element::i16},
    {::tensorflow::DataType::DT_INT32, ngraph::element::i32},
    {::tensorflow::DataType::DT_INT64, ngraph::element::i64},
    {::tensorflow::DataType::DT_HALF, ngraph::element::f16},
    {::tensorflow::DataType::DT_FLOAT, ngraph::element::f32},
    {::tensorflow::DataType::DT_DOUBLE, ngraph::element::f64},
    {::tensorflow::DataType::DT_UINT8, ngraph::element::u8},
    {::tensorflow::DataType::DT_INT8, ngraph::element::i8},
    {::tensorflow::DataType::DT_BFLOAT16, ngraph::element::bf16}};

std::shared_ptr<Variant> DecoderTFProto::get_attribute(const std::string& name,
                                                       const VariantTypeInfo& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return nullptr;
    }

    if (type_info == VariantWrapper<std::string>::type_info) {
        return std::make_shared<VariantWrapper<std::string>>(attrs[0].s());
    } else if (type_info == VariantWrapper<int64_t>::type_info) {
        return std::make_shared<VariantWrapper<int64_t>>(attrs[0].i());
    } else if (type_info == VariantWrapper<std::vector<int64_t>>::type_info) {
        std::vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<int64_t>>>(longs);
    } else if (type_info == VariantWrapper<int32_t>::type_info) {
        return std::make_shared<VariantWrapper<int32_t>>(static_cast<int32_t>(attrs[0].i()));
    } else if (type_info == VariantWrapper<std::vector<int32_t>>::type_info) {
        std::vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return std::make_shared<VariantWrapper<std::vector<int32_t>>>(ints);
    } else if (type_info == VariantWrapper<float>::type_info) {
        return std::make_shared<VariantWrapper<float>>(attrs[0].f());
    } else if (type_info == VariantWrapper<std::vector<float>>::type_info) {
        std::vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<float>>>(floats);
    } else if (type_info == VariantWrapper<ngraph::element::Type>::type_info) {
        auto data_type = attrs[0].type();
        return std::make_shared<VariantWrapper<ngraph::element::Type>>(TYPE_MAP[data_type]);
    } else if (type_info == VariantWrapper<bool>::type_info) {
        return std::make_shared<VariantWrapper<bool>>(attrs[0].b());
    } else if (type_info == VariantWrapper<::tensorflow::DataType>::type_info) {
        return std::make_shared<VariantWrapper<::tensorflow::DataType>>(attrs[0].type());
    } else if (type_info == VariantWrapper<::tensorflow::TensorProto>::type_info) {
        return std::make_shared<VariantWrapper<::tensorflow::TensorProto>>(attrs[0].tensor());
    } else if (type_info == VariantWrapper<::ngraph::PartialShape>::type_info) {
        std::vector<ngraph::Dimension> dims;
        auto tf_shape = attrs[0].shape();
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.push_back(tf_shape.dim(i).size());
        }
        auto pshape = ngraph::PartialShape(dims);
        return std::make_shared<VariantWrapper<::ngraph::PartialShape>>(pshape);
    }

    // type is not supported by decoder
    return nullptr;
}

size_t DecoderTFProto::get_input_size() const {
    return m_node_def->input_size();
}

void DecoderTFProto::get_input_node(const size_t input_port_idx,
                                    std::string& producer_name,
                                    size_t& producer_output_port_index) const {
    // TODO: handle body graph nodes with a couple of columns
    std::string producer_port_name = m_node_def->input(input_port_idx);
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != std::string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
        producer_output_port_index = std::stoi(producer_port_name.substr(delim_pos));
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

std::string DecoderTFProto::get_op_type() const {
    return m_node_def->op();
}

std::string DecoderTFProto::get_op_name() const {
    return m_node_def->name();
}

std::vector<::tensorflow::AttrValue> DecoderTFProto::decode_attribute_helper(const std::string& name) const {
    auto attr_map = m_node_def->attr();
    FRONT_END_GENERAL_CHECK(attr_map.contains(name),
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            this->get_op_type(),
                            "node");
    auto value = m_node_def->attr().at(name);
    return {value};
}

}  // namespace frontend
}  // namespace ngraph
