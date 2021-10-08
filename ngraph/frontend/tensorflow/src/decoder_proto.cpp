// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "node_context.hpp"

namespace ngraph {
namespace frontend {
namespace tf {

std::shared_ptr<ov::Variant> DecoderTFProto::get_attribute(const std::string& name,
                                                           const VariantTypeInfo& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return nullptr;
    }

    if (type_info == VariantWrapper<std::string>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<std::string>>(attrs[0].s());
    } else if (type_info == VariantWrapper<int64_t>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<int64_t>>(attrs[0].i());
    } else if (type_info == VariantWrapper<std::vector<int64_t>>::get_type_info_static()) {
        std::vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<int64_t>>>(longs);
    } else if (type_info == VariantWrapper<int32_t>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<int32_t>>(static_cast<int32_t>(attrs[0].i()));
    } else if (type_info == VariantWrapper<std::vector<int32_t>>::get_type_info_static()) {
        std::vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return std::make_shared<VariantWrapper<std::vector<int32_t>>>(ints);
    } else if (type_info == VariantWrapper<float>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<float>>(attrs[0].f());
    } else if (type_info == VariantWrapper<std::vector<float>>::get_type_info_static()) {
        std::vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<float>>>(floats);
    } else if (type_info == VariantWrapper<ngraph::element::Type>::get_type_info_static()) {
        auto data_type = attrs[0].type();
        return std::make_shared<VariantWrapper<ngraph::element::Type>>(TYPE_MAP[data_type]);
    } else if (type_info == VariantWrapper<bool>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<bool>>(attrs[0].b());
    } else if (type_info == VariantWrapper<::tensorflow::DataType>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<::tensorflow::DataType>>(attrs[0].type());
    } else if (type_info == VariantWrapper<::tensorflow::TensorProto>::get_type_info_static()) {
        return std::make_shared<VariantWrapper<::tensorflow::TensorProto>>(attrs[0].tensor());
    } else if (type_info == VariantWrapper<::ngraph::PartialShape>::get_type_info_static()) {
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

const std::string& DecoderTFProto::get_op_type() const {
    return m_node_def->op();
}

const std::string& DecoderTFProto::get_op_name() const {
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
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
