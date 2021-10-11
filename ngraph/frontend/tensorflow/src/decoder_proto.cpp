// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "node_context.hpp"

using namespace std;

namespace ngraph {
namespace frontend {
namespace tf {

template <class T>
bool is_type(const VariantTypeInfo& type_info) {
    return type_info == VariantWrapper<T>::get_type_info_static();
}

template <class T>
shared_ptr<VariantWrapper<T>> create_variant(const T& data) {
    return make_shared<VariantWrapper<T>>(data);
}

shared_ptr<Variant> DecoderTFProto::get_attribute(const string& name, const VariantTypeInfo& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return nullptr;
    }

    if (is_type<string>(type_info)) {
        return create_variant<string>(attrs[0].s());
    }
    if (is_type<int64_t>(type_info)) {
        return create_variant<int64_t>(attrs[0].i());
    } else if (is_type<vector<int64_t>>(type_info)) {
        vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return create_variant<vector<int64_t>>(longs);
    } else if (is_type<int32_t>(type_info)) {
        return create_variant<int32_t>(static_cast<int32_t>(attrs[0].i()));
    } else if (is_type<vector<int32_t>>(type_info)) {
        vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return create_variant<vector<int32_t>>(ints);
    } else if (is_type<float>(type_info)) {
        return create_variant<float>(attrs[0].f());
    } else if (is_type<vector<float>>(type_info)) {
        vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return create_variant<vector<float>>(floats);
    } else if (is_type<ngraph::element::Type>(type_info)) {
        auto data_type = attrs[0].type();
        return create_variant<ngraph::element::Type>(TYPE_MAP[data_type]);
    } else if (is_type<bool>(type_info)) {
        return create_variant<bool>(attrs[0].b());
    } else if (is_type<::tensorflow::DataType>(type_info)) {
        return create_variant<::tensorflow::DataType>(attrs[0].type());
    } else if (is_type<::tensorflow::TensorProto>(type_info)) {
        return create_variant<::tensorflow::TensorProto>(attrs[0].tensor());
    } else if (is_type<::ngraph::PartialShape>(type_info)) {
        vector<ngraph::Dimension> dims;
        auto tf_shape = attrs[0].shape();
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.push_back(tf_shape.dim(i).size());
        }
        auto pshape = ngraph::PartialShape(dims);
        return create_variant<::ngraph::PartialShape>(pshape);
    }

    // type is not supported by decoder
    return nullptr;
}

size_t DecoderTFProto::get_input_size() const {
    return m_node_def->input_size();
}

void DecoderTFProto::get_input_node(const size_t input_port_idx,
                                    string& producer_name,
                                    size_t& producer_output_port_index) const {
    // TODO: handle body graph nodes with a couple of columns
    string producer_port_name = m_node_def->input(input_port_idx);
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
        producer_output_port_index = stoi(producer_port_name.substr(delim_pos));
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

string DecoderTFProto::get_op_type() const {
    return m_node_def->op();
}

string DecoderTFProto::get_op_name() const {
    return m_node_def->name();
}

vector<::tensorflow::AttrValue> DecoderTFProto::decode_attribute_helper(const string& name) const {
    auto attr_map = m_node_def->attr();
    if (attr_map.contains(name))
        return {m_node_def->attr().at(name)};
    return {};
}
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
