// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>

using namespace ngraph;

bool ngraph::pass::ConvertPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    static std::map<ngraph::NodeTypeInfo, std::function<bool(std::shared_ptr<Node>, element::Type, size_t idx)>> type_to_fuse {
        {opset3::ShapeOf::type_info, fuse_type_to_shapeof},
        {opset3::Convert::type_info, fuse_type_to_convert},
        {opset3::Constant::type_info, fuse_type_to_constant},
        {opset3::Parameter::type_info, fuse_type_to_parameter},
    };
    // TODO: add nodes from body
    for (auto &node : f->get_ordered_ops()) {
        node->validate_and_infer_types();

        for (auto output : node->outputs()) {
            if (output.get_element_type() == m_from) {
                // If node type in map and convert can be fused into node we skip Convert creation
                if (type_to_fuse.count(node->get_type_info()) &&
                    type_to_fuse.at(node->get_type_info())(node, m_to, output.get_index())) {
                    continue;
                }
                // Create Convert operation and reconnect consumers
                auto consumers = output.get_target_inputs();
                auto convert = std::make_shared<opset3::Convert>(output, m_to);
                for (auto & input : consumers) {
                    input.replace_source_output(convert);
                }
            }
        }
    }

    // Check that function has no more operations with output type equal to m_to
    for (auto &node : f->get_ordered_ops()) {
        for (auto &output : node->outputs()) {
        }
    }
}

bool fuse_type_to_shapeof(std::shared_ptr<Node> node, element::Type to, size_t idx) {
    if (auto shapeof = as_type_ptr<opset3::ShapeOf>(node)) {
        if (to == element::i32 || to == element::i64) {
            shapeof->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_parameter(std::shared_ptr<Node> node, element::Type to, size_t idx) {
    if (auto param = as_type_ptr<opset3::Parameter>(node)) {
        param->set_element_type(to);
    }
    return false;
}

bool fuse_type_to_convert(std::shared_ptr<Node> node, element::Type to, size_t idx) {
    if (auto convert = as_type_ptr<opset3::Convert>(node)) {
        convert->set_convert_element_type(to);
    }
    return false;
}

template <element::Type_t PREC_FROM, element::Type_t PREC_TO>
std::shared_ptr<Node> change_constant_precision(std::shared_ptr<opset3::Constant> constant) {
    using src_type = typename element_type_traits<PREC_FROM>::value_type;
    using dst_type = typename element_type_traits<PREC_TO>::value_type;
    return std::make_shared<ngraph::opset3::Constant>(PREC_TO, constant->get_shape(), constant->cast_vector<dst_type>());
}

bool fuse_type_to_constant(std::shared_ptr<Node> node, element::Type to, size_t idx) {
    if (auto constant = as_type_ptr<opset3::Constant>(node)) {
        auto from = constant->get_element_type();
        std::shared_ptr<Node> new_const;
        if (from == element::u64 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u64, element::Type_t::i32>(constant);
        } else if (from == element::i64 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::i64, element::Type_t::i32>(constant);
        } else if (from == element::u8 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u8, element::Type_t::i32>(constant);
        } else if (from == element::u16 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u16, element::Type_t::i32>(constant);
        } else if (from == element::u32 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u32, element::Type_t::i32>(constant);
        } else if (from == element::f16 && to == element::f32) {
            new_const = change_constant_precision<element::Type_t::f16, element::Type_t::f32>(constant);
        } else if (from == element::boolean && to == element::u8) {
            new_const = change_constant_precision<element::Type_t::boolean, element::Type_t::u8>(constant);
        } else if (from == element::boolean && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::boolean, element::Type_t::i32>(constant);
        } else {
            throw ngraph_error("not supported");
        }
        replace_node(constant, new_const);
    }
    return false;
}


