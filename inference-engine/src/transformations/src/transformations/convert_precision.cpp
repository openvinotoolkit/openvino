// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>

using namespace ngraph;

bool fuse_type_to_constant(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, const std::vector<ngraph::Input<ngraph::Node>> & consumers);
bool fuse_type_to_shapeof(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_shapeof_v0(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_parameter(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_convert(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nms3(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nms4(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_topk(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nonzero(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_bucketize(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_generic_ie(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);

bool extend_select_type(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);

template <typename T>
bool fuse_type_to_binary_comparision(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(*casted, element::TypeVector{}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <typename T>
bool fuse_type_to_logical(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        type_relaxed->set_origin_input_type(element::boolean, 0);
        type_relaxed->set_origin_input_type(element::boolean, 1);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(*casted,
                element::TypeVector{element::boolean, element::boolean}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <class T>
bool fuse_type_to_reduce_logical(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        type_relaxed->set_origin_input_type(element::boolean, 0);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(*casted,
                element::TypeVector{element::boolean}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

bool ngraph::pass::ConvertPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    static std::map<ngraph::NodeTypeInfo, std::function<bool(std::shared_ptr<Node>&, element::Type, size_t idx)>> type_to_fuse {
        {opset4::Parameter::type_info, fuse_type_to_parameter},
        {opset4::Convert::type_info, fuse_type_to_convert},
        {opset4::ShapeOf::type_info, fuse_type_to_shapeof},
        {opset3::NonMaxSuppression::type_info, fuse_type_to_nms3},
        {opset4::NonMaxSuppression::type_info, fuse_type_to_nms4},
        {opset4::TopK::type_info, fuse_type_to_topk},
        {opset4::NonZero::type_info, fuse_type_to_nonzero},
        {opset4::Bucketize::type_info, fuse_type_to_bucketize},
        {NodeTypeInfo("GenericIE", 1), fuse_type_to_generic_ie},
        {opset4::Equal::type_info, fuse_type_to_binary_comparision<opset4::Equal>},
        {opset4::NotEqual::type_info, fuse_type_to_binary_comparision<opset4::NotEqual>},
        {opset4::Greater::type_info, fuse_type_to_binary_comparision<opset4::Greater>},
        {opset4::GreaterEqual::type_info, fuse_type_to_binary_comparision<opset4::GreaterEqual>},
        {opset4::Less::type_info, fuse_type_to_binary_comparision<opset4::Less>},
        {opset4::LessEqual::type_info, fuse_type_to_binary_comparision<opset4::LessEqual>},
        {opset4::LogicalAnd::type_info, fuse_type_to_logical<opset4::LogicalAnd>},
        {opset4::LogicalOr::type_info, fuse_type_to_logical<opset4::LogicalOr>},
        {opset4::LogicalXor::type_info, fuse_type_to_logical<opset4::LogicalXor>},
        {opset4::LogicalNot::type_info, fuse_type_to_logical<opset4::LogicalNot>},
        {opset4::ReduceLogicalAnd::type_info, fuse_type_to_reduce_logical<opset4::ReduceLogicalAnd>},
        {opset4::ReduceLogicalOr::type_info, fuse_type_to_reduce_logical<opset4::ReduceLogicalOr>},
        {opset1::ShapeOf::type_info, fuse_type_to_shapeof_v0}
    };

    static std::map<ngraph::NodeTypeInfo, std::function<bool(std::shared_ptr<Node>&, element::Type, size_t idx)>> type_to_extend {
            {opset4::Select::type_info, extend_select_type},
    };

    // As Constant operations can be shared between multiple nGraph Functions so before
    // changing precision we need to understand which Constant consumers belongs
    // to the current nGraph Function
    std::map<std::shared_ptr<Node>, std::vector<Input<Node>>> const_to_internal_output;

    std::function<void(const std::shared_ptr<Function> &)> register_constants =
            [&const_to_internal_output, &register_constants](const std::shared_ptr<Function> & f) {
        for (auto & node : f->get_ordered_ops()) {
            for (auto & input : node->inputs()) {
                if (auto const_node = std::dynamic_pointer_cast<opset4::Constant>(input.get_source_output().get_node_shared_ptr())) {
                    const_to_internal_output[const_node].emplace_back(input);
                }
            }
        }
    };

    auto convert_node_output_precision = [this, &const_to_internal_output](std::shared_ptr<Node> & node) {
        for (auto output : node->outputs()) {
            if (output.get_element_type() == m_from) {
                // Handle case with Constants as they can have consumers from other nGraph Function object
                if (ngraph::op::is_constant(node) && const_to_internal_output.count(node)) {
                    fuse_type_to_constant(node, m_to, const_to_internal_output.at(node));
                    break;
                }

                // Check that node type exists in map and we can fuse type into node
                if (type_to_fuse.count(node->get_type_info()) &&
                    type_to_fuse.at(node->get_type_info())(node, m_to, output.get_index())) {
                    // We need to break if original node was replaced
                    break;
                }
            }
        }
    };

    auto convert_node_input_precision = [this](std::shared_ptr<Node> & node) {
        for (auto input : node->inputs()) {
            if (input.get_element_type() == m_from) {
                // For some operations we need to extend their input types to support new type
                if (type_to_extend.count(node->get_type_info()) &&
                    type_to_extend.at(node->get_type_info())(node, m_to, input.get_index())) {
                    break;
                }
            }
        }
    };

    std::function<void(const std::shared_ptr<Function> &)> convert_function_precision =
            [this, &const_to_internal_output,
                   &register_constants,
                   &convert_node_output_precision,
                   &convert_node_input_precision,
                   &convert_function_precision] (const std::shared_ptr<Function> & f) {
        // Iterate over all nodes in topological order and then iterate over node outputs.
        // If output type mismatch given type we try to fuse type into this operation
        // otherwise we insert Convert operation.
        for (auto &node : f->get_ordered_ops()) {
            // Recursively run for TensorIterator body function
            if (auto ti = std::dynamic_pointer_cast<opset4::TensorIterator>(node)) {
                convert_function_precision(ti->get_body());
            }
            convert_node_input_precision(node);
        }
        // Register internal constants only after fixing input type that could lead to nodes replacement
        register_constants(f);

        for (auto &node : f->get_ordered_ops()) {
            convert_node_output_precision(node);
        }
    };

    convert_function_precision(f);
    f->validate_nodes_and_infer_types();

    // TODO: we need to split NopElimination pass to separate MatcherPasses and call Convert elimination here
    for (auto &node : f->get_ordered_ops()) {
        if (auto convert = std::dynamic_pointer_cast<opset4::Convert>(node)) {
            // WA for topK, dont remove fake convert
            if (convert->input(0).get_element_type() == convert->get_convert_element_type() &&
                convert->input_value(0).get_node_shared_ptr()->get_output_size() == 1) {
                replace_output_update_name(convert->output(0), convert->input_value(0));
            }
        }
    }
    return true;
}

bool fuse_type_to_shapeof(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto shapeof = as_type_ptr<opset4::ShapeOf>(node)) {
        if (to == element::i32 || to == element::i64) {
            shapeof->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_parameter(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto param = as_type_ptr<opset4::Parameter>(node)) {
        param->set_element_type(to);
        param->validate_and_infer_types();
        return true;
    }
    return false;
}

bool fuse_type_to_convert(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto convert = as_type_ptr<opset4::Convert>(node)) {
        convert->set_convert_element_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms3(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nms = as_type_ptr<opset3::NonMaxSuppression>(node)) {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms4(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nms = as_type_ptr<opset4::NonMaxSuppression>(node)) {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_topk(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto topk = as_type_ptr<opset4::TopK>(node)) {
        if (idx == 1 && (to == element::i32 || to == element::i64)) {
            topk->set_index_element_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_nonzero(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nonzero = as_type_ptr<opset4::NonZero>(node)) {
        if (to == element::i32 || to == element::i64) {
            nonzero->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_bucketize(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto b = as_type_ptr<opset4::Bucketize>(node)) {
        if (to == element::i32 || to == element::i64) {
            b->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_generic_ie(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    node->set_output_type(idx, to, node->output(idx).get_partial_shape());
    // return false as we do not replace original node
    return false;
}

bool fuse_type_to_shapeof_v0(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<opset1::ShapeOf>(node)) {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<opset1::ShapeOf>>(*casted,
                element::TypeVector{}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

bool extend_select_type(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node)) {
        type_relaxed->set_origin_input_type(element::boolean, 0);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<opset4::Select>(node)) {
        auto relaxed_op = std::make_shared<op::TypeRelaxed<opset4::Select>>(*casted,
                element::TypeVector{element::boolean},
                element::TypeVector{});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <element::Type_t PREC_FROM, element::Type_t PREC_TO>
std::shared_ptr<Node> change_constant_precision(std::shared_ptr<opset4::Constant> & constant) {
    using src_type = typename element_type_traits<PREC_FROM>::value_type;
    using dst_type = typename element_type_traits<PREC_TO>::value_type;

    const auto * src_data = constant->get_data_ptr<src_type>();
    const auto size = shape_size(constant->get_shape());

    auto new_constant = std::make_shared<ngraph::opset4::Constant>(PREC_TO, constant->get_shape());
    auto * dst_data = const_cast<dst_type *>(reinterpret_cast<const dst_type *>(new_constant->get_data_ptr()));

    std::vector<dst_type> final_data;
    for (size_t i = 0; i < size; ++i) {
        const auto & val = src_data[i];
        if (val > std::numeric_limits<dst_type>::max()) {
            dst_data[i] = std::numeric_limits<dst_type>::max();
        } else {
            dst_data[i] = static_cast<dst_type>(val);
        }
    }
    return new_constant;
}

bool fuse_type_to_constant(std::shared_ptr<Node> & node, element::Type to, const std::vector<Input<Node>> & consumers) {
    if (auto constant = as_type_ptr<opset4::Constant>(node)) {
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
        for (auto & output : consumers) {
            output.replace_source_output(new_const);
        }

        new_const->validate_and_infer_types();
        if (constant->get_output_target_inputs(0).size() == consumers.size()) {
            new_const->set_friendly_name(constant->get_friendly_name());
        }
    }
    return false;
}
