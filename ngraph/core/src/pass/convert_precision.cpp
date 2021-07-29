// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "ngraph_ops/type_relaxed.hpp"

#include <ngraph/runtime/reference/convert.hpp>

using namespace ngraph;

bool fuse_type_to_constant(const std::shared_ptr<ngraph::Node>& node,
                           ngraph::element::Type to,
                           const std::vector<ngraph::Input<ngraph::Node>>& consumers);
bool fuse_type_to_shapeof(const std::shared_ptr<ngraph::Node>& node,
                          ngraph::element::Type to,
                          size_t idx);
bool fuse_type_to_shapeof_v0(const std::shared_ptr<ngraph::Node>& node,
                             ngraph::element::Type to,
                             size_t idx);
bool fuse_type_to_range_v4(const std::shared_ptr<ngraph::Node>& node,
                           ngraph::element::Type to,
                           size_t idx);
bool fuse_type_to_parameter(const std::shared_ptr<ngraph::Node>& node,
                            ngraph::element::Type to,
                            size_t idx);
bool fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node,
                          ngraph::element::Type to,
                          size_t idx);
bool fuse_type_to_nms3(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx);
bool fuse_type_to_nms4(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx);
bool fuse_type_to_nms5(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx);
bool fuse_type_to_topk(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx);
bool fuse_type_to_nonzero(const std::shared_ptr<ngraph::Node>& node,
                          ngraph::element::Type to,
                          size_t idx);
bool fuse_type_to_bucketize(const std::shared_ptr<ngraph::Node>& node,
                            ngraph::element::Type to,
                            size_t idx);
bool fuse_type_to_ctc_greedy_decoder_seq_len(const std::shared_ptr<ngraph::Node>& node,
                                             ngraph::element::Type to,
                                             size_t idx);

bool extend_select_type(const std::shared_ptr<ngraph::Node>& node,
                        ngraph::element::Type to,
                        size_t idx);

template <typename T>
bool fuse_type_to_binary_comparision(const std::shared_ptr<ngraph::Node>& node,
                                     ngraph::element::Type to,
                                     size_t idx)
{
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node))
    {
        type_relaxed->set_overridden_output_type(to);
        return true;
    }
    else if (auto casted = std::dynamic_pointer_cast<T>(node))
    {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(
            *casted, element::TypeVector{}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <typename T>
bool fuse_type_to_logical(const std::shared_ptr<ngraph::Node>& node,
                          ngraph::element::Type to,
                          size_t idx)
{
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node))
    {
        type_relaxed->set_overridden_output_type(to);
        type_relaxed->set_origin_input_type(element::boolean, 0);
        type_relaxed->set_origin_input_type(element::boolean, 1);
        return true;
    }
    else if (auto casted = std::dynamic_pointer_cast<T>(node))
    {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(
            *casted,
            element::TypeVector{element::boolean, element::boolean},
            element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <class T>
bool fuse_type_to_reduce_logical(const std::shared_ptr<ngraph::Node>& node,
                                 ngraph::element::Type to,
                                 size_t idx)
{
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node))
    {
        type_relaxed->set_overridden_output_type(to);
        type_relaxed->set_origin_input_type(element::boolean, 0);
        return true;
    }
    else if (auto casted = std::dynamic_pointer_cast<T>(node))
    {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<T>>(
            *casted, element::TypeVector{element::boolean}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

namespace
{
    void validate_nodes_and_infer_types(const std::vector<std::shared_ptr<Node>>& ops)
    {
        for (auto& node : ops)
        {
            node->revalidate_and_infer_types();
        }
    }

    bool convert_precision(pass::PassBase& pass,
                           const std::shared_ptr<ngraph::Function>& f,
                           const type_to_fuse_map& type_to_fuse,
                           const type_to_fuse_map& type_to_extend,
                           element::Type from,
                           element::Type to)
    {
        // As Constant operations can be shared between multiple nGraph Functions so before
        // changing precision we need to understand which Constant consumers belongs
        // to the current nGraph Function
        std::unordered_map<const ngraph::Node*, std::vector<Input<Node>>> const_to_internal_output;

        auto register_constants =
            [&const_to_internal_output](const std::vector<std::shared_ptr<Node>>& ops) {
                for (auto& node : ops)
                {
                    for (auto& input : node->inputs())
                    {
                        if (auto const_node = std::dynamic_pointer_cast<opset4::Constant>(
                                input.get_source_output().get_node_shared_ptr()))
                        {
                            const_to_internal_output[const_node.get()].emplace_back(input);
                        }
                    }
                }
            };

        auto convert_node_output_precision = [&](const std::shared_ptr<ngraph::Node>& node) {
            for (auto output : node->outputs())
            {
                if (output.get_element_type() == from)
                {
                    // Handle case with Constants as they can have consumers from other nGraph
                    // Function object
                    auto it = const_to_internal_output.find(node.get());
                    if (it != const_to_internal_output.end())
                    {
                        return fuse_type_to_constant(node, to, it->second);
                    }

                    // Check that node type exists in map and we can fuse type into node
                    auto t2f_it = type_to_fuse.find(node->get_type_info());
                    if (t2f_it != type_to_fuse.end() &&
                        t2f_it->second(node, to, output.get_index()))
                    {
                        // We need to break if original node was replaced
                        return true;
                    }
                }
            }
            return false;
        };

        auto convert_node_input_precision = [&](const std::shared_ptr<ngraph::Node>& node) {
            for (auto input : node->inputs())
            {
                if (input.get_element_type() == from)
                {
                    // For some operations we need to extend their input types to support new type
                    auto it = type_to_extend.find(node->get_type_info());
                    if (it != type_to_extend.end() && it->second(node, to, input.get_index()))
                    {
                        return true;
                    }
                }
            }
            return false;
        };

        std::function<bool(const std::shared_ptr<Function>&, bool)> convert_function_precision =
            [&](const std::shared_ptr<Function>& f, bool is_subgraph) {
                bool is_changed = false;

                auto ops = f->get_ordered_ops();

                // Iterate over all nodes in topological order and then iterate over node outputs.
                // If output type mismatch given type we try to fuse type into this operation
                // otherwise we insert Convert operation.
                for (auto& node : ops)
                {
                    pass.transformation_callback(node);
                    // Recursively apply transformation for sub-graph based operations
                    if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
                    {
                        if (auto sub_graph = sub_graph_node->get_function())
                        {
                            is_changed |= convert_function_precision(sub_graph, true);
                        }
                    }
                    is_changed |= convert_node_input_precision(node);
                }

                if (is_changed)
                    ops = f->get_ordered_ops();

                // Register internal constants only after fixing input type that could lead to nodes
                // replacement
                register_constants(ops);

                bool is_output_precision_changed = false;

                for (auto& node : ops)
                {
                    is_output_precision_changed |= convert_node_output_precision(node);
                }

                if (is_output_precision_changed)
                {
                    ops = f->get_ordered_ops();
                    is_changed |= is_output_precision_changed;
                }

                if (!is_subgraph)
                {
                    if (is_changed)
                        validate_nodes_and_infer_types(ops);

                    // TODO: we need to split NopElimination pass to separate MatcherPasses and call
                    // Convert elimination here
                    for (auto& node : ops)
                    {
                        if (auto convert = std::dynamic_pointer_cast<opset4::Convert>(node))
                        {
                            // WA for topK, dont remove fake convert
                            if (convert->input(0).get_element_type() ==
                                    convert->get_convert_element_type() &&
                                convert->input_value(0).get_node_shared_ptr()->get_output_size() ==
                                    1)
                            {
                                replace_output_update_name(convert->output(0),
                                                           convert->input_value(0));
                            }
                        }
                    }
                }

                return is_changed;
            };

        return convert_function_precision(f, false);
    }

    struct EnumClassHash
    {
        template <class T>
        std::size_t operator()(T t) const
        {
            return static_cast<size_t>(t);
        }
    };

    using precisions_set_t = std::unordered_set<ngraph::element::Type_t, EnumClassHash>;

    precisions_set_t find_all_used_precisions(const std::shared_ptr<ngraph::Function>& fn)
    {
        precisions_set_t used_precisions;

        ngraph::traverse_nodes(fn, [&](std::shared_ptr<ngraph::Node> node) {
            for (auto output : node->outputs())
            {
                used_precisions.emplace(output.get_element_type());
            }
            if (auto sub_graph_node = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(node))
            {
                if (auto sub_graph = sub_graph_node->get_function())
                {
                    auto sub_graph_precisions = find_all_used_precisions(sub_graph);
                    used_precisions.insert(sub_graph_precisions.begin(),
                                           sub_graph_precisions.end());
                }
            }
        });

        return used_precisions;
    }

} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPrecision, "ConvertPrecision", 0);

bool ngraph::pass::ConvertPrecision::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    type_to_fuse_map type_to_fuse{
        {opset4::Parameter::type_info, fuse_type_to_parameter},
        {opset4::Convert::type_info, fuse_type_to_convert},
        {opset4::ShapeOf::type_info, fuse_type_to_shapeof},
        {opset3::NonMaxSuppression::type_info, fuse_type_to_nms3},
        {opset4::NonMaxSuppression::type_info, fuse_type_to_nms4},
        {opset5::NonMaxSuppression::type_info, fuse_type_to_nms5},
        {opset6::CTCGreedyDecoderSeqLen::type_info, fuse_type_to_ctc_greedy_decoder_seq_len},
        {opset4::TopK::type_info, fuse_type_to_topk},
        {opset4::NonZero::type_info, fuse_type_to_nonzero},
        {opset4::Bucketize::type_info, fuse_type_to_bucketize},
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
        {opset4::ReduceLogicalAnd::type_info,
         fuse_type_to_reduce_logical<opset4::ReduceLogicalAnd>},
        {opset4::ReduceLogicalOr::type_info, fuse_type_to_reduce_logical<opset4::ReduceLogicalOr>},
        {opset1::ShapeOf::type_info, fuse_type_to_shapeof_v0},
        {opset4::Range::type_info, fuse_type_to_range_v4}};

    type_to_fuse.insert(m_additional_type_to_fuse_map.begin(), m_additional_type_to_fuse_map.end());

    static type_to_fuse_map type_to_extend{
        {opset4::Select::type_info, extend_select_type},
    };

    bool is_changed = false;

    auto const used_precisions = find_all_used_precisions(f);

    for (auto const& p : m_precisions)
    {
        if (used_precisions.count(p.first))
            is_changed =
                is_changed |
                convert_precision(*this, f, type_to_fuse, type_to_extend, p.first, p.second);
    }

    (void)is_changed; // ignored

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}

bool fuse_type_to_shapeof(const std::shared_ptr<ngraph::Node>& node, element::Type to, size_t idx)
{
    if (auto shapeof = as_type_ptr<opset4::ShapeOf>(node))
    {
        if (to == element::i32 || to == element::i64)
        {
            shapeof->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_range_v4(const std::shared_ptr<ngraph::Node>& node, element::Type to, size_t idx)
{
    if (auto range = as_type_ptr<opset4::Range>(node))
    {
        if (to.is_integral_number() || to.is_real())
        {
            range->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_parameter(const std::shared_ptr<ngraph::Node>& node, element::Type to, size_t idx)
{
    if (auto param = as_type_ptr<opset4::Parameter>(node))
    {
        param->set_element_type(to);
        param->validate_and_infer_types();
        return true;
    }
    return false;
}

bool fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node, element::Type to, size_t idx)
{
    if (auto convert = as_type_ptr<opset4::Convert>(node))
    {
        convert->set_convert_element_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms3(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx)
{
    if (auto nms = as_type_ptr<opset3::NonMaxSuppression>(node))
    {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms4(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx)
{
    if (auto nms = as_type_ptr<opset4::NonMaxSuppression>(node))
    {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms5(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx)
{
    if (auto nms = as_type_ptr<opset5::NonMaxSuppression>(node))
    {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_topk(const std::shared_ptr<ngraph::Node>& node,
                       ngraph::element::Type to,
                       size_t idx)
{
    if (auto topk = as_type_ptr<opset4::TopK>(node))
    {
        if (idx == 1 && (to == element::i32 || to == element::i64))
        {
            topk->set_index_element_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_ctc_greedy_decoder_seq_len(const std::shared_ptr<ngraph::Node>& node,
                                             ngraph::element::Type to,
                                             size_t idx)
{
    if (auto ctc_decoder = as_type_ptr<opset6::CTCGreedyDecoderSeqLen>(node))
    {
        if (idx == 0 && (to == element::i32 || to == element::i64))
        {
            ctc_decoder->set_classes_index_type(to);
            return true;
        }
        if (idx == 1 && (to == element::i32 || to == element::i64))
        {
            ctc_decoder->set_sequence_length_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_nonzero(const std::shared_ptr<ngraph::Node>& node,
                          ngraph::element::Type to,
                          size_t idx)
{
    if (auto nonzero = as_type_ptr<opset4::NonZero>(node))
    {
        if (to == element::i32 || to == element::i64)
        {
            nonzero->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_bucketize(const std::shared_ptr<ngraph::Node>& node,
                            ngraph::element::Type to,
                            size_t idx)
{
    if (auto b = as_type_ptr<opset4::Bucketize>(node))
    {
        if (to == element::i32 || to == element::i64)
        {
            b->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_shapeof_v0(const std::shared_ptr<ngraph::Node>& node,
                             ngraph::element::Type to,
                             size_t idx)
{
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node))
    {
        type_relaxed->set_overridden_output_type(to);
        return true;
    }
    else if (auto casted = std::dynamic_pointer_cast<opset1::ShapeOf>(node))
    {
        auto relaxed_op = std::make_shared<ngraph::op::TypeRelaxed<opset1::ShapeOf>>(
            *casted, element::TypeVector{}, element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

bool extend_select_type(const std::shared_ptr<ngraph::Node>& node,
                        ngraph::element::Type to,
                        size_t idx)
{
    if (auto type_relaxed = std::dynamic_pointer_cast<op::TypeRelaxedBase>(node))
    {
        type_relaxed->set_origin_input_type(element::boolean, 0);
        return true;
    }
    else if (auto casted = std::dynamic_pointer_cast<opset4::Select>(node))
    {
        auto relaxed_op = std::make_shared<op::TypeRelaxed<opset4::Select>>(
            *casted, element::TypeVector{element::boolean}, element::TypeVector{});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <typename src_type, typename dst_type>
inline dst_type convert_value(src_type val)
{
    if (val > std::numeric_limits<dst_type>::max())
    {
        return std::numeric_limits<dst_type>::max();
    }
    else if (val < std::numeric_limits<dst_type>::lowest())
    {
        return std::numeric_limits<dst_type>::lowest();
    }
    return static_cast<dst_type>(val);
}

// We need to treat U64->I32 and U32->I32 as a separate case, because of C++'s implicit promotion
// from signed to unsigned, and we don't need to compare and clamp the input to
// std::numeric_limits<int32_t>::lowest()
template <>
inline int32_t convert_value<uint64_t, int32_t>(uint64_t val)
{
    if (val > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(val);
}

template <>
inline int32_t convert_value<uint32_t, int32_t>(uint32_t val)
{
    if (val > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
    {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(val);
}

namespace
{
    template <element::Type_t PREC_FROM, element::Type_t PREC_TO>
    std::shared_ptr<ngraph::Node>
        change_constant_precision(std::shared_ptr<opset4::Constant>& constant)
    {
        using src_type = typename element_type_traits<PREC_FROM>::value_type;
        using dst_type = typename element_type_traits<PREC_TO>::value_type;

        const auto* src_data = constant->get_data_ptr<src_type>();
        const auto size = shape_size(constant->get_shape());

        auto new_constant =
            std::make_shared<ngraph::opset4::Constant>(PREC_TO, constant->get_shape());
        auto* dst_data =
            const_cast<dst_type*>(reinterpret_cast<const dst_type*>(new_constant->get_data_ptr()));
        if (dst_data == nullptr)
            throw ngraph_error("Can't get destination data pointer");

        for (size_t i = 0; i < size; ++i)
        {
            dst_data[i] = convert_value<src_type, dst_type>(src_data[i]);
        }
        return new_constant;
    }

    template <>
    std::shared_ptr<Node> change_constant_precision<element::Type_t::f16, element::Type_t::f32>(
        std::shared_ptr<opset4::Constant>& constant)
    {
        using src_type = typename element_type_traits<element::Type_t::f16>::value_type;
        using dst_type = typename element_type_traits<element::Type_t::f32>::value_type;

        const auto* src_data = constant->get_data_ptr<src_type>();
        const auto size = shape_size(constant->get_shape());

        auto new_constant =
            std::make_shared<ngraph::opset4::Constant>(element::Type_t::f32, constant->get_shape());
        auto* dst_data =
            const_cast<dst_type*>(reinterpret_cast<const dst_type*>(new_constant->get_data_ptr()));
        if (dst_data == nullptr)
            throw ngraph_error("Can't get destination data pointer");

        ngraph::runtime::reference::convert<src_type, dst_type>(src_data, dst_data, size);

        return new_constant;
    }

    /**
     * @brief Method converts low precision integer types
     * The method uses the next logic for conversion:
     *  * For unsigned types we just copy all bits to destination type (which is bigger):
     *    int4 [1011] -> int8 [00001011]
     *  * For signed types we copy all bits (except sign bit) to destination type and after
     *    that for negative values we set to 1 all higher bits:
     *    int4 [1011] -> int8 [11111011]
     *
     * @param src source value      !!! the type must be unsigned !!!
     * @param dst destination value !!! the type must be unsigned !!!
     * @param src_offset source offset (for custom data types)
     * @param src_size source size (for custom data types)
     * @param dst_offset destination offset
     * @param dst_size destination size
     * @param is_signed the type of source data
     */
    template <class SRC, class DST>
    void convert_lp_value(const SRC& src,
                          DST& dst,
                          size_t src_offset,
                          size_t src_size,
                          size_t dst_offset,
                          size_t dst_size,
                          bool is_signed)
    {
        constexpr SRC src_max = std::numeric_limits<SRC>::max();
        constexpr DST dst_max = std::numeric_limits<DST>::max();
        // Make a shift for the source value
        // src [11101000] offset 2, size 4
        // val [00011101]
        SRC val = src >> src_offset;
        // dst     [10001111 00000100] offset 5 size 9
        // new_val [00000000 00000000]
        DST new_val = 0;

        // Calculate diff in order to clean bits which don't exist in the source value
        // diff 4
        size_t diff = sizeof(SRC) * 8 - src_size;
        // Clean unnecessary bits
        // val [11010000]
        val = val << diff;
        // val [00001101]
        val = val >> diff;

        // Get the sign of value
        // sign [00000001]
        SRC sign = (val >> (src_size - 1)) & 0b1;

        // If source type is signed and negative
        if (is_signed && sign)
        {
            // val [11111101]
            val |= src_max << diff;
            // new_val [00000001 11111111]
            new_val = dst_max >> (sizeof(DST) * 8 - dst_size);
            // new_val [00000001 11111101]
            new_val &= (dst_max << sizeof(SRC) * 8) | val;
        }
        else
        {
            // new_val [00000000 00001101]
            new_val = val;
        }

        // Make a mask in order to save other values if DST contains several values
        // mask [11000000 00011111]
        DST mask = 0;
        if (dst_offset + dst_size < sizeof(DST) * 8)
            mask = (dst_max << (dst_offset + dst_size));
        if (dst_offset != 0)
            mask |= (dst_max >> (sizeof(DST) * 8 - dst_offset));

        // Add mask to our converted value
        // signed:   new_val [11100000 10111111]
        // unsigned: new_val [11000001 10111111]
        new_val = mask | (new_val << dst_offset);

        // Add our value to destination
        // dst: [10111111 11100100]
        dst |= ~mask;
        // signed:   dst [10100000 10100100]
        // unsigned: dst [10000001 10100100]
        dst &= new_val;
    }

    std::shared_ptr<Node> convert_low_precisions_int(std::shared_ptr<opset4::Constant>& constant,
                                                     element::Type to)
    {
        // Supported integer precisions
        static const precisions_set_t supported_integer_precisions = {
            element::i4, element::u4, element::u1};
        // Get source element type and source data
        auto src_type = constant->get_element_type();
        const auto* src_data = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());

        // We support conversion only if several elements can be represented in one instance of some
        // C++ common data type without any exception, destination data type should be bigger than
        // source and destination data type should be real
        if (!supported_integer_precisions.count(src_type) ||
            (src_type.size() * 8) % src_type.bitwidth() || (to.size() * 8) % to.bitwidth() ||
            to.is_real() || to.bitwidth() < src_type.bitwidth())
            throw ngraph_error("Convert low precision for " +
                               constant->get_element_type().get_type_name() + " to " +
                               to.get_type_name() + " is not implemented!");

        // Create a new constant operation and get destination data
        auto new_constant = std::make_shared<ngraph::opset4::Constant>(to, constant->get_shape());
        auto* dst_data =
            const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_constant->get_data_ptr()));
        // Check pointers
        if (src_data == nullptr || dst_data == nullptr)
            throw ngraph_error("Can't get data pointer");

        // Convert values
        const auto size = shape_size(constant->get_shape());
        size_t src_idx(0), dst_idx(0), dst_off(0), src_off(0);
        if (src_type.bitwidth() < 8)
        {
            src_off = 8 - src_type.bitwidth();
        }

        if (to.bitwidth() < 8)
        {
            dst_off = 8 - to.bitwidth();
        }

        for (size_t i = 0; i < size; i++)
        {
            // Source type at the current moment always less than 1 byte
            // Select the right destination type
            switch (to.size())
            {
            case 1:
                convert_lp_value<uint8_t, uint8_t>(src_data[src_idx],
                                                   dst_data[dst_idx],
                                                   src_off,
                                                   src_type.bitwidth(),
                                                   dst_off,
                                                   to.bitwidth(),
                                                   src_type.is_signed());
                break;
            case 2:
                convert_lp_value<uint8_t, uint16_t>(src_data[src_idx],
                                                    reinterpret_cast<uint16_t*>(dst_data)[dst_idx],
                                                    src_off,
                                                    src_type.bitwidth(),
                                                    dst_off,
                                                    to.bitwidth(),
                                                    src_type.is_signed());
                break;
            case 4:
                convert_lp_value<uint8_t, uint32_t>(src_data[src_idx],
                                                    reinterpret_cast<uint32_t*>(dst_data)[dst_idx],
                                                    src_off,
                                                    src_type.bitwidth(),
                                                    dst_off,
                                                    to.bitwidth(),
                                                    src_type.is_signed());
                break;
            case 8:
                convert_lp_value<uint8_t, uint64_t>(src_data[src_idx],
                                                    reinterpret_cast<uint64_t*>(dst_data)[dst_idx],
                                                    src_off,
                                                    src_type.bitwidth(),
                                                    dst_off,
                                                    to.bitwidth(),
                                                    src_type.is_signed());
                break;
            default: throw ngraph_error("Unsupported element size!");
            }
            // Calculate offsets and indexes
            if (src_type.bitwidth() < 8)
            {
                if (src_off == 0)
                {
                    src_off = 8;
                    src_idx++;
                }
                src_off -= src_type.bitwidth();
            }
            else
            {
                src_idx++;
            }
            if (to.bitwidth() < 8)
            {
                if (dst_off == 0)
                {
                    dst_off = 8;
                    dst_idx++;
                }
                dst_off -= to.bitwidth();
            }
            else
            {
                dst_idx++;
            }
        }

        return new_constant;
    }

} // namespace

bool fuse_type_to_constant(const std::shared_ptr<ngraph::Node>& node,
                           element::Type to,
                           const std::vector<Input<Node>>& consumers)
{
    if (auto constant = as_type_ptr<opset4::Constant>(node))
    {
        auto from = constant->get_element_type();
        std::shared_ptr<ngraph::Node> new_const;
        if (from == element::u64 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::u64, element::Type_t::i32>(constant);
        }
        else if (from == element::i64 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::i64, element::Type_t::i32>(constant);
        }
        else if (from == element::u8 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::u8, element::Type_t::i32>(constant);
        }
        else if (from == element::u16 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::u16, element::Type_t::i32>(constant);
        }
        else if (from == element::i16 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::i16, element::Type_t::i32>(constant);
        }
        else if (from == element::u32 && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::u32, element::Type_t::i32>(constant);
        }
        else if (from == element::f64 && to == element::f32)
        {
            new_const =
                change_constant_precision<element::Type_t::f64, element::Type_t::f32>(constant);
        }
        else if (from == element::bf16 && to == element::f32)
        {
            new_const =
                change_constant_precision<element::Type_t::bf16, element::Type_t::f32>(constant);
        }
        else if (from == element::f32 && to == element::f16)
        {
            new_const =
                change_constant_precision<element::Type_t::f32, element::Type_t::f16>(constant);
        }
        else if (from == element::f16 && to == element::f32)
        {
            new_const =
                change_constant_precision<element::Type_t::f16, element::Type_t::f32>(constant);
        }
        else if (from == element::boolean && to == element::u8)
        {
            new_const =
                change_constant_precision<element::Type_t::boolean, element::Type_t::u8>(constant);
        }
        else if (from == element::boolean && to == element::i32)
        {
            new_const =
                change_constant_precision<element::Type_t::boolean, element::Type_t::i32>(constant);
        }
        else if (from == element::i4 || from == element::u4 || from == element::u1)
        {
            new_const = convert_low_precisions_int(constant, to);
        }
        else
        {
            throw ngraph_error("Precision conversion from " + from.get_type_name() + " to " +
                               to.get_type_name() + " is not supported");
        }
        for (auto& output : consumers)
        {
            output.replace_source_output(new_const);
        }

        new_const->validate_and_infer_types();
        new_const->set_friendly_name(constant->get_friendly_name());
        return true;
    }
    return false;
}
