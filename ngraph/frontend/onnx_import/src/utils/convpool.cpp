// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace convpool
        {
            Shape get_kernel_shape(const Node& node)
            {
                const auto& data_shape = node.get_ng_inputs().at(0).get_partial_shape();
                const size_t input_spatial_dims = data_shape.rank().get_length() - 2;
                return node.get_attribute_value<std::vector<size_t>>(
                    "kernel_shape", std::vector<size_t>(input_spatial_dims, 1UL));
            }

            namespace detail
            {
                /// \brief      Gets the attribute default value.
                ///
                /// \param[in]  node       The node we get attribute value from.
                /// \param[in]  attr_name  The attribute name.
                ///
                /// \return     The attribute default value.
                ///
                std::vector<std::size_t> get_attr_default_value(const Node& node,
                                                                const std::string& attr_name)
                {
                    const auto data_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();
                    CHECK_VALID_NODE(node,
                                     data_rank.is_static(),
                                     "If '",
                                     attr_name,
                                     "' is not provided data rank must be static.");
                    const auto data_spatial_dims = data_rank.get_length() - 2;

                    return std::vector<std::size_t>(data_spatial_dims, 1UL);
                }

                ///
                /// \brief      Helper method used to read vector attribute.
                ///
                /// \note       Default value is vector of size spatial dims filled with ones.
                ///
                /// \param[in]  node         Node from which attribute is read
                /// \param[in]  attr_name    Attribute name (such as `strides`, `dilations`)
                /// \param[in]  kernel_rank  The optional kernel rank.
                ///
                /// \return     Read vector attribute if available or default value
                ///
                std::vector<std::size_t> get_attribute_value(const Node& node,
                                                             const std::string& attr_name,
                                                             const std::size_t kernel_rank = 0UL)
                {
                    if (node.has_attribute(attr_name))
                    {
                        return node.get_attribute_value<std::vector<std::size_t>>(attr_name);
                    }
                    else if (kernel_rank != 0)
                    {
                        return std::vector<std::size_t>(kernel_rank, 1UL);
                    }
                    else
                    {
                        return get_attr_default_value(node, attr_name);
                    }
                }
            } // namespace detail

            Strides get_strides(const Node& node, const std::size_t kernel_rank)
            {
                return detail::get_attribute_value(node, "strides", kernel_rank);
            }

            Strides get_dilations(const Node& node, const std::size_t kernel_rank)
            {
                return detail::get_attribute_value(node, "dilations", kernel_rank);
            }

            ngraph::op::RoundingType get_rounding_type(const Node& node)
            {
                return static_cast<ngraph::op::RoundingType>(
                    node.get_attribute_value<std::int64_t>("ceil_mode", 0));
            }

            ngraph::op::PadType get_auto_pad(const Node& node)
            {
                // Default value means use explicitly provided padding values.
                ngraph::op::PadType pad_type{ngraph::op::PadType::NOTSET};
                if (node.has_attribute("auto_pad"))
                {
                    static std::unordered_multimap<std::string, ngraph::op::PadType>
                        auto_pad_values{
                            {"VALID", ngraph::op::PadType::VALID},
                            {"SAME_UPPER", ngraph::op::PadType::SAME_UPPER},
                            {"SAME_LOWER", ngraph::op::PadType::SAME_LOWER},
                            {"NOTSET", ngraph::op::PadType::NOTSET},
                        };

                    const std::string& pad_str{
                        node.get_attribute_value<std::string>("auto_pad", "NOTSET")};
                    const auto pad_val_it = auto_pad_values.find(pad_str);
                    CHECK_VALID_NODE(node,
                                     pad_val_it != auto_pad_values.end(),
                                     "Provided `auto_pad` attribute value: '",
                                     pad_str,
                                     "' is invalid.");
                    pad_type = pad_val_it->second;
                }
                return pad_type;
            }

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const size_t kernel_rank)
            {
                CoordinateDiff pads(kernel_rank, 0);
                if (node.has_attribute("pads"))
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }
                else if (node.has_attribute("paddings"))
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("paddings");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }

                if (pads.size() == kernel_rank * 2)
                {
                    return {{std::begin(pads), std::begin(pads) + pads.size() / 2},
                            {std::begin(pads) + pads.size() / 2, std::end(pads)}};
                }
                else
                {
                    // No paddings provided or only one side values provided, which means same
                    // padding at both begin and end of axis.
                    return {pads, pads};
                }
            }

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node)
            {
                const auto data_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();
                CHECK_VALID_NODE(node,
                                 data_rank.is_static(),
                                 "The rank of node must be static in order to calculate pads");
                const auto data_spatial_dims = data_rank.get_length() - 2;

                return get_pads(node, data_spatial_dims);
            }

            void calculate_auto_pads(const Shape& data_shape,
                                     const Shape& filter_shape,
                                     const Strides& strides,
                                     const Strides& dilations,
                                     const ngraph::op::PadType& pad_type,
                                     CoordinateDiff& padding_below,
                                     CoordinateDiff& padding_above)
            {
                if (pad_type == ngraph::op::PadType::SAME_UPPER ||
                    pad_type == ngraph::op::PadType::SAME_LOWER)
                {
                    padding_below.clear();
                    padding_above.clear();
                    // Extract kernel shape - remove (N,C) channels
                    Shape kernel_shape(std::next(std::begin(filter_shape), 2),
                                       std::end(filter_shape));
                    ngraph::infer_auto_padding(data_shape,
                                               kernel_shape,
                                               strides,
                                               dilations,
                                               pad_type,
                                               padding_above,
                                               padding_below);
                }
            }

            Output<ngraph::Node> get_reshaped_filters(const Output<ngraph::Node>& filters,
                                                      int64_t groups)
            {
                const auto zero_node = default_opset::Constant::create(element::i64, Shape(), {0});
                const auto split_lengths =
                    default_opset::Constant::create(element::i64, Shape{2}, {1, -1});
                const auto groups_node =
                    default_opset::Constant::create(element::i64, Shape{1}, {groups});

                const auto filters_shape = std::make_shared<default_opset::ShapeOf>(filters);
                const auto splitted_shape = std::make_shared<default_opset::VariadicSplit>(
                    filters_shape, zero_node, split_lengths);

                const auto first_dim =
                    std::make_shared<default_opset::Divide>(splitted_shape->output(0), groups_node);
                const auto new_filters_shape = std::make_shared<default_opset::Concat>(
                    OutputVector{groups_node, first_dim, splitted_shape->output(1)}, 0);

                const auto reshaped_filters =
                    std::make_shared<default_opset::Reshape>(filters, new_filters_shape, false);

                return reshaped_filters;
            }
        } // namespace convpool
    }     // namespace onnx_import
} // namespace ngraph
