// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/convpool.hpp"

#include <unordered_map>

#include "exceptions.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace ov;
using namespace ov::op;
using ov::CoordinateDiff;

namespace ov {
namespace frontend {
namespace onnx {
namespace convpool {
ov::Shape get_kernel_shape(const Node& node) {
    const auto& data_shape = node.get_ov_inputs().at(0).get_partial_shape();
    const size_t input_spatial_dims = data_shape.rank().get_length() - 2;
    return node.get_attribute_value<std::vector<size_t>>("kernel_shape", std::vector<size_t>(input_spatial_dims, 1UL));
}

namespace {
/// \brief      Gets the attribute default value.
///
/// \param[in]  node       The node we get attribute value from.
/// \param[in]  attr_name  The attribute name.
///
/// \return     The attribute default value.
///
std::vector<std::size_t> get_attr_default_value(const Node& node, const std::string& attr_name) {
    const auto data_rank = node.get_ov_inputs().at(0).get_partial_shape().rank();
    CHECK_VALID_NODE(node, data_rank.is_static(), "If '", attr_name, "' is not provided data rank must be static.");
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
                                             const std::size_t kernel_rank = 0UL) {
    if (node.has_attribute(attr_name)) {
        return node.get_attribute_value<std::vector<std::size_t>>(attr_name);
    } else if (kernel_rank != 0) {
        return std::vector<std::size_t>(kernel_rank, 1UL);
    } else {
        return get_attr_default_value(node, attr_name);
    }
}
}  // namespace

ov::Strides get_strides(const Node& node, const std::size_t kernel_rank) {
    return get_attribute_value(node, "strides", kernel_rank);
}

ov::Strides get_dilations(const Node& node, const std::size_t kernel_rank) {
    return get_attribute_value(node, "dilations", kernel_rank);
}

ov::op::RoundingType get_rounding_type(const Node& node) {
    return static_cast<ov::op::RoundingType>(node.get_attribute_value<std::int64_t>("ceil_mode", 0));
}

ov::op::PadType get_auto_pad(const Node& node) {
    // Default value means use explicitly provided padding values.
    ov::op::PadType pad_type{ov::op::PadType::NOTSET};
    if (node.has_attribute("auto_pad")) {
        static std::unordered_multimap<std::string, ov::op::PadType> auto_pad_values{
            {"VALID", ov::op::PadType::VALID},
            {"SAME_UPPER", ov::op::PadType::SAME_UPPER},
            {"SAME_LOWER", ov::op::PadType::SAME_LOWER},
            {"NOTSET", ov::op::PadType::NOTSET},
            {"", ov::op::PadType::NOTSET},  // empty string considered as undefined attribute
        };

        const std::string& pad_str{node.get_attribute_value<std::string>("auto_pad", "NOTSET")};
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

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node, const size_t kernel_rank) {
    CoordinateDiff pads(kernel_rank, 0);
    if (node.has_attribute("pads")) {
        auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
        pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
    } else if (node.has_attribute("paddings")) {
        auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("paddings");
        pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
    }

    if (pads.size() == kernel_rank * 2) {
        return {{std::begin(pads), std::begin(pads) + pads.size() / 2},
                {std::begin(pads) + pads.size() / 2, std::end(pads)}};
    } else {
        // No paddings provided or only one side values provided, which means same
        // padding at both begin and end of axis.
        return {pads, pads};
    }
}

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node) {
    const auto data_rank = node.get_ov_inputs().at(0).get_partial_shape().rank();
    CHECK_VALID_NODE(node, data_rank.is_static(), "The rank of node must be static in order to calculate pads");
    const auto data_spatial_dims = data_rank.get_length() - 2;

    return get_pads(node, data_spatial_dims);
}

void calculate_auto_pads(const ov::Shape& data_shape,
                         const ov::Shape& filter_shape,
                         const ov::Strides& strides,
                         const ov::Strides& dilations,
                         const ov::op::PadType& pad_type,
                         CoordinateDiff& padding_below,
                         CoordinateDiff& padding_above) {
    if (pad_type == ov::op::PadType::SAME_UPPER || pad_type == ov::op::PadType::SAME_LOWER) {
        const auto num_spatial = strides.size();
        padding_below.resize(num_spatial);
        padding_above.resize(num_spatial);
        auto data_dim = data_shape.cend() - num_spatial;
        auto filter_dim = filter_shape.cend() - num_spatial;

        const auto padding_swap = pad_type == ov::op::PadType::SAME_UPPER;
        auto&& pad_b = padding_swap ? padding_below.begin() : padding_above.begin();
        auto&& pad_e = padding_swap ? padding_above.begin() : padding_below.begin();

        for (size_t i = 0; i < num_spatial; ++i, ++pad_b, ++pad_e, ++data_dim, ++filter_dim) {
            int64_t filter_size = (static_cast<int64_t>(*filter_dim - 1)) * dilations[i] + 1;
            auto filter_stride = static_cast<int64_t>(strides[i]);
            auto output_size = (*data_dim + filter_stride - 1) / filter_stride;

            auto padding_needed = std::max<int64_t>(0, (output_size - 1) * filter_stride + filter_size - *data_dim);
            *pad_b = padding_needed / 2;
            *pad_e = padding_needed - *pad_b;
        }
    }
}

Output<ov::Node> get_reshaped_filters(const Output<ov::Node>& filters, int64_t groups) {
    const auto zero_node = v0::Constant::create(ov::element::i64, ov::Shape(), {0});
    const auto split_lengths = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, -1});
    const auto groups_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {groups});

    const auto filters_shape = std::make_shared<v3::ShapeOf>(filters);
    const auto splitted_shape = std::make_shared<v1::VariadicSplit>(filters_shape, zero_node, split_lengths);

    const auto first_dim = std::make_shared<v1::Divide>(splitted_shape->output(0), groups_node);
    const auto new_filters_shape =
        std::make_shared<v0::Concat>(ov::OutputVector{groups_node, first_dim, splitted_shape->output(1)}, 0);

    const auto reshaped_filters = std::make_shared<v1::Reshape>(filters, new_filters_shape, false);

    return reshaped_filters;
}
}  // namespace convpool
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
