// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/autobroadcast.hpp"

#include <memory>
#include <numeric>
#include <sstream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/check.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/util.hpp"

using namespace std;

namespace ngraph {
namespace builder {
OPENVINO_SUPPRESS_DEPRECATED_START
numpy_autobroadcast_incompatible_shapes::numpy_autobroadcast_incompatible_shapes(const Shape& shape1,
                                                                                 const Shape& shape2)
    : ngraph_error(error_str(shape1, shape2)),
      m_shape1(shape1),
      m_shape2(shape2) {}

string numpy_autobroadcast_incompatible_shapes::error_str(const Shape& shape1, const Shape& shape2) {
    ostringstream os;
    os << "Auto-broadcast not possible for these input shapes:"
       << " shape1=" << vector_to_string(shape1) << " shape2=" << vector_to_string(shape2);
    return os.str();
}
OPENVINO_SUPPRESS_DEPRECATED_END

///
/// \brief      Calculate the output shape of numpy-style broadcast operation for two
///             shapes.
///
/// \note       More info:
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
///             Example: left: [3, 1, 10] right: [5, 1] return: [3, 5, 10]
///
/// \param      lhs_shape  First input shape.
/// \param      rhs_shape  Second input Shape.
///
/// \return     Broadcast shape of input shapes.
///
static Shape calculate_broadcast_shape(Shape lhs_shape, Shape rhs_shape) {
    Shape result;
    auto lhs_rank = lhs_shape.size();
    auto rhs_rank = rhs_shape.size();
    auto max_rank = max(lhs_rank, rhs_rank);

    // left-pad the lhs_shape with ones
    lhs_shape.insert(begin(lhs_shape), max_rank - lhs_rank, 1);
    // left-pad the rhs_shape with ones
    rhs_shape.insert(begin(rhs_shape), max_rank - rhs_rank, 1);

    for (size_t index = 0; index < max_rank; ++index) {
        size_t lhs_dim = lhs_shape.at(index);
        size_t rhs_dim = rhs_shape.at(index);

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw numpy_autobroadcast_incompatible_shapes(lhs_shape, rhs_shape);
        }

        result.push_back(max(lhs_dim, rhs_dim));
    }

    return result;
};

pair<Shape, vector<Shape>> get_numpy_broadcast_shapes(const vector<Shape>& input_shapes) {
    Shape target_shape = accumulate(begin(input_shapes), end(input_shapes), Shape{}, calculate_broadcast_shape);

    vector<Shape> full_shapes;
    for (const Shape& input : input_shapes) {
        Shape padded_shape{input};
        padded_shape.insert(begin(padded_shape), target_shape.size() - padded_shape.size(), 1);
        full_shapes.push_back(std::move(padded_shape));
    }

    return {target_shape, full_shapes};
}

static pair<Shape, vector<Shape>> get_numpy_broadcast_shapes(const OutputVector& values) {
    vector<Shape> input_shapes;

    for (const auto& input : values) {
        input_shapes.push_back(input.get_partial_shape().to_shape());
    }

    return get_numpy_broadcast_shapes(input_shapes);
}

/// \brief      Broadcast input node.
///
/// \note       The source shape does not have to be the actual shape of input node. However
///             it should be a superset of it (containing it as a continuous subset). This
///             implies we may expand the number of axes of input node. The ranks of
///             source_shape and output_shape must be equal. This means that the
///             source_shape has to be padded with ones for this operation.
///
/// \param[in]  value         The input Node to be broadcast.
/// \param[in]  output_shape  The output shape.
/// \param[in]  source_shape  The source shape from which we want to broadcast input node.
///
/// \return     The broadcasted Node.
///
static shared_ptr<Node> numpy_broadcast_node(const Output<Node>& value,
                                             const Shape& output_shape,
                                             const Shape& source_shape) {
    shared_ptr<Node> broadcasted_node = value.get_node_shared_ptr();
    // If node already has the required shape, return original node
    if (output_shape == value.get_partial_shape().to_shape()) {
        return broadcasted_node;
    }

    NGRAPH_CHECK(source_shape.size() == output_shape.size(),
                 "Ranks of source_shape and output_shape dont match: ",
                 source_shape.size(),
                 " vs ",
                 output_shape.size());

    AxisVector broadcast_axes;
    Shape squeezed_shape;
    // Positions of axes which have length of 1 are needed to calculate broadcast_axes
    // for nGraph broadcast operation. We need to remove ones from source shape
    // to avoid broadcasting axis conflict.
    for (size_t index = 0; index < output_shape.size(); ++index) {
        if (source_shape.at(index) == 1 && output_shape.at(index) != 1) {
            broadcast_axes.push_back(index);
        } else {
            squeezed_shape.push_back(source_shape.at(index));
        }
    }

    if (squeezed_shape != value.get_partial_shape().to_shape()) {
        broadcasted_node = builder::opset1::reshape(value, squeezed_shape);
    }

    if (!broadcast_axes.empty()) {
        auto shape_const = op::Constant::create(element::u64, Shape{output_shape.size()}, output_shape);
        broadcasted_node =
            make_shared<op::v1::Broadcast>(broadcasted_node,
                                           shape_const,
                                           opset1::get_axes_mapping_output(output_shape, broadcast_axes));
    }

    return broadcasted_node;
}

/// \brief      Broadcast input node.
///
/// \param[in]  value         The input Node to be broadcast.
/// \param[in]  output_shape  The output shape.
/// \param[in]  axis          The start index to align with output_shape
///
/// \return     The broadcasted Node.
///
static shared_ptr<Node> broadcast_value_pdpd_style(const Output<Node>& value, const Shape& output_shape, int64_t axis) {
    auto value_shape = value.get_partial_shape().to_shape();

    // If node already has the required shape, return original node
    if (output_shape == value_shape) {
        return value.get_node_shared_ptr();
    }

    if (axis == -1) {
        axis = output_shape.size() - value_shape.size();
    }

    auto trimmed_value_shape = value_shape;
    while (trimmed_value_shape.size() > 0 && trimmed_value_shape.back() == 1) {
        trimmed_value_shape.pop_back();
    }

    AxisSet axes;
    for (int64_t i = 0; i < axis; ++i) {
        axes.insert(static_cast<size_t>(i));
    }

    for (size_t i = axis + trimmed_value_shape.size(); i < output_shape.size(); ++i) {
        axes.insert(i);
    }

    auto trimmed_value = value;
    if (value_shape != trimmed_value_shape) {
        trimmed_value = builder::opset1::reshape(value, trimmed_value_shape);
    }

    auto shape_const = op::Constant::create(element::u64, Shape{output_shape.size()}, output_shape);
    auto value_bcast =
        make_shared<op::v1::Broadcast>(trimmed_value, shape_const, opset1::get_axes_mapping_output(output_shape, axes));

    return std::move(value_bcast);
}

pair<shared_ptr<Node>, shared_ptr<Node>> numpy_broadcast(const pair<Output<Node>, Output<Node>>& args) {
    NGRAPH_CHECK(args.first.get_node());
    NGRAPH_CHECK(args.second.get_node());

    const Shape& arg1_in_shape = args.first.get_partial_shape().to_shape();
    const Shape& arg2_in_shape = args.second.get_partial_shape().to_shape();

    // Handle the trivial case...
    if (arg1_in_shape == arg2_in_shape) {
        return make_pair(args.first.get_node_shared_ptr(), args.second.get_node_shared_ptr());
    }

    NodeVector bcasted_outputs = as_node_vector(numpy_broadcast_outputs({args.first, args.second}));

    return make_pair(bcasted_outputs.at(0), bcasted_outputs.at(1));
}

OutputVector numpy_broadcast_outputs(const OutputVector& values) {
    if (values.size() <= 1) {
        return values;
    }

    // find the output tensor's shape, then broadcast all inputs so that they are compatible
    auto bcast_shapes = get_numpy_broadcast_shapes(values);

    OutputVector broadcasted_inputs;
    for (size_t i = 0; i < values.size(); ++i) {
        broadcasted_inputs.push_back(numpy_broadcast_node(values[i], bcast_shapes.first, bcast_shapes.second[i]));
    }
    return broadcasted_inputs;
}

shared_ptr<Node> numpy_broadcast(const Output<Node>& value, const Shape& shape) {
    auto bcast_shape = get_numpy_broadcast_shapes({value.get_partial_shape().to_shape(), shape});
    return numpy_broadcast_node(value, bcast_shape.first, bcast_shape.second[0]);
}

OutputVector numpy_broadcast_for_matmul_operation(const Output<Node>& left, const Output<Node>& right) {
    const auto& left_shape = left.get_partial_shape().to_shape();
    const auto& right_shape = right.get_partial_shape().to_shape();
    // Broadcast only _stack of matrices_ axes.
    const auto& numpy_shapes = get_numpy_broadcast_shapes(
        {Shape{begin(left_shape), next(end(left_shape), -2)}, Shape{begin(right_shape), next(end(right_shape), -2)}});

    // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
    auto left_output_shape = numpy_shapes.first;
    auto right_output_shape = numpy_shapes.first;
    // Append the last two axes original dimensions.
    left_output_shape.insert(end(left_output_shape), next(begin(left_shape), left_shape.size() - 2), end(left_shape));
    right_output_shape.insert(end(right_output_shape),
                              next(begin(right_shape), right_shape.size() - 2),
                              end(right_shape));

    auto left_full_shape = numpy_shapes.second.at(0);
    auto right_full_shape = numpy_shapes.second.at(1);
    // Append the last two axes original dimensions.
    left_full_shape.insert(end(left_full_shape), next(begin(left_shape), left_shape.size() - 2), end(left_shape));
    right_full_shape.insert(end(right_full_shape), next(begin(right_shape), right_shape.size() - 2), end(right_shape));

    return {numpy_broadcast_node(left, left_output_shape, left_full_shape),
            numpy_broadcast_node(right, right_output_shape, right_full_shape)};
}

OutputVector pdpd_broadcast(const OutputVector& inputs, int64_t axis) {
    if (inputs.size() <= 1) {
        return inputs;
    }

    OutputVector broadcasted_inputs{inputs[0]};
    for (size_t i = 1; i < inputs.size(); ++i) {
        broadcasted_inputs.push_back(
            broadcast_value_pdpd_style(inputs[i], inputs[0].get_partial_shape().to_shape(), axis));
    }
    return broadcasted_inputs;
}

std::shared_ptr<Node> calculate_broadcast_axes(const Shape& output_shape,
                                               const Shape& input_shape,
                                               size_t start_match_axis) {
    vector<size_t> axes(output_shape.size() - input_shape.size());
    // Populate the axes vector with monotonic increasing series from 0 until
    // output_shape_size, excluding values in range:
    // [start_match_axis, start_match_axis + input_shape.size()]
    iota(begin(axes), begin(axes) + start_match_axis, 0);
    iota(begin(axes) + start_match_axis, end(axes), start_match_axis + input_shape.size());

    auto axes_mapping = opset1::get_axes_mapping(output_shape, axes);
    return op::Constant::create(element::i64, Shape{axes_mapping.size()}, axes_mapping);
}

namespace opset1 {
Output<Node> legacy_broadcast_for_binary_operation(const Output<Node>& left,
                                                   const Output<Node>& right,
                                                   size_t start_match_axis) {
    const auto& left_shape = left.get_partial_shape().to_shape();
    const auto& right_shape = right.get_partial_shape().to_shape();

    bool dimensions_identical = (left_shape == right_shape);
    if (dimensions_identical) {
        return right;
    }

    // Prepare new shape of right operand for broadcasting
    // Remove dimensions with length=1 from back
    auto new_right_shape = right_shape;
    for (int dimension = static_cast<int>(new_right_shape.size()) - 1; dimension >= 0; --dimension) {
        if (new_right_shape.at(dimension) == 1) {
            new_right_shape.pop_back();
        } else {
            break;
        }
    }

    // Find first dimensions at front with length different from 1
    size_t num_ones = 0;
    for (size_t dimension : new_right_shape) {
        if (dimension == 1) {
            ++num_ones;
        } else {
            break;
        }
    }

    // Remove dimensions with length=1 from front
    new_right_shape.erase(begin(new_right_shape), next(begin(new_right_shape), num_ones));

    auto reshape_right = reshape(right, new_right_shape);

    // Move broadcast start axis parameter to right
    start_match_axis += num_ones;

    return make_broadcast(reshape_right, left_shape, start_match_axis);
}

vector<size_t> get_axes_mapping(const Shape& output_shape, const AxisSet& broadcast_axes) {
    NGRAPH_CHECK((broadcast_axes.size() <= output_shape.size()));
    vector<size_t> axes_mapping(output_shape.size());
    iota(axes_mapping.begin(), axes_mapping.end(), 0);
    for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); ++i) {
        axes_mapping.erase(axes_mapping.begin() + *i);
    }
    return axes_mapping;
}

Output<Node> get_axes_mapping_output(const PartialShape& output_shape,
                                     const PartialShape& input_shape,
                                     std::size_t start_match_axis) {
    NGRAPH_CHECK((input_shape.rank().is_static() && output_shape.rank().is_static()),
                 "Tensor's rank has to be static.");
    NGRAPH_CHECK(
        (input_shape.rank().get_length() + static_cast<int64_t>(start_match_axis) <= output_shape.rank().get_length()),
        "Unable to figure out axes mapping.");

    vector<int64_t> mapping(input_shape.rank().get_length());
    iota(begin(mapping), end(mapping), start_match_axis);

    return op::Constant::create(element::i64, Shape{mapping.size()}, mapping);
}

Output<Node> get_axes_mapping_output(const Shape& output_shape, const AxisSet& broadcast_axes) {
    vector<size_t> axes_mapping{get_axes_mapping(output_shape, broadcast_axes)};
    return op::Constant::create(element::i64, Shape{axes_mapping.size()}, axes_mapping);
}

static Output<Node> get_axes_mapping_output(const PartialShape& output_shape,
                                            const Output<Node>& input_shape,
                                            std::size_t start_match_axis) {
    const auto one_node = opset7::Constant::create(element::i64, Shape{}, {1});
    const auto zero_node = opset7::Constant::create(element::i64, Shape{}, {0});
    const auto start_match_axis_node = opset7::Constant::create(element::i64, Shape{}, {start_match_axis});
    const auto target_shape_rank_node =
        builder::opset1::reshape(std::make_shared<opset7::ShapeOf>(input_shape), Shape{});

    const auto range_node = std::make_shared<opset7::Range>(zero_node, target_shape_rank_node, one_node, element::i64);

    // workaround for GPU plugin type incompatibility
    const auto range_node_converted =
        std::make_shared<opset7::Convert>(range_node, start_match_axis_node->get_element_type());
    // end of workaround

    const auto result = std::make_shared<opset7::Add>(range_node_converted, start_match_axis_node);
    return result;
}

Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, const AxisSet& broadcast_axes) {
    return make_shared<op::v1::Broadcast>(node,
                                          op::Constant::create(element::i64, Shape{target_shape.size()}, target_shape),
                                          get_axes_mapping_output(target_shape, broadcast_axes));
}

Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, size_t start_match_axis) {
    const auto node_shape = std::make_shared<opset7::ShapeOf>(node);
    return make_shared<op::v1::Broadcast>(node,
                                          op::Constant::create(element::i64, Shape{target_shape.size()}, target_shape),
                                          get_axes_mapping_output(target_shape, node_shape, start_match_axis));
}

}  // namespace opset1
}  // namespace builder
}  // namespace ngraph
