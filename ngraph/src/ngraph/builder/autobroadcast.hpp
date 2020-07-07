//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <utility>

#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/broadcast.hpp"

namespace ngraph
{
    namespace builder
    {
        class NGRAPH_API numpy_autobroadcast_incompatible_shapes : public ngraph::ngraph_error
        {
        public:
            numpy_autobroadcast_incompatible_shapes(const ngraph::Shape& shape1,
                                                    const ngraph::Shape& shape2);

        private:
            const ngraph::Shape m_shape1;
            const ngraph::Shape m_shape2;

            static std::string error_str(const ngraph::Shape& shape1, const ngraph::Shape& shape2);
        };

        ///
        /// \brief      Broadcast all values, if necessary, to obtain equal shapes according
        ///             to NumPy's auto-broadcasting scheme.
        ///
        /// \note       There are some shape combinations which the autobroadcast algoritm cannot
        ///             handle. An exception is thrown when such combinations are provided to this
        ///             function.
        ///
        /// \param      values  Vector of output values.
        ///
        /// \exception  ngraph::builder::numpy_autobroadcast_incompatible_shapes
        ///
        /// \return     Vector of broadcasted values.
        ///
        NGRAPH_API
        OutputVector numpy_broadcast_outputs(const OutputVector& values);

        ///
        /// \brief      Broadcast input value to provided shape using NumPy's auto-broadcasting
        ///             rules.
        ///
        /// \param      value  Input value
        /// \param      shape  Requested output shape
        ///
        /// \return     Node producing values with requested shape.
        ///
        NGRAPH_API
        std::shared_ptr<Node> numpy_broadcast(const Output<Node>& value, const Shape& shape);

        /// \brief Wrap two graph values, if necessary, to obtain values with identical shapes,
        /// using NumPy's auto-broadcast rules.
        ///
        /// The elements in the std::pair returned by this function correspond to those supplied
        /// in the std::pair provided via \p args.
        ///
        /// If \p args.first and \p args.second produce identical shapes, then the returned
        /// std::pair will have the same value as \p args.
        ///
        /// If \p args.first and \p args.second produce different shapes, then this function creates
        /// new ngraph::op::Reshape and/or ngraph::op::Broadcast nodes, as needed, to wrap
        /// \p args.first and/or \p args.second in a manner that yields values with the same shape.
        ///
        /// There are some shape combinations which the autobroadcast algoritm cannot handle.
        /// An exception is thrown when such combinations are provided to this function.
        ///
        /// \pre
        /// - \p args.first is not null
        /// - \p args.second is not null
        ///
        /// \post
        /// - The ngraph::Node objects pointed to by \p args.first and \p args.second have not been
        ///   altered by this function, except by possibly having added consumers of their values.
        ///
        /// - If an exception was not thrown, then the return value's \p first and \p second
        ///   elements point to ngraph::Node objects whose output values have the same shape.
        ///
        /// \exception ngraph::builder::numpy_autobroadcast_incompatible_shapes
        NGRAPH_API
        std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>
            numpy_broadcast(const std::pair<Output<Node>, Output<Node>>& args);

        /// Create a new \p NodeType node, and any additional nodes required to simulate NumPy-style
        /// autobroadcast semantics.  Intended for binary operations such as "Add".
        ///
        /// \param [in] operand1_reshapeable The first operand to supply to the \p NodeType
        ///                                  constructor.  Subject to being wrapped with additional
        ///                                  nodes required for autobroadcasting.  Must not be null.
        ///
        /// \param [in] operand2_reshapeable The second operand to supply to the \p NodeType
        ///                                  constructor.  Subject to being wrapped with additional
        ///                                  nodes required for autobroadcasting.  Must not be null.
        ///
        /// \return The sink node of any/all nodes created by this function.  Will never be null.
        ///
        /// \exception ngraph::builder::numpy_autobroadcast_incompatible_shapes
        template <typename NodeType>
        std::shared_ptr<NodeType>
            make_with_numpy_broadcast(const Output<Node>& operand1_reshapeable,
                                      const Output<Node>& operand2_reshapeable)
        {
            auto shaped_op1_op2 = numpy_broadcast({operand1_reshapeable, operand2_reshapeable});
            return std::make_shared<NodeType>(shaped_op1_op2.first, shaped_op1_op2.second);
        }

        /// Create a new \p NodeType node, and any additional nodes required to simulate NumPy-style
        /// autobroadcast semantics.  Intended for non-binary operations such as "Select", where
        /// precisely the second and third operands are subject to autobroadcast semantics.
        ///
        /// \param [in] operand1 This operand is not subject to autobraodcast logic, and will be
        ///                      passed as-is as the first argument to the \p NodeType constructor.
        ///
        /// \param [in] operand2_reshapeable The second operand to supply to the \p NodeType
        ///                                  constructor.  Subject to being wrapped with additional
        ///                                  nodes required for autobroadcasting.  Must not be null.
        ///
        /// \param [in] operand3_reshapeable The third operand to supply to the \p NodeType
        ///                                  constructor.  Subject to being wrapped with additional
        ///                                  nodes required for autobroadcasting.  Must not be null.
        ///
        /// \return The sink node of any/all nodes created by this function.  Will never be null.
        ///
        /// \exception ngraph::builder::numpy_autobroadcast_incompatible_shapes
        template <typename NodeType>
        std::shared_ptr<Node> make_with_numpy_broadcast(const Output<Node>& operand1,
                                                        const Output<Node>& operand2_reshapeable,
                                                        const Output<Node>& operand3_reshapeable)
        {
            auto shaped_op2_op3 = numpy_broadcast({operand2_reshapeable, operand3_reshapeable});
            return std::make_shared<NodeType>(
                operand1, shaped_op2_op3.first, shaped_op2_op3.second);
        }

        ///
        /// \brief      Cast shape of two outputs to make them compatible for an element-wise binary
        ///             operation.
        ///
        /// \note       If necessary the right-hand-side argument will be broadcast to match the
        ///             shape of left-hand-side argument. The starting of the mutually equal shape
        ///             is specified by the argument "start_match_axis", and if it is not set suffix
        ///             matching is assumed.
        ///
        /// \note       This style of broadcast was used in ONNX Op sets prior to version 7, where
        ///             it was replaced by NumPy style auto-broadcasting mechanism.
        ///
        /// \param      left              Node which contain input of binary op.
        /// \param      right             Node which contain input of binary op.
        /// \param      start_match_axis  Position in shape denoting start of the mutually equal
        ///                               shape
        ///
        /// \return     Left and right node after broadcasting.
        ///
        NGRAPH_API
        OutputVector legacy_broadcast_for_binary_operation(const Output<Node>& left,
                                                           const Output<Node>& right,
                                                           size_t start_match_axis);

        /// \brief      Broadcast shape of two nodes to make them compatible for a matrix
        ///             multiplication.
        ///
        /// \note       This function is reflecting broadcasting behaviour of NumPy's `matmul`
        ///             operation.
        ///             (https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)
        ///             This mean that only \"stack of matrices\" axes are bidirectionally
        ///             broadcasted. The last two dimension are left untouched.
        ///
        /// \param[in]  left   The Node providing data for the left-hand side of matrix
        ///                    multiplication.
        /// \param[in]  right  The Node providing data for the right-hand side of matrix
        ///                    multiplication.
        ///
        /// \return     The vector containing both outputs broadcasted.
        ///
        NGRAPH_API
        OutputVector numpy_broadcast_for_matmul_operation(const Output<Node>& left,
                                                          const Output<Node>& right);

        /// \brief Cast shape of all input nodes for an element-wise operation that requires
        ///        shape-compatibility
        ///
        /// \param inputs Original list of inputs
        /// \param axis Index starting to align
        ///
        /// \return pdpd-style broadcasted list of nodes.
        NGRAPH_API
        OutputVector pdpd_broadcast(const OutputVector& inputs, int64_t axis);

        /// \brief Generate a list of broadcast axes.
        ///
        /// \details Informally, a broadcast "adds" axes to the input tensor, replicating
        ///          elements from the input tensor as needed to fill the new dimensions.
        ///          Function calculate which of the output axes are added in this way.
        ///
        /// \param output_shape      The new shape for the output tensor.
        /// \param input_shape       The shape of input tensor.
        /// \param start_match_axis  The axis along which we want to replicate elements.
        ///                          The starting axis position (0-based) int the output
        ///                          shape from which the current shape of the tensor
        ///                          matches the desired new shape.
        ///
        /// \return The indices of added axes.
        NGRAPH_API
        AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                         const Shape& input_shape,
                                         std::size_t start_match_axis);

        ///
        /// \brief      Calculate the output shape of numpy-style broadcast operation for all input
        ///             shapes.
        ///
        ///             This function finds the maximum tensor shape that will be the result of
        ///             element-wise operation that will be applied to the input shapes vector.
        ///             The function also prepares the shape of each input for the element-wise
        ///             operation by left-padding those shapes so that their rank is equal to the
        ///             left_shape's rank.
        ///
        /// \param      input_shapes  A vector of input shapes for which a common shape should be
        ///                           found
        ///
        /// \return     A pair that contains the target shape as its first object and a vector of
        ///             padded input shapes ready to be broadcasted as the second object
        ///
        NGRAPH_API
        std::pair<Shape, std::vector<Shape>>
            get_numpy_broadcast_shapes(const std::vector<Shape>& input_shapes);

        /// \brief Generate a list of broadcast along axes.
        ///
        /// \details Broadcast "adds" elements along axes to the input tensor, replicating
        ///          elements from the input tensor as needed to fill the new dimensions.
        ///          Function calculate which of the output axes are added in this way.
        ///
        ///          This function will attempt to match shapes, assuming the current shape
        ///          matches the rightmost positions of the desired new shape. This behaviour
        ///          is similar to NumPy's broadcasting.
        ///
        /// \param output_shape The new shape for the output tensor.
        /// \param input_shape  The shape of input tensor.
        ///
        /// \return             The indices of added axes.
        inline AxisSet calculate_broadcast_axes(const Shape& output_shape, const Shape& input_shape)
        {
            return calculate_broadcast_axes(
                output_shape, input_shape, output_shape.size() - input_shape.size());
        }

        inline std::shared_ptr<Node> make_broadcast_node(const Output<Node>& output,
                                                         Shape new_shape)
        {
            return std::make_shared<op::v0::Broadcast>(
                output, new_shape, calculate_broadcast_axes(new_shape, output.get_shape()));
        }

        inline std::shared_ptr<Node> make_broadcast_node(const Output<Node>& value,
                                                         const Shape& new_shape,
                                                         std::size_t start_match_axis)
        {
            return std::make_shared<op::v0::Broadcast>(
                value,
                new_shape,
                calculate_broadcast_axes(new_shape, value.get_shape(), start_match_axis));
        }

        namespace opset1
        {
            ///
            /// \brief      Broadcast right node to left node's shape using legacy scheme.
            ///
            /// \param[in]  left              The left hand side node of binary operation.
            /// \param[in]  right             The right hand side node of binary operation. The one
            ///                               to be broadcasted.
            /// \param[in]  start_match_axis  The axis index starting mutually equal shapes
            ///                               of both nodes.
            ///
            /// \return     The Output object connected to node producing broadcasted right node.
            ///
            NGRAPH_API
            Output<Node> legacy_broadcast_for_binary_operation(const Output<Node>& left,
                                                               const Output<Node>& right,
                                                               size_t start_match_axis);

            ///
            /// \brief      Reconstructs axes mapping vector for Broadcast:v1 operation.
            ///
            /// \param[in]  output_shape    The output shape of Broadcast operation.
            /// \param[in]  broadcast_axes  The broadcast axes used for Broadcast:v0 operator.
            ///
            /// \return     The vector with axes indexes mapping .
            ///
            NGRAPH_API
            std::vector<std::size_t> get_axes_mapping(const Shape& output_shape,
                                                      const AxisSet& broadcast_axes);

            ///
            /// \brief      Creates Node returning the axes mapping for Broadcast:v1 operation.
            ///
            /// \param[in]  output_shape      The output shape of Broadcast operation.
            /// \param[in]  input_shape       The input shape.
            /// \param[in]  start_match_axis  The axis index at which input shape starts to be
            ///                               identical as the output shape.
            ///
            /// \return     Returns the Output object pointing to node with the axes mapping.
            ///
            NGRAPH_API
            Output<Node> get_axes_mapping_output(const Shape& output_shape,
                                                 const Shape& input_shape,
                                                 std::size_t start_match_axis);

            ///
            /// \brief      Creates Node returning the axes mapping for Broadcast:v1 operation.
            ///
            /// \param[in]  output_shape    The output shape of Broadcast operation.
            /// \param[in]  broadcast_axes  The broadcast axes used for Broadcast:v0 operator.
            ///
            /// \return     The Output object with Node returning axes mapping.
            ///
            NGRAPH_API
            Output<Node> get_axes_mapping_output(const Shape& output_shape,
                                                 const AxisSet& broadcast_axes);

            NGRAPH_API
            Output<Node> make_broadcast(const Output<Node>& node,
                                        const Shape& target_shape,
                                        const AxisSet& broadcast_axes);

            NGRAPH_API
            Output<Node> make_broadcast(const Output<Node>& node,
                                        const Shape& target_shape,
                                        std::size_t start_match_axis);

        } // namespace opset1
    }     // namespace builder
} // namespace ngraph
