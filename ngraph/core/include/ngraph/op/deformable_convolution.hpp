// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/deformable_convolution_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief DeformableConvolution operation.
            class NGRAPH_API DeformableConvolution : public op::util::DeformableConvolutionBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs a conversion operation.
                DeformableConvolution() = default;
                /// \brief Constructs a conversion operation.
                ///
                /// \param arg                Node that produces the input tensor.
                /// \param offsets            Node producing the deformable values tensor.
                /// \param filters            Node producing the filters(kernels) tensor with OIZYX
                ///                           layout.
                /// \param strides            Convolution strides.
                /// \param pads_begin         Amount of padding to be added to the beginning along
                ///                           each axis. For example in case of a 2D input the value
                ///                           of (1, 2) means that 1 element will be added to the
                ///                           top and 2 elements to the left.
                /// \param pads_end           Amount of padding to be added to the end along each
                ///                           axis.
                /// \param dilations          The distance in width and height between the weights
                ///                           in the filters tensor.
                /// \param auto_pad           Specifies how the automatic calculation of padding
                ///                           should be done.
                /// \param group              The number of groups which both output and input
                ///                           should be split into.
                /// \param deformable_group   The number of groups which deformable values and
                ///                           output should be split into along the channel axis.
                DeformableConvolution(const Output<Node>& arg,
                                      const Output<Node>& offsets,
                                      const Output<Node>& filters,
                                      const Strides& strides,
                                      const CoordinateDiff& pads_begin,
                                      const CoordinateDiff& pads_end,
                                      const Strides& dilations,
                                      const PadType& auto_pad = PadType::EXPLICIT,
                                      const int64_t group = 1,
                                      const int64_t deformable_group = 1);

                std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v1


        namespace v8
        {
            class NGRAPH_API DeformableConvolution : public op::util::DeformableConvolutionBase {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a conversion operation.
                DeformableConvolution() = default;
                /// \brief Constructs a conversion operation.
                ///
                /// \param arg                Node that produces the input tensor.
                /// \param offsets            Node producing the deformable values tensor.
                /// \param filters            Node producing the filters(kernels) tensor with OIZYX
                ///                           layout.
                /// \param strides            Convolution strides.
                /// \param pads_begin         Amount of padding to be added to the beginning along
                ///                           each axis. For example in case of a 2D input the value
                ///                           of (1, 2) means that 1 element will be added to the
                ///                           top and 2 elements to the left.
                /// \param pads_end           Amount of padding to be added to the end along each
                ///                           axis.
                /// \param dilations          The distance in width and height between the weights
                ///                           in the filters tensor.
                /// \param auto_pad           Specifies how the automatic calculation of padding
                ///                           should be done.
                /// \param group              The number of groups which both output and input
                ///                           should be split into.
                /// \param deformable_group   The number of groups which deformable values and
                ///                           output should be split into along the channel axis.
                DeformableConvolution(const Output<Node>& arg,
                                      const Output<Node>& offsets,
                                      const Output<Node>& filters,
                                      const Strides& strides,
                                      const CoordinateDiff& pads_begin,
                                      const CoordinateDiff& pads_end,
                                      const Strides& dilations,
                                      const PadType& auto_pad = PadType::EXPLICIT,
                                      const int64_t group = 1,
                                      const int64_t deformable_group = 1,
                                      const int64_t offset = 0);

                DeformableConvolution(const Output<Node>& arg,
                                      const Output<Node>& offsets,
                                      const Output<Node>& filters,
                                      const Output<Node>& scalars,
                                      const Strides& strides,
                                      const CoordinateDiff& pads_begin,
                                      const CoordinateDiff& pads_end,
                                      const Strides& dilations,
                                      const PadType& auto_pad = PadType::EXPLICIT,
                                      const int64_t group = 1,
                                      const int64_t deformable_group = 1,
                                      const int64_t offset = 0
                                      );
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

/*                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override { return true; }*/

            private:
                int64_t m_offset;

            };
        }
    }     // namespace op
} // namespace ngraph
