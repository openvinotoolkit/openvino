// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Base class for operations DeformableConvolution v1 and DeformableConvolution
            /// v8.
            class NGRAPH_API DeformableConvolutionBase : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a conversion operation.
                DeformableConvolutionBase() = default;

                /// \brief Constructs a conversion operation.
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
                DeformableConvolutionBase(const OutputVector& arguments,
                                          const Strides& strides,
                                          const CoordinateDiff& pads_begin,
                                          const CoordinateDiff& pads_end,
                                          const Strides& dilations,
                                          const PadType& auto_pad = PadType::EXPLICIT,
                                          int64_t group = 1,
                                          int64_t deformable_group = 1);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                const CoordinateDiff& get_pads_end() const { return m_pads_end; }
                void set_pads_end(const CoordinateDiff& pads_end) { m_pads_end = pads_end; }
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                int64_t get_group() const { return m_group; }
                void set_group(const int64_t group) { m_group = group; }
                int64_t get_deformable_group() const { return m_deformable_group; }
                void set_deformable_group(const int64_t deformable_group)
                {
                    m_deformable_group = deformable_group;
                }

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
                int64_t m_group;
                int64_t m_deformable_group;
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph
