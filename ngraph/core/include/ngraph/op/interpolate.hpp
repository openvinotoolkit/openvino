// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Structure that specifies attributes for interpolation
            struct InterpolateAttrs
            {
                // specify dimension indices where interpolation is applied, and `axes` is any
                // unordered list of indeces of different dimensions of input tensor. Required.
                AxisSet axes;
                // specifies type of interpolation
                // one of `nearest`, `linear`, `cubic`, `area`. Required.
                std::string mode;
                // a flag that specifies whether to align corners or not.
                // `true` (default) means the alignment is applied,
                // `false` means the alignment isn't applied.
                bool align_corners = true;
                // a flag that specifies whether to perform anti-aliasing. default is `false`
                bool antialias = false;
                // specify the number of pixels to add to the beginning of the image being
                // interpolated. This addition of pixels is done before interpolation calculation.
                std::vector<size_t> pads_begin;
                // specify the number of pixels to add to the end of the image being interpolated.
                // This addition of pixels is done before interpolation calculation.
                std::vector<size_t> pads_end;
            };

            /// \brief Layer which performs bilinear interpolation
            class NGRAPH_API Interpolate : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                enum class InterpolateMode
                {
                    nearest,
                    linear,
                    cubic,
                    area
                };

                Interpolate() = default;
                /// \brief Constructs a Interpolate operation
                ///
                /// \param image        Input image
                /// \param output_shape Output shape of spatial axes
                /// \param attrs        Interpolation attributes
                Interpolate(const Output<Node>& image,
                            const Output<Node>& output_shape,
                            const InterpolateAttrs& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const InterpolateAttrs& get_attrs() const { return m_attrs; }

            private:
                InterpolateAttrs m_attrs;
            };
        } // namespace v0

        namespace v4
        {
            class NGRAPH_API Interpolate : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Shape calculation mode
                ///
                /// sizes  - output shape for interpolated axes is calculated using input `sizes`
                /// scales - output shape for interpolated axes is calculated using input `scales`
                enum class ShapeCalcMode
                {
                    sizes,
                    scales
                };

                /// \brief Interpolation mode
                ///
                /// nearest     - nearest interpolation
                /// linear      - linear interpolation as in TensorFlow
                /// linear_onnx - linear interpolation as in ONNX
                /// cubic       - cubic interpolation
                enum class InterpolateMode
                {
                    nearest,
                    linear,
                    linear_onnx,
                    cubic
                };

                /// \brief Mode of the calculation of the source coordinate from resized one
                ///
                /// These modes are modes from ONNX runtime.
                enum class CoordinateTransformMode
                {
                    half_pixel,
                    pytorch_half_pixel,
                    asymmetric,
                    tf_half_pixel_for_nn,
                    align_corners
                };

                /// \brief Round modes for the nearest interpolation.
                enum class NearestMode
                {
                    round_prefer_floor,
                    round_prefer_ceil,
                    floor,
                    ceil,
                    simple
                };

                struct InterpolateAttrs
                {
                    // specifies type of interpolation
                    // one of `nearest`, `linear`, `linear_onnx`, `cubic` Required.
                    InterpolateMode mode = InterpolateMode::nearest;
                    // specifies shape calculation mode
                    // one of `sizes`, `scales` Required
                    ShapeCalcMode shape_calculation_mode = ShapeCalcMode::sizes;
                    // specify the number of pixels to add to the beginning of the image being
                    // interpolated. This addition of pixels is done before interpolation
                    // calculation.
                    std::vector<size_t> pads_begin;
                    // specify the number of pixels to add to the end of the image being
                    // interpolated. This addition of pixels is done before interpolation
                    // calculation.
                    std::vector<size_t> pads_end;
                    // specifies how to transform the coordinate in the resized tensor to the
                    // coordinate in the original tensor. one of `half_pixel`, `pytorch_half_pixel`,
                    // `asymmetric`, `tf_half_pixel_for_nn`, `align_corners`
                    CoordinateTransformMode coordinate_transformation_mode =
                        CoordinateTransformMode::half_pixel;
                    // specifies round mode when `mode == nearest` and is used only when `mode ==
                    // nearest`. one of `round_prefer_floor`, `round_prefer_ceil`, `floor`, `ceil`,
                    // `simple`
                    NearestMode nearest_mode = NearestMode::round_prefer_floor;
                    // a flag that specifies whether to perform anti-aliasing. default is `false`
                    bool antialias = false;
                    // specifies the parameter *a* for cubic interpolation (see, e.g.
                    // [article](https://ieeexplore.ieee.org/document/1163711/)).  *cube_coeff* is
                    // used only when `mode == cubic`
                    double cube_coeff = -0.75f;

                    InterpolateAttrs() = default;

                    InterpolateAttrs(InterpolateMode mode,
                                     ShapeCalcMode shape_calculation_mode,
                                     std::vector<size_t> pads_begin,
                                     std::vector<size_t> pads_end,
                                     CoordinateTransformMode coordinate_transformation_mode =
                                         CoordinateTransformMode::half_pixel,
                                     NearestMode nearest_mode = NearestMode::round_prefer_floor,
                                     bool antialias = false,
                                     double cube_coeff = -0.75)
                        : mode(mode)
                        , shape_calculation_mode(shape_calculation_mode)
                        , pads_begin(pads_begin)
                        , pads_end(pads_end)
                        , coordinate_transformation_mode(coordinate_transformation_mode)
                        , nearest_mode(nearest_mode)
                        , antialias(antialias)
                        , cube_coeff(cube_coeff)
                    {
                    }
                };

                Interpolate() = default;
                /// \brief Constructs a Interpolate operation without 'axes' input.
                ///
                /// \param image  Input image
                /// \param output_shape Output shape of spatial axes
                /// \param scales Scales of spatial axes, i.e. output_shape / input_shape
                /// \param attrs  Interpolation attributes
                Interpolate(const Output<Node>& image,
                            const Output<Node>& output_shape,
                            const Output<Node>& scales,
                            const InterpolateAttrs& attrs);

                /// \brief Constructs a Interpolate operation with 'axes' input.
                ///
                /// \param image  Input image
                /// \param output_shape Output shape of spatial axes
                /// \param scales Scales of spatial axes, i.e. output_shape / input_shape
                /// \param axes   Interpolation axes
                /// \param attrs  Interpolation attributes
                Interpolate(const Output<Node>& image,
                            const Output<Node>& output_shape,
                            const Output<Node>& scales,
                            const Output<Node>& axes,
                            const InterpolateAttrs& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

                const InterpolateAttrs& get_attrs() const { return m_attrs; }

            protected:
                /// \return The interpolation axes.
                std::vector<int64_t> get_axes() const;

            private:
                bool evaluate_interpolate(const HostTensorVector& outputs,
                                          const HostTensorVector& inputs) const;
                InterpolateAttrs m_attrs;

                /// \brief Corrects pads_begin and pads_end attributes.
                ///
                /// \details When Interpolate-4 is a result of some transformation, it is possible
                ///          that pads_begin.size() != pads_end.size() or
                ///          pads_begin.size() != input_rank. In such case, we should correct
                ///          pads_begin and pads_end, using padding of pads_begin and pads_end by
                ///          zeros or using pads_begin[0 : input_rank], pads_end[0 : input_rank].
                ///
                ///          Padding of pads_begin is performed when pads_begin.size() < input_rank,
                ///          and pads_begin[0 : input_rank] is used when
                ///          pads_begin.size() < input_rank.
                ///
                ///          Similarly for pads_end.
                void correct_pads();

                /// \brief Calculates input shape after padding.
                ///
                /// \param input_shape Shape of input data.
                ///
                /// \return Padded input shape, i.e. input_shape + pads_begin + pads_end
                PartialShape get_padded_input_shape(const PartialShape& input_shape) const;

                /// \brief Infers output shape using scales.
                ///
                /// \param output_shape[in,out] output shape
                /// \param axes Interpolation axes
                /// \param scales Scales for interpolated axes
                /// \param padded_input_shape input shape after padding
                void infer_using_scales(PartialShape& output_shape,
                                        const std::vector<int64_t>& axes,
                                        const std::vector<float>& scales,
                                        const PartialShape& padded_input_shape) const;

                /// \brief Infers output shape using sizes.
                ///
                /// \param output_shape[in,out] output shape
                /// \param axes Interpolation axes
                /// \param sizes sizes for interpolated axes
                void infer_using_shapes(PartialShape& output_shape,
                                        const std::vector<int64_t>& axes,
                                        const std::vector<int64_t>& sizes) const;
            };
        } // namespace v4

        using v0::Interpolate;
        using v0::InterpolateAttrs;
    } // namespace op

    //---------------------------------------- v0 --------------------------------------------------
    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v0::Interpolate::InterpolateMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v0::Interpolate::InterpolateMode>
        : public EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode>
    {
    public:
        AttributeAdapter(op::v0::Interpolate::InterpolateMode& value)
            : EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v0::Interpolate::InterpolateMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    //---------------------------------------- v4 --------------------------------------------------

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::InterpolateMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v4::Interpolate::InterpolateMode>
        : public EnumAttributeAdapterBase<op::v4::Interpolate::InterpolateMode>
    {
    public:
        AttributeAdapter(op::v4::Interpolate::InterpolateMode& value)
            : EnumAttributeAdapterBase<op::v4::Interpolate::InterpolateMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v4::Interpolate::InterpolateMode>", 4};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s,
                             const op::v4::Interpolate::CoordinateTransformMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v4::Interpolate::CoordinateTransformMode>
        : public EnumAttributeAdapterBase<op::v4::Interpolate::CoordinateTransformMode>
    {
    public:
        AttributeAdapter(op::v4::Interpolate::CoordinateTransformMode& value)
            : EnumAttributeAdapterBase<op::v4::Interpolate::CoordinateTransformMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v4::Interpolate::CoordinateTransformMode>", 4};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::NearestMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v4::Interpolate::NearestMode>
        : public EnumAttributeAdapterBase<op::v4::Interpolate::NearestMode>
    {
    public:
        AttributeAdapter(op::v4::Interpolate::NearestMode& value)
            : EnumAttributeAdapterBase<op::v4::Interpolate::NearestMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v4::Interpolate::NearestMode>", 4};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::ShapeCalcMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v4::Interpolate::ShapeCalcMode>
        : public EnumAttributeAdapterBase<op::v4::Interpolate::ShapeCalcMode>
    {
    public:
        AttributeAdapter(op::v4::Interpolate::ShapeCalcMode& value)
            : EnumAttributeAdapterBase<op::v4::Interpolate::ShapeCalcMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v4::Interpolate::ShapeCalcMode>", 4};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
