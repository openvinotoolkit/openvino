// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/interpolate_base.hpp"

namespace ov {
namespace op {
namespace v0 {

/// \brief Layer which performs bilinear interpolation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public Op {
public:
    OPENVINO_OP("Interpolate", "opset1");
    /// \brief Structure that specifies attributes for interpolation
    struct Attributes {
        // specify dimension indices where interpolation is applied, and `axes` is any
        // unordered list of indices of different dimensions of input tensor. Required.
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

    enum class InterpolateMode {
        NEAREST,
        LINEAR,
        CUBIC,
        AREA,
        nearest OPENVINO_ENUM_DEPRECATED("Please use NEAREST instead") = NEAREST,
        linear OPENVINO_ENUM_DEPRECATED("Please use LINEAR instead") = LINEAR,
        cubic OPENVINO_ENUM_DEPRECATED("Please use CUBIC instead") = CUBIC,
        area OPENVINO_ENUM_DEPRECATED("Please use AREA instead") = AREA
    };

    Interpolate() = default;
    /// \brief Constructs a Interpolate operation
    ///
    /// \param image        Input image
    /// \param output_shape Output shape of spatial axes
    /// \param attrs        Interpolation attributes
    Interpolate(const Output<Node>& image, const Output<Node>& output_shape, const Attributes& attrs);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

private:
    Attributes m_attrs;
};
}  // namespace v0

namespace v4 {
/// \brief Interpolate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public util::InterpolateBase {
public:
    OPENVINO_OP("Interpolate", "opset4", util::InterpolateBase);

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

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

    const InterpolateAttrs& get_attrs() const {
        return m_attrs;
    }
    void set_attrs(const InterpolateAttrs& attrs) {
        this->m_attrs = attrs;
    }

protected:
    /// \return The interpolation axes.
    std::vector<int64_t> get_axes() const;

private:
    bool evaluate_interpolate(const HostTensorVector& outputs, const HostTensorVector& inputs) const;

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
    /// \param input_shape PartialShape of input data.
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

    template <class T>
    friend void shape_infer(const Interpolate* op,
                            std::vector<size_t>& pads_begin,
                            std::vector<size_t>& pads_end,
                            const std::vector<T>& input_shapes,
                            std::vector<T>& output_shapes,
                            const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data);
};
}  // namespace v4

namespace v11 {
/// \brief Interpolate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public util::InterpolateBase {
public:
    OPENVINO_OP("Interpolate", "opset11", util::InterpolateBase);
    Interpolate() = default;
    /// \brief Constructs a Interpolate operation without 'axes' input.
    ///
    /// \param image  Input image
    /// \param scales_or_sizes Scales of spatial axes, i.e. output_shape / input_shape
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image, const Output<Node>& scales_or_sizes, const InterpolateAttrs& attrs);

    /// \brief Constructs a Interpolate operation with 'axes' input.
    ///
    /// \param image  Input image
    /// \param scales_or_sizes Scales of spatial axes, i.e. output_shape / input_shape
    /// \param axes   Interpolation axes
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image,
                const Output<Node>& scales_or_sizes,
                const Output<Node>& axes,
                const InterpolateAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool has_evaluate() const override {
        return false;
    }
};
}  // namespace v11
}  // namespace op

//---------------------------------------- v0 --------------------------------------------------
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v0::Interpolate::InterpolateMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v0::Interpolate::InterpolateMode>
    : public EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode> {
public:
    AttributeAdapter(op::v0::Interpolate::InterpolateMode& value)
        : EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v0::Interpolate::InterpolateMode>");
};
}  // namespace ov
