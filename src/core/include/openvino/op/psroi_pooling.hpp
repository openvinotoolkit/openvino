// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief PSROIPooling operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PSROIPooling : public Op {
public:
    OPENVINO_OP("PSROIPooling", "opset1");
    BWDCMP_RTTI_DECLARATION;

    PSROIPooling() = default;
    /// \brief Constructs a PSROIPooling operation
    ///
    /// \param input          Input feature map {N, C, ...}
    /// \param coords         Coordinates of bounding boxes
    /// \param output_dim     Output channel number
    /// \param group_size     Number of groups to encode position-sensitive scores
    /// \param spatial_scale  Ratio of input feature map over input image size
    /// \param spatial_bins_x Numbers of bins to divide the input feature maps over
    /// width
    /// \param spatial_bins_y Numbers of bins to divide the input feature maps over
    /// height
    /// \param mode           Mode of pooling - Avg or Bilinear
    PSROIPooling(const Output<Node>& input,
                 const Output<Node>& coords,
                 const size_t output_dim,
                 const size_t group_size,
                 const float spatial_scale,
                 int spatial_bins_x,
                 int spatial_bins_y,
                 const std::string& mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_output_dim() const {
        return m_output_dim;
    }
    size_t get_group_size() const {
        return m_group_size;
    }
    float get_spatial_scale() const {
        return m_spatial_scale;
    }
    int get_spatial_bins_x() const {
        return m_spatial_bins_x;
    }
    int get_spatial_bins_y() const {
        return m_spatial_bins_y;
    }
    const std::string& get_mode() const {
        return m_mode;
    }

private:
    size_t m_output_dim;
    size_t m_group_size;
    float m_spatial_scale;
    int m_spatial_bins_x;
    int m_spatial_bins_y;
    std::string m_mode;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
