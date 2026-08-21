// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {

/// \brief A helper structure describing a continuous bound as [min, max, step].
struct OPENVINO_API Bound {
    float min = 0.f;
    float max = 0.f;
    float step = 1.f;
};

/// \brief BevPoolV2 operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BevPoolV2 : public Op {
public:
    OPENVINO_OP("BevPoolV2", "opset15");
    BevPoolV2() = default;

    /// \brief Constructs a BevPoolV2 operation.
    ///
    /// \param inputs Dynamic list of input tensors required by the kernel implementation.
    /// \param input_channels Number of input channels.
    /// \param output_channels Number of output channels.
    /// \param image_width Input image width.
    /// \param image_height Input image height.
    /// \param feature_width Feature map width.
    /// \param feature_height Feature map height.
    /// \param x_bound X-axis bound [min, max, step].
    /// \param y_bound Y-axis bound [min, max, step].
    /// \param z_bound Z-axis bound [min, max, step].
    /// \param d_bound D-axis bound [min, max, step].
    BevPoolV2(const OutputVector& inputs,
              uint32_t input_channels,
              uint32_t output_channels,
              uint32_t image_width,
              uint32_t image_height,
              uint32_t feature_width,
              uint32_t feature_height,
              const Bound& x_bound,
              const Bound& y_bound,
              const Bound& z_bound,
              const Bound& d_bound);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    uint32_t get_input_channels() const;
    uint32_t get_output_channels() const;
    uint32_t get_image_width() const;
    uint32_t get_image_height() const;
    uint32_t get_feature_width() const;
    uint32_t get_feature_height() const;
    const Bound& get_x_bound() const;
    const Bound& get_y_bound() const;
    const Bound& get_z_bound() const;
    const Bound& get_d_bound() const;

    void set_input_channels(uint32_t value);
    void set_output_channels(uint32_t value);
    void set_image_width(uint32_t value);
    void set_image_height(uint32_t value);
    void set_feature_width(uint32_t value);
    void set_feature_height(uint32_t value);
    void set_x_bound(const Bound& value);
    void set_y_bound(const Bound& value);
    void set_z_bound(const Bound& value);
    void set_d_bound(const Bound& value);

private:
    uint32_t m_input_channels = 0;
    uint32_t m_output_channels = 0;
    uint32_t m_image_width = 0;
    uint32_t m_image_height = 0;
    uint32_t m_feature_width = 0;
    uint32_t m_feature_height = 0;
    Bound m_x_bound{};
    Bound m_y_bound{};
    Bound m_z_bound{};
    Bound m_d_bound{};
};

}  // namespace v15
}  // namespace op
}  // namespace ov
