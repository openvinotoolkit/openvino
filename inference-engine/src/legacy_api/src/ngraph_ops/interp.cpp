// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/interp.hpp"

#include <limits>
#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Interp::type_info;

op::Interp::Interp(const Output<Node>& image, const InterpolateIEAttrs& attrs)
    : Op({image}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::Interp::validate_and_infer_types() {
    if (get_input_partial_shape(0).is_static()) {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape output_shape(4);
        // Assumes {N, C, H, W}
        output_shape[0] = input_shape[0];
        output_shape[1] = input_shape[1];

        auto is_zero = [](float value) {
            return std::fabs(value) < std::numeric_limits<float>::epsilon();
        };

        bool should_scale =
            !(is_zero(m_attrs.zoom_factor) && is_zero(m_attrs.shrink_factor) && is_zero(m_attrs.scale_factor));

        if (should_scale) {
            float scale = m_attrs.scale_factor;
            if (!is_zero(m_attrs.shrink_factor) || !is_zero(m_attrs.zoom_factor)) {
                if (!is_zero(m_attrs.zoom_factor)) {
                    scale = m_attrs.zoom_factor;
                }
                if (!is_zero(m_attrs.shrink_factor)) {
                    scale /= m_attrs.shrink_factor;
                }
            }

            output_shape[2] = static_cast<Shape::value_type>(input_shape[2] * scale);
            output_shape[3] = static_cast<Shape::value_type>(input_shape[3] * scale);
        }

        if (m_attrs.height > 0) {
            output_shape[2] = m_attrs.height;
        }
        if (m_attrs.width > 0) {
            output_shape[3] = m_attrs.width;
        }

        set_output_type(0, get_input_element_type(0), output_shape);
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::Interp::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<Interp>(new_args.at(0), m_attrs);
}

constexpr NodeTypeInfo op::ResampleV2::type_info;

op::ResampleV2::ResampleV2(const Output<Node>& image, const Output<Node>& output_shape,
                           const ResampleIEAttrs& attrs)
    : Op({image, output_shape}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

op::ResampleV2::ResampleV2(const Output<Node>& image, const ResampleIEAttrs& attrs)
        : Op({image}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::ResampleV2::validate_and_infer_types() {
    if (m_attrs.factor != 0) {
        Shape output_shape(get_input_shape(0));
        for (size_t i = 2; i < output_shape.size(); ++i) {
            output_shape[i] *= m_attrs.factor;
        }
        set_output_type(0, get_input_element_type(0), output_shape);
    } else if (auto const_shape = dynamic_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())) {
        NODE_VALIDATION_CHECK(this, shape_size(const_shape->get_shape()) == 4 || shape_size(const_shape->get_shape()) == 5,
                              "Layer shape must have rank 4 or 5", const_shape->get_shape());

        auto out_shape = const_shape->cast_vector<int64_t>();
        Shape output_shape;
        for (size_t i = 0; i < const_shape->get_shape()[0]; i++) {
            output_shape.push_back((out_shape[i] > 0) ? out_shape[i] : 0);
        }
        set_output_type(0, get_input_element_type(0), output_shape);
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ResampleV2::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ResampleV2>(new_args.at(0), new_args.at(1), m_attrs);
}


static inline float triangleCoeff(float x) {
    return std::max(0.0f, 1 - std::abs(x));
}

bool op::ResampleV2::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    if (inputs.front()->get_element_type() != outputs.front()->get_element_type()) {
        throw ngraph_error("Input and output data types must be the same!");
    }
    auto src_data = inputs.front()->get_data_ptr<const float>();
    auto* dst_data = outputs.front()->get_data_ptr<float>();

    Shape inputShape = get_input_partial_shape(0).to_shape();
    Shape output_shape(get_input_shape(0));
    for (size_t i = 2; i < output_shape.size(); ++i) {
        output_shape[i] *= m_attrs.factor;
    }

    size_t ndims = inputShape.size();
    size_t N = inputShape[0];
    size_t C = inputShape[1];
    size_t ID = ndims == 5 ? inputShape[ndims - 3] : 1;
    size_t IH = inputShape[ndims - 2];
    size_t IW = inputShape[ndims - 1];
    size_t OD = ndims == 5 ? ID * m_attrs.factor : 1;
    size_t OH = IH * m_attrs.factor;
    size_t OW = IW * m_attrs.factor;

    float fx = static_cast<float>(IW) / static_cast<float>(OW);
    float fy = static_cast<float>(IH) / static_cast<float>(OH);
    float fz = static_cast<float>(ID) / static_cast<float>(OD);

    if (m_attrs.mode == "nearest") {
        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float* in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float* out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;

                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx;
                            float iy = oy * fy;
                            float iz = oz * fz;

                            size_t ix_r = static_cast<size_t>(std::floor(ix));
                            size_t iy_r = static_cast<size_t>(std::floor(iy));
                            size_t iz_r = static_cast<size_t>(std::floor(iz));

                            out_ptr[oz * OH * OW + oy * OW + ox] = in_ptr[iz_r * IH * IW + iy_r * IW + ix_r];
                        }
                    }
                }
            }
        }
    } else if (m_attrs.mode == "linear") {
        size_t kernel_width = 2;
        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
        bool antialias = isDownsample && m_attrs.antialias;

        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float* in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float* out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;

                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fx / 2.0f - 0.5f;
                            float iy = oy * fy + fy / 2.0f - 0.5f;
                            float iz = oz * fz + fz / 2.0f - 0.5f;

                            int ix_r = static_cast<int>(round(ix));
                            int iy_r = static_cast<int>(round(iy));
                            int iz_r = static_cast<int>(round(iz));

                            float sum = 0;
                            float wsum = 0;

                            float ax = 1.0f / (antialias ? fx : 1.0f);
                            float ay = 1.0f / (antialias ? fy : 1.0f);
                            float az = 1.0f / (antialias ? fz : 1.0f);

                            int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
                            int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
                            int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

                            for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                                for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                                    for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                        if (z < 0 || y < 0 || x < 0 || z >= static_cast<int>(ID) || y >= static_cast<int>(IH) || x >= static_cast<int>(IW))
                                            continue;

                                        float dx = ix - x;
                                        float dy = iy - y;
                                        float dz = iz - z;

                                        float w = ax * triangleCoeff(ax * dx) * ay * triangleCoeff(ay * dy) * az * triangleCoeff(az * dz);

                                        sum += w * in_ptr[z * IH * IW + y * IW + x];
                                        wsum += w;
                                    }
                                }
                            }
                            out_ptr[oz * OH * OW + oy * OW + ox] = (!wsum) ? 0 : (sum / wsum);
                        }
                    }
                }
            }
        }
    }

    return true;
}
