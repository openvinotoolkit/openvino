// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/op/interpolate.hpp"

#include <map>

namespace cldnn {

/// @brief Performs nearest neighbor/bilinear resample
/// Also supports built-in Relu @ref activation available by setting it in arguments.
struct resample : public primitive_base<resample> {
    CLDNN_DECLARE_PRIMITIVE(resample)

    resample() : primitive_base("", {}) {}

    using InterpolateOp = ov::op::util::InterpolateBase;

    /// @brief Constructs Resample primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Resample scale.
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    /// @param sample_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    resample(const primitive_id& id,
             const input_info& input,
             tensor output_size,
             uint32_t num_filter,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::NEAREST)
        : primitive_base(id, {input}),
          output_size(output_size),
          num_filter(num_filter),
          sizes({}),
          scales({}),
          axes({}),
          pads_begin({}),
          pads_end({}),
          operation_type(operation_type),
          shape_calc_mode(InterpolateOp::ShapeCalcMode::SIZES),
          antialias(0),
          cube_coeff(0.0f),
          coord_trans_mode(InterpolateOp::CoordinateTransformMode::ASYMMETRIC),
          round_mode(InterpolateOp::NearestMode::FLOOR) {
        if (scales.size() != axes.size())
            throw std::runtime_error("Resample's scales/axes count does not match");
        if (operation_type == InterpolateOp::InterpolateMode::LINEAR) {
            coord_trans_mode = InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
        }
    }

    /// @brief resample with constant sizes/scales
    resample(const primitive_id& id,
             const input_info& input,
             const std::vector<int64_t>& sizes,
             const std::vector<float>& scales,
             const std::vector<int64_t>& axes,
             const std::vector<size_t>& pads_begin = {},
             const std::vector<size_t>& pads_end = {},
             int32_t antialias = 0,
             float cube_coeff = -0.75f,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::LINEAR,
             InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES,
             InterpolateOp::CoordinateTransformMode ctm = InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
             InterpolateOp::NearestMode nm = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR)
        : primitive_base(id, {input}),
          output_size(tensor()),
          num_filter(0),
          sizes(sizes),
          scales(scales),
          axes(axes),
          pads_begin(pads_begin),
          pads_end(pads_end),
          operation_type(operation_type),
          shape_calc_mode(shape_calc_mode),
          antialias(antialias),
          cube_coeff(cube_coeff),
          coord_trans_mode(ctm),
          round_mode(nm) {
        if (scales.size() != axes.size() && shape_calc_mode == InterpolateOp::ShapeCalcMode::SCALES)
            throw std::runtime_error("Resample's scales/axes count does not match");
    }

    /// @brief resample with dynamic sizes/scales
    resample(const primitive_id& id,
             const input_info& input,
             const input_info& sizes_id,
             const input_info& scales_id,
             const std::vector<int64_t>& axes,
             const std::vector<size_t>& pads_begin = {},
             const std::vector<size_t>& pads_end = {},
             int32_t antialias = 0,
             float cube_coeff = -0.75f,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::LINEAR,
             InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES,
             InterpolateOp::CoordinateTransformMode ctm = InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
             InterpolateOp::NearestMode nm = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR,
             const int scales_port = 2)
        : primitive_base(id, {input, sizes_id, scales_id}),
          output_size(tensor()),
          num_filter(0),
          scales_port(scales_port),
          sizes({}),
          scales({}),
          axes(axes),
          pads_begin(pads_begin),
          pads_end(pads_end),
          operation_type(operation_type),
          shape_calc_mode(shape_calc_mode),
          antialias(antialias),
          cube_coeff(cube_coeff),
          coord_trans_mode(ctm),
          round_mode(nm) {}

    InterpolateOp::InterpolateAttrs get_attrs() const {
        return InterpolateOp::InterpolateAttrs(this->operation_type,
                                               this->shape_calc_mode,
                                               this->pads_begin,
                                               this->pads_end,
                                               this->coord_trans_mode,
                                               this->round_mode,
                                               static_cast<bool>(this->antialias),
                                               this->cube_coeff);
    }

    tensor output_size;
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    uint32_t num_filter = 0;
    /// @param num_filter Port number of scales.
    uint32_t scales_port;
    /// @param sizes Describing output shape for spatial axes.
    std::vector<int64_t> sizes;
    /// @param scales Scales of spatial axes, i.e. output_shape / input_shape
    std::vector<float> scales;
    /// @param axes Interpolation axes.
    std::vector<int64_t> axes;
    /// @param pads_begin Begin paddings for input.
    std::vector<size_t> pads_begin;
    /// @param pads_end End paddings for input.
    std::vector<size_t> pads_end;
    /// @param operation_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::LINEAR;
    /// @param shape_calc_mode Specifies which input, sizes or scales, is used to calculate an output shape.
    InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES;
    /// @param antialias is a flag that specifies whether to perform anti-aliasing.
    int32_t antialias = 0;
    /// @param cube_coeff specifies the parameter a for cubic interpolation. cube_coeff is used only when mode == cubic.
    float cube_coeff = -0.75f;
    /// @param coord_trans_mode specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor
    InterpolateOp::CoordinateTransformMode coord_trans_mode = InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    /// @param round_mode specifies round mode when mode == nearest and is used only when mode == nearest.
    InterpolateOp::NearestMode round_mode = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_filter);
        seed = hash_range(seed, scales.begin(), scales.end());
        seed = hash_range(seed, axes.begin(), axes.end());
        seed = hash_range(seed, pads_begin.begin(), pads_begin.end());
        seed = hash_range(seed, pads_end.begin(), pads_end.end());
        seed = hash_combine(seed, operation_type);
        seed = hash_combine(seed, shape_calc_mode);
        seed = hash_combine(seed, antialias);
        seed = hash_combine(seed, cube_coeff);
        seed = hash_combine(seed, coord_trans_mode);
        seed = hash_combine(seed, round_mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const resample>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(num_filter) &&
               cmp_fields(sizes) &&
               cmp_fields(scales) &&
               cmp_fields(axes) &&
               cmp_fields(pads_begin) &&
               cmp_fields(pads_end) &&
               cmp_fields(operation_type) &&
               cmp_fields(shape_calc_mode) &&
               cmp_fields(antialias) &&
               cmp_fields(cube_coeff) &&
               cmp_fields(coord_trans_mode) &&
               cmp_fields(round_mode);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<resample>::save(ob);
        ob << output_size;
        ob << num_filter;
        ob << sizes;
        ob << scales;
        ob << axes;
        ob << pads_begin;
        ob << pads_end;
        ob << make_data(&operation_type, sizeof(InterpolateOp::InterpolateMode));
        ob << make_data(&shape_calc_mode, sizeof(InterpolateOp::ShapeCalcMode));
        ob << antialias;
        ob << cube_coeff;
        ob << make_data(&coord_trans_mode, sizeof(InterpolateOp::CoordinateTransformMode));
        ob << make_data(&round_mode, sizeof(InterpolateOp::NearestMode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<resample>::load(ib);
        ib >> output_size;
        ib >> num_filter;
        ib >> sizes;
        ib >> scales;
        ib >> axes;
        ib >> pads_begin;
        ib >> pads_end;
        ib >> make_data(&operation_type, sizeof(InterpolateOp::InterpolateMode));
        ib >> make_data(&shape_calc_mode, sizeof(InterpolateOp::ShapeCalcMode));
        ib >> antialias;
        ib >> cube_coeff;
        ib >> make_data(&coord_trans_mode, sizeof(InterpolateOp::CoordinateTransformMode));
        ib >> make_data(&round_mode, sizeof(InterpolateOp::NearestMode));
    }
};
}  // namespace cldnn
