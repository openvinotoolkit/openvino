// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/constant.hpp"

#include "api/strided_slice.hpp"
#include "api/reshape.hpp"
#include "api/crop.hpp"

namespace CLDNNPlugin {

void CreateStridedSliceOp(Program& p, const std::shared_ptr<ngraph::op::v1::StridedSlice>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    do {
        auto data_output = op->input_value(0);
        auto begin_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
        auto end_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(2).get_node_shared_ptr());
        auto stride_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(3).get_node_shared_ptr());

        auto partial_input_shape = op->get_input_partial_shape(0);

        if (!begin_node || !end_node || !stride_node || partial_input_shape.is_dynamic()) {
            break;
        }

        auto input_shape = op->get_input_shape(0);
        auto output_shape = op->get_output_shape(0);

        auto begin = begin_node->cast_vector<int64_t>();
        auto end = end_node->cast_vector<int64_t>();
        auto strides = stride_node->cast_vector<int64_t>();

        bool ones_stride = true;
        for (auto & s : strides) {
            if (s != 1)
                ones_stride = false;
        }

        if (!ones_stride)
            break;

        auto convert_to_set = [](const std::vector<int64_t> mask) {
            ngraph::AxisSet axis_set{};
            for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
                if (mask[i] == 1) {
                    axis_set.emplace(i);
                }
            }
            return axis_set;
        };

        auto shrink_axis_mask = convert_to_set(op->get_shrink_axis_mask());
        auto new_axis_mask = convert_to_set(op->get_new_axis_mask());
        auto ellipsis_mask = convert_to_set(op->get_ellipsis_mask());
        auto begin_mask = convert_to_set(op->get_begin_mask());
        auto end_mask = convert_to_set(op->get_end_mask());

        std::vector<size_t> reshape_pattern,
                            axes,
                            offset,
                            dim;

        size_t input_shape_idx = 0;
        uint64_t uniq_id = 0;
        for (size_t axis = 0; axis < begin.size(); ++axis) {
            // add dimensions hidden under the ellipsis mask if ellipsis mask is set
            if (ellipsis_mask.count(axis)) {
                // only one bit in ellipsis mask is allowed
                int num_new_axis_after_ellipses = 0;
                int num_input_axis_before_ellipses = 0;
                for (size_t i = 0; i < axis; ++i) {
                    if (!new_axis_mask.count(i))
                        num_input_axis_before_ellipses++;
                }
                for (size_t i = axis + 1; i < begin.size(); ++i) {
                    if (new_axis_mask.count(i))
                        num_new_axis_after_ellipses++;
                }

                // -1 because it's a position of ellipses
                unsigned long num_input_axis_after_ellipses = (begin.size() - axis - num_new_axis_after_ellipses - 1);
                unsigned long num_of_hidden_dims = input_shape.size() - num_input_axis_after_ellipses
                                                    - num_input_axis_before_ellipses;
                for (size_t i = 0; i < num_of_hidden_dims; ++i) {
                    axes.emplace_back(uniq_id);
                    uniq_id++;
                    reshape_pattern.emplace_back(input_shape[input_shape_idx]);
                    offset.emplace_back(0);

                    dim.emplace_back(input_shape[input_shape_idx]);
                    input_shape_idx++;
                }
            } else {
                // add new single dimension if new_axis_mask is set
                if (new_axis_mask.count(axis)) {
                    reshape_pattern.emplace_back(1);
                    dim.emplace_back(1);
                    offset.emplace_back(0);
                } else if (shrink_axis_mask.count(axis)) {
                    // skip this dimension if shrink_axis_mask is set (input_shape_idx++)
                    reshape_pattern.emplace_back(1);
                    dim.emplace_back(1);
                    int64_t lb = begin[axis];
                    if (lb < 0)
                        lb = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + lb,
                                        static_cast<int64_t>(0));
                    offset.emplace_back(begin_mask.count(axis) ? 0 : lb);
                    input_shape_idx++;
                } else {
                    // calculate dimension using begin, end, begin_mask, end_mask, stride
                    reshape_pattern.emplace_back(input_shape[input_shape_idx]);

                    int64_t lb = begin[axis];
                    int64_t ub = end[axis];

                    // convert negative indexes to positive
                    if (lb < 0)
                        lb = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + lb,
                                        static_cast<int64_t>(0));
                    if (ub < 0)
                        ub = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + ub,
                                        static_cast<int64_t>(0));

                    // apply restrictions when begin or end values more/less than max/min possible values.
                    lb = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), lb);
                    ub = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), ub);

                    offset.emplace_back(lb);

                    // set default value for stride or use given value
                    int64_t stride = 1;
                    if (strides.size() > axis)
                        stride = strides[axis];

                    int64_t dimension = 0;
                    if (stride < 0) {
                        // apply masks
                        if (begin_mask.count(axis))
                            lb = static_cast<int64_t>(input_shape[input_shape_idx]) - 1;
                        if (end_mask.count(axis))
                            ub = -1;

                        lb = std::min(lb, static_cast<int64_t>(input_shape[input_shape_idx]) - 1);
                        lb -= 1;  // we always get 1st element, so we need decrease range
                        if (ub <= lb)
                            dimension = (ub - lb) / stride + 1;
                    } else {
                        // apply masks
                        if (begin_mask.count(axis))
                            lb = 0;
                        if (end_mask.count(axis))
                            ub = static_cast<int64_t>(input_shape[input_shape_idx]);

                        lb += 1;  // we always get 1st element, so we need decrease range
                        if (ub >= lb)
                            dimension = (ub - lb) / stride + 1;
                    }

                    dim.emplace_back(dimension);
                    input_shape_idx++;
                }
                axes.emplace_back(uniq_id);
                uniq_id++;
            }
        }

        for (; input_shape_idx < input_shape.size(); ++input_shape_idx) {
            reshape_pattern.emplace_back(input_shape[input_shape_idx]);
            offset.emplace_back(0);
            dim.emplace_back(input_shape[input_shape_idx]);
            axes.emplace_back(uniq_id);
            uniq_id++;
        }

        if (axes.size() > 4) {
            break;
        }

        auto inPrimitive = inputPrimitives[0];
        // Reshape in case of new axis
        if (!new_axis_mask.empty()) {
            auto targetShape = CldnnTensorFromIEDims(reshape_pattern);
            auto reshapeInName = op->get_friendly_name() + "/Reshape_before";
            auto reshapePrim = cldnn::reshape(reshapeInName, inputPrimitives[0], targetShape);
            p.AddPrimitive(reshapePrim);
            p.AddInnerPrimitiveToProfiler(reshapeInName, layerName, op);
            inPrimitive = reshapeInName;
        }

        auto data_node_shape = data_output.get_shape();

        std::vector<cldnn::tensor::value_type> offset_tensor{ 0, 0, 0, 0 };
        for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] < 0 || axes[i] > 3) {
                THROW_IE_EXCEPTION << "Invalid crop axis: " << std::to_string(axes[i]) << " in op " + op->get_friendly_name();
            }
            offset_tensor[axes[i]] = offset[i];
        }

        ngraph::Shape crop_shape(reshape_pattern);
        for (size_t i = 0; i < axes.size(); ++i) {
            crop_shape[axes[i]] = dim[i];
        }


        cldnn::tensor refSize = CldnnTensorFromIEDims(crop_shape);
        cldnn::tensor offSize = CldnnTensorFromIEDims(offset, 0);


        auto cropPrim = cldnn::crop(layerName, inPrimitive, refSize, offSize);
        p.AddPrimitive(cropPrim);
        p.AddPrimitiveToProfiler(layerName, op);

        // Reshape in case of deleting of axis
        if (!shrink_axis_mask.empty()) {
            auto targetShape = CldnnTensorFromIEDims(output_shape);
            auto reshapeOutName = op->get_friendly_name() + "/Crop";
            auto reshapePrim = cldnn::reshape(reshapeOutName, layerName, targetShape);
            p.AddPrimitive(reshapePrim);
            p.AddInnerPrimitiveToProfiler(reshapeOutName, layerName, op);
        }
        return;
    } while (false);

    auto end_mask_ = op->get_end_mask();
    auto begin_mask_ = op->get_begin_mask();
    auto new_axis_mask_ = op->get_new_axis_mask();
    auto shrink_axis_mask_ = op->get_shrink_axis_mask();
    std::vector<uint8_t> begin_mask(begin_mask_.begin(), begin_mask_.end());
    std::vector<uint8_t> end_mask(end_mask_.begin(), end_mask_.end());
    std::vector<uint8_t> new_axis_mask(new_axis_mask_.begin(), new_axis_mask_.end());
    std::vector<uint8_t> shrink_axis_mask(shrink_axis_mask_.begin(), shrink_axis_mask_.end());

    // Plugin requires inverted mask values. Consider changing primitive impl to be aligned with the spec.
    for (auto& b : begin_mask) {
        b = 1 - b;
    }
    for (auto& e : end_mask) {
        e = 1 - e;
    }

    auto out_size = CldnnTensorFromIEDims(op->get_output_shape(0));

    auto stridedSlicePrim = cldnn::strided_slice(layerName,
                                                 inputPrimitives[0],
                                                 inputPrimitives[1],
                                                 inputPrimitives[2],
                                                 inputPrimitives[3],
                                                 begin_mask,
                                                 end_mask,
                                                 new_axis_mask,
                                                 shrink_axis_mask,
                                                 out_size);

    p.AddPrimitive(stridedSlicePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, StridedSlice);

}  // namespace CLDNNPlugin
