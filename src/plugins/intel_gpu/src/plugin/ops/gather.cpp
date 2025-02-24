// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "ov_ops/gather_compressed.hpp"

#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/crop.hpp"

namespace ov {
namespace op {
namespace internal {
using GatherCompressed = ov::op::internal::GatherCompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

template <typename T>
void CreateGatherOpBase(ProgramBuilder& p, const std::shared_ptr<T>& op, const int64_t batch_dim = 0, bool support_neg_ind = false,
                        bool weights_compressed = false) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = op->get_axis();

    // Dynamic path will do shape infer internally, so no need to pass valid out shape for that case
    bool is_static = op->get_output_partial_shape(0).is_static();
    ov::Shape out_shape = is_static ? op->get_output_shape(0) : ov::Shape{};

    // Update output_shape in case of scalar indice
    auto out_shape_original = out_shape;

    // WA for NMS->Gather construction. NMS fills part of the output blob by the -1 if these values
    // must not be taken into account.
    // CPU also uses this like of WA.
    if (support_neg_ind) {
        const auto& rti = op->get_rt_info();
        const auto& reverse = rti.find("dontReverseIndices");
        if (reverse != rti.end()) {
            support_neg_ind = false;
        }
    }

    // Set layer name for Gather
    auto reshapeName = layerName + "";

    // Check if Gather could be converted to other primitive
    const auto input_shape = op->get_input_partial_shape(0);
    const auto input_rank = input_shape.rank().get_length();
    const auto& indices = op->input_value(1);
    const auto& indices_node = indices.get_node_shared_ptr();
    auto indices_constant = ov::as_type_ptr<ov::op::v0::Constant>(indices_node);
    bool is_indices_constant = (indices_constant) ? true : false;

    if (is_static && axis == 0 && input_rank > 1 && indices.get_partial_shape().rank().get_length() == 0 &&
        std::equal(input_shape.begin()+1, input_shape.end(), out_shape.begin()+1) && is_indices_constant) {
        // Gather -> Crop
        // this Gather simply divides an input tensor along Batch axis
        auto get_crop_layer_name = [&](std::string name, size_t idx)->std::string {
            return (name + "/crop_" + std::to_string(idx));
        };

        // Get indices info to calculate offset
        float result = 0.f;
        OPENVINO_ASSERT(ov::op::util::get_single_value(indices_constant, result),
                        "Unsupported indices node in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        // Set tensors for crop shape and offset
        ov::Shape start_offset(input_shape.size());
        int32_t new_offset0 = static_cast<int32_t>(result);
        if (support_neg_ind && new_offset0 < 0) {
            // According to Gather-8,
            // Negative values of indices indicate reverse indexing from data.shape[axis]
            new_offset0 += static_cast<int32_t>(input_shape.get_shape()[axis]);
        }

        start_offset[0] = static_cast<size_t>(new_offset0);
        auto offsetTensor = tensor_from_dims(start_offset, 0);
        auto outTensor = tensor_from_dims(out_shape, 1);

        // Create Crop
        layerName = get_crop_layer_name(layerName, static_cast<size_t>(result));
        auto cropPrim = cldnn::crop(layerName, inputs[0], outTensor, offsetTensor);
        p.add_primitive(*op, cropPrim);
    } else {
        if (!weights_compressed) {
            auto gatherPrim = cldnn::gather(layerName,
                                            inputs[0],
                                            inputs[1],
                                            axis,
                                            input_rank,
                                            out_shape,
                                            batch_dim,
                                            support_neg_ind);
            p.add_primitive(*op, gatherPrim);
        } else {
            float zp_value = 0.0f;
            bool has_scalar_zp = false;
            if (op->get_input_size() == 5) {
                std::shared_ptr<ov::op::v0::Constant> zp_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(4));
                if (zp_const && ov::shape_size(zp_const->get_output_shape(0)) == 1) {
                    has_scalar_zp = true;
                    zp_value = zp_const->cast_vector<float>()[0];
                }
            }

            std::shared_ptr<ov::op::internal::GatherCompressed> op_compressed = std::dynamic_pointer_cast<ov::op::internal::GatherCompressed>(op);

            auto gatherPrim = cldnn::gather(layerName,
                                            inputs[0],
                                            inputs[1],
                                            axis,
                                            inputs[3],
                                            (has_scalar_zp || op->get_input_size() == 4) ? cldnn::input_info() : inputs[4],
                                            op_compressed->get_output_element_type(0),
                                            input_rank,
                                            out_shape,
                                            batch_dim,
                                            support_neg_ind);

            if (has_scalar_zp) {
                gatherPrim.decompression_zero_point_scalar = zp_value;
            }

            p.add_primitive(*op, gatherPrim);
        }
    }
}

static void CreateGatherOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Gather>& op) {
    validate_inputs_count(op, {2, 3});
    CreateGatherOpBase<ov::op::v1::Gather>(p, op);
}

REGISTER_FACTORY_IMPL(v1, Gather);

static void CreateGatherOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v7::Gather>& op) {
    validate_inputs_count(op, {2, 3, 4});
    CreateGatherOpBase<ov::op::v7::Gather>(p, op, op->get_batch_dims());
}

REGISTER_FACTORY_IMPL(v7, Gather);

static void CreateGatherOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::Gather>& op) {
    validate_inputs_count(op, {2, 3, 4});
    CreateGatherOpBase<ov::op::v8::Gather>(p, op, op->get_batch_dims(), true);
}

REGISTER_FACTORY_IMPL(v8, Gather);

static void CreateGatherCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherCompressed>& op) {
    validate_inputs_count(op, {4, 5});
    CreateGatherOpBase<ov::op::internal::GatherCompressed>(p, op, op->get_batch_dims(), true, true);
}

REGISTER_FACTORY_IMPL(internal, GatherCompressed);

}  // namespace ov::intel_gpu
