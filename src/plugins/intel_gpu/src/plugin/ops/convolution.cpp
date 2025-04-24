// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/convolution.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov {
namespace op {
namespace internal {
using Convolution = ov::intel_gpu::op::Convolution;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {


static void CreateConvolutionOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::Convolution>& op) {
    validate_inputs_count(op, {3, 6});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDims = op->get_output_partial_shape(0);

    cldnn::primitive_id weights = inputs[op::Convolution::Args::WEIGHTS].pid;
    const uint32_t groups = std::max<int64_t>(op->get_groups(), 1);
    const bool weights_have_group_dim = op->get_groups() > 0;

    auto strides = op->get_strides();
    auto dilations = op->get_dilations();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    auto auto_pad = op->get_auto_pad();

    if (!op->is_dynamic() && !p.use_new_shape_infer()) {
        // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
        strides.resize(std::max<size_t>(2, strides.size()), 1);
        dilations.resize(std::max<size_t>(2, strides.size()), 1);
        pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
        pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);
    }

    std::shared_ptr<cldnn::convolution> prim = nullptr;

    if (op->is_asymmetric()) {
        auto azp = inputs[op::Convolution::Args::AZP];
        auto wzp = inputs[op::Convolution::Args::WZP];
        auto compensation = inputs[op::Convolution::Args::COMPENSATION];
        prim = std::make_shared<cldnn::convolution>(layerName,
                                                    inputs[op::Convolution::Args::INPUT],
                                                    weights,
                                                    "",
                                                    wzp.pid,
                                                    azp.pid,
                                                    compensation.pid,
                                                    groups,
                                                    strides,
                                                    dilations,
                                                    pads_begin,
                                                    pads_end,
                                                    weights_have_group_dim,
                                                    op->get_output_element_type(0),
                                                    auto_pad);
    } else {
        prim = std::make_shared<cldnn::convolution>(layerName,
                                                    inputs[op::Convolution::Args::INPUT],
                                                    weights,
                                                    "",
                                                    groups,
                                                    strides,
                                                    dilations,
                                                    pads_begin,
                                                    pads_end,
                                                    weights_have_group_dim,
                                                    auto_pad);
    }

    p.add_primitive(*op, prim);
}

static void CreateConvolutionBackpropDataOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ConvolutionBackpropData>& op) {
    // 3rd input is an optional output shape
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            OPENVINO_THROW("Unsupported dilation in ConvolutionBackpropData ", op->get_friendly_name());
        }
    }

    auto weightsName = inputs[1];
    auto weights_node = op->get_input_node_shared_ptr(1);
    // WA: For the cases like Const(weights)->Sub(zp)->Deconv. And also for the cases with real runtime weights.
    // Dimensions order of weights blob is IOYX, but
    // the selected format is OIYX by default. So we need to swap (and transpose) I and O dimensions to match the format
    // For Constant node on input transpose is not needed, because the data is transposed on const node creation
    {
        std::string permuteName = layerName + "_cldnn_weights_permute";
        auto weights_rank = op->get_input_shape(1).size();
        std::vector<uint16_t> permute_order(weights_rank);
        std::iota(std::begin(permute_order), std::end(permute_order), 0);
        // Should be 1, 0, 2, 3 {, 4} to swap O and I
        std::swap(permute_order[1], permute_order[0]);
        auto permutePrim = cldnn::permute(permuteName,
                                          weightsName,
                                          permute_order);

        p.add_primitive(*op, permutePrim);

        weightsName.pid = permuteName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName.pid};
    const bool weights_have_group_dim = false;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    auto output_padding = op->get_output_padding();

    if (!op->is_dynamic() && !p.use_new_shape_infer()) {
        // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
        strides.resize(std::max<size_t>(2, strides.size()), 1);
        dilations.resize(std::max<size_t>(2, strides.size()), 1);
        pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
        auto deconvPrim = cldnn::deconvolution(layerName,
                                               inputs[0],
                                               weights,
                                               {},
                                               1,
                                               strides,
                                               pads_begin,
                                               dilations,
                                               tensor_from_dims(op->get_output_tensor(0).get_shape()),
                                               weights_have_group_dim);
        p.add_primitive(*op, deconvPrim);
    } else {
        auto deconvPrim = cldnn::deconvolution(layerName,
                                               inputs[0],
                                               weights,
                                               {},
                                               1,
                                               strides,
                                               pads_begin,
                                               dilations,
                                               pads_begin,
                                               pads_end,
                                               output_padding,
                                               weights_have_group_dim);
        if (op->get_input_size() == 3) {
            auto output_shape_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (output_shape_constant) {
                auto output_shape = output_shape_constant->cast_vector<int64_t>();
                ov::Shape shape(output_shape.begin(), output_shape.end());
                ov::PartialShape output_pshape(shape);
                deconvPrim.output_partial_shape = output_pshape;
            } else {
                deconvPrim.output_shape_id = inputs[2].pid;
            }
        }
        p.add_primitive(*op, deconvPrim);
    }
}

static void CreateGroupConvolutionBackpropDataOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData>& op) {
    // 3rd input is an optional output shape
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            OPENVINO_THROW("Unsupported dilation in GroupConvolutionBackpropData ", op->get_friendly_name());
        }
    }

    uint32_t groups = static_cast<uint32_t>(op->get_input_shape(1)[0]);

    auto weightsName = inputs[1];
    auto weights_node = op->get_input_node_shared_ptr(1);
    // WA: For the cases like Const(weights)->Sub(zp)->Deconv. And also for the cases with real runtime weights.
    // Dimensions order of weights blob is IOYX, but
    // the selected format is OIYX by default. So we need to swap I and O dimensions to match the format.
    // For Constant node on input transpose is not needed, because the data is transposed on const node creation
    {
        std::string permuteName = layerName + "_cldnn_weights_permute";
        auto weights_rank = op->get_input_shape(1).size();
        std::vector<uint16_t> permute_order(weights_rank);
        std::iota(std::begin(permute_order), std::end(permute_order), 0);
        // Should be 0, 2, 1, 3, 4 {, 5} to swap O and I
        std::swap(permute_order[2], permute_order[1]);
        auto permutePrim = cldnn::permute(permuteName,
                                          weightsName,
                                          permute_order);

        p.add_primitive(*op, permutePrim);

        weightsName.pid = permuteName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName.pid};
    const bool weights_have_group_dim = true;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    auto output_padding = op->get_output_padding();

    if (!op->is_dynamic()) {
        // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
        strides.resize(std::max<size_t>(2, strides.size()), 1);
        dilations.resize(std::max<size_t>(2, strides.size()), 1);
        pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);

        auto deconvPrim = cldnn::deconvolution(layerName,
                                               inputs[0],
                                               weights,
                                               {},
                                               groups,
                                               strides,
                                               pads_begin,
                                               dilations,
                                               tensor_from_dims(op->get_output_tensor(0).get_shape()),
                                               weights_have_group_dim);
        p.add_primitive(*op, deconvPrim);
    } else {
        auto deconvPrim = cldnn::deconvolution(layerName,
                                               inputs[0],
                                               weights,
                                               {},
                                               groups,
                                               strides,
                                               pads_begin,
                                               dilations,
                                               pads_begin,
                                               pads_end,
                                               output_padding,
                                               weights_have_group_dim);
        if (op->get_input_size() == 3) {
            auto output_shape_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (output_shape_constant) {
                auto output_shape = output_shape_constant->cast_vector<int64_t>();
                ov::Shape shape(output_shape.begin(), output_shape.end());
                ov::PartialShape output_pshape(shape);
                deconvPrim.output_partial_shape = output_pshape;
            } else {
                deconvPrim.output_shape_id = inputs[2].pid;
            }
        }
        p.add_primitive(*op, deconvPrim);
    }
}

static void DeformableConvolutionImpl(ProgramBuilder& p,
                                      const std::shared_ptr<ov::Node>& op,
                                      const int64_t groups,
                                      const ov::Strides& strides,
                                      const ov::Strides& dilations,
                                      const ov::CoordinateDiff& padding,
                                      std::int64_t deformable_groups_num,
                                      bool bilinear_interpolation_pad = false) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto outDims = op->get_output_shape(0);

    cldnn::primitive_id weights = inputs[2].pid;
    // Remove weights from inputs
    inputs.erase(inputs.begin() + 2);
    auto convPrim = cldnn::convolution(layerName,
                                       inputs,
                                       weights,
                                       "",
                                       true,
                                       groups,
                                       deformable_groups_num,
                                       strides,
                                       dilations,
                                       padding,
                                       padding,
                                       bilinear_interpolation_pad);

    p.add_primitive(*op, convPrim);
}

static void CreateDeformableConvolutionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::DeformableConvolution>& op) {
    validate_inputs_count(op, {3});
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    DeformableConvolutionImpl(p, op, op->get_group(), strides, dilations, pads_begin, op->get_deformable_group());
}

static void CreateDeformableConvolutionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::DeformableConvolution>& op) {
    validate_inputs_count(op, {3, 4});
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    DeformableConvolutionImpl(p,
                              op,
                              op->get_group(),
                              strides,
                              dilations,
                              pads_begin,
                              op->get_deformable_group(),
                              op->get_bilinear_interpolation_pad());
}

REGISTER_FACTORY_IMPL(internal, Convolution);
REGISTER_FACTORY_IMPL(v1, ConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, GroupConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, DeformableConvolution);
REGISTER_FACTORY_IMPL(v8, DeformableConvolution);

}  // namespace ov::intel_gpu
