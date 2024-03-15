// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/deconvolution.hpp"

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
using Deconvolution = ov::intel_gpu::op::Deconvolution;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {


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

    if (!op->is_dynamic()) {
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

static void CreateDeconvolutionOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::Deconvolution>& op) {
    // 3rd input is an optional output shape
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    auto input_size = op->get_input_size();
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            OPENVINO_THROW("Unsupported dilation in ConvolutionBackpropData ", op->get_friendly_name());
        }
    }

    const uint32_t groups = std::max<int64_t>(op->get_groups(), 1);
    const bool weights_have_group_dim = op->get_groups() > 0;

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
        if (weights_have_group_dim)
                std::swap(permute_order[2], permute_order[1]);
        else
                std::swap(permute_order[1], permute_order[0]);

        auto permutePrim = cldnn::permute(permuteName,
                                          weightsName,
                                          permute_order);

        p.add_primitive(*op, permutePrim);

        weightsName.pid = permuteName;
    }

    cldnn::primitive_id weights = inputs[op::Deconvolution::Args::WEIGHTS].pid;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    auto output_padding = op->get_output_padding();

    if (!op->is_dynamic()) {
        // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
        strides.resize(std::max<size_t>(2, strides.size()), 1);
        dilations.resize(std::max<size_t>(2, strides.size()), 1);
        pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
        pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);
    }

    std::shared_ptr<cldnn::deconvolution> prim = nullptr;
    if (input_size == 3) {
        prim = std::make_shared<cldnn::deconvolution>(layerName,
                                                    inputs[op::Deconvolution::Args::INPUT],
                                                    weightsName.pid,
                                                    "",
                                                    groups,
                                                    strides,
                                                    pads_begin,
                                                    pads_end,
                                                    dilations,
                                                    output_padding,
                                                    weights_have_group_dim,
                                                    op->get_output_element_type(0));
    } else {
        cldnn::primitive_id output_shape_in = inputs[op::Deconvolution::Args::OUTPUT_SHAPE].pid;

        prim = std::make_shared<cldnn::deconvolution>(layerName,
                                                    inputs[op::Deconvolution::Args::INPUT],
                                                    weightsName.pid,
                                                    "",
                                                    output_shape_in,
                                                    groups,
                                                    strides,
                                                    pads_begin,
                                                    pads_end,
                                                    dilations,
                                                    output_padding,
                                                    weights_have_group_dim,
                                                    op->get_output_element_type(0));
    }

    p.add_primitive(*op, prim);
}

static void DeformableConvolutionImpl(ProgramBuilder& p,
                                      const std::shared_ptr<ov::Node>& op,
                                      const int64_t groups,
                                      const ov::Strides& strides,
                                      const ov::Strides& dilations,
                                      const ov::CoordinateDiff& padding,
                                      std::int64_t deformableGroupsNum,
                                      bool bilinearInterpolationPad = false) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto outDims = op->get_output_shape(0);

    cldnn::primitive_id weights = inputs[2].pid;
    // Remove weights from inputs
    inputs.erase(inputs.begin() + 2);
    auto device_info = p.get_engine().get_device_info();
    bool supports_subgroups = device_info.supports_khr_subgroups || device_info.supports_intel_subgroups;
    if (groups == 1 && supports_subgroups) {
        std::string defConvLayerNameInterp = layerName + "_interp";
        std::string defConvLayerNameConv = layerName;
        cldnn::tensor kernel;
        auto weights_shape = op->get_input_shape(2);
        size_t nonSpatialDimsNum = 2;
        if (weights_shape.size() == 3) {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[nonSpatialDimsNum + 2],
                                                  weights_shape[nonSpatialDimsNum + 1],
                                                  weights_shape[nonSpatialDimsNum + 0]));
        } else {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[nonSpatialDimsNum + 1],
                                                  weights_shape[nonSpatialDimsNum + 0],
                                                  1));
        }

        auto defConvPrimInterp = cldnn::deformable_interp(defConvLayerNameInterp,
                                                          inputs,
                                                          groups,
                                                          deformableGroupsNum,
                                                          strides,
                                                          padding,
                                                          dilations,
                                                          tensor_from_dims(outDims),
                                                          kernel,
                                                          bilinearInterpolationPad);
        p.add_primitive(*op, defConvPrimInterp);
        auto defConvPrim = cldnn::deformable_conv(defConvLayerNameConv,
                                                  defConvLayerNameInterp,
                                                  { weights },
                                                  {},
                                                  groups,
                                                  tensor_from_dims(outDims));
        p.add_primitive(*op, defConvPrim);
    } else {
        auto convPrim = cldnn::convolution(layerName,
                                           inputs,
                                           weights,
                                           "",
                                           true,
                                           groups,
                                           deformableGroupsNum,
                                           strides,
                                           dilations,
                                           padding,
                                           padding,
                                           bilinearInterpolationPad);

        p.add_primitive(*op, convPrim);
    }
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
REGISTER_FACTORY_IMPL(internal, Deconvolution);
REGISTER_FACTORY_IMPL(v1, DeformableConvolution);
REGISTER_FACTORY_IMPL(v8, DeformableConvolution);

}  // namespace intel_gpu
}  // namespace ov
