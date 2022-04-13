// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/op/util/op_types.hpp"

#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/binary_convolution.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateGroupConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::GroupConvolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t groups = op->get_input_shape(1)[0];
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    const bool weights_have_group_dim = true;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       groups,
                                       strides,
                                       pads_begin,
                                       dilations,
                                       tensor_from_dims(outDims),
                                       DataTypeFromPrecision(outPrecision),
                                       weights_have_group_dim,
                                       op->get_friendly_name());

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::Convolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    const bool weights_have_group_dim = false;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       1,
                                       strides,
                                       pads_begin,
                                       dilations,
                                       tensor_from_dims(outDims),
                                       DataTypeFromPrecision(outPrecision),
                                       weights_have_group_dim,
                                       op->get_friendly_name());

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateConvolutionBackpropDataOp(Program& p, const std::shared_ptr<ngraph::op::v1::ConvolutionBackpropData>& op) {
    // 3rd input is an optional output shape
    p.ValidateInputs(op, {2, 3});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            IE_THROW() << "Unsupported dilation in ConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    auto weightsName = inputs[1];
    auto weights_node = op->get_input_node_shared_ptr(1);
    bool hasConstantWeights = IsNodeOnConstPath(weights_node);
    // WA: For the cases like Const(weights)->Sub(zp)->Deconv. And also for the cases with real runtime weights.
    // Dimensions order of weights blob is IOYX, but
    // the selected format is OIYX by default. So we need to swap (and transpose) I and O dimensions to match the format
    // For Constant node on input transpose is not needed, because the data is transposed on const node creation
    if ((hasConstantWeights && std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_node) == nullptr) || !hasConstantWeights) {
        std::string permuteName = layerName + "_cldnn_weights_permute";
        auto weights_rank = op->get_input_shape(1).size();
        std::vector<uint16_t> permute_order(weights_rank);
        std::iota(std::begin(permute_order), std::end(permute_order), 0);
        // Should be 1, 0, 2, 3 {, 4} to swap O and I
        std::swap(permute_order[1], permute_order[0]);
        auto permutePrim = cldnn::permute(permuteName,
                                          weightsName,
                                          permute_order,
                                          op->get_friendly_name());

        p.AddPrimitive(permutePrim);
        p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);

        weightsName = permuteName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};
    const bool weights_have_group_dim = false;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);

    auto deconvPrim = cldnn::deconvolution(layerName,
                                           inputs[0],
                                           weights,
                                           {},
                                           1,
                                           strides,
                                           pads_begin,
                                           tensor_from_dims(op->get_output_tensor(0).get_shape()),
                                           weights_have_group_dim,
                                           op->get_friendly_name());

    p.AddPrimitive(deconvPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateGroupConvolutionBackpropDataOp(Program& p, const std::shared_ptr<ngraph::op::v1::GroupConvolutionBackpropData>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            IE_THROW() << "Unsupported dilation in GroupConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    uint32_t groups = op->get_input_shape(1)[0];

    auto weightsName = inputs[1];
    auto weights_node = op->get_input_node_shared_ptr(1);
    bool hasConstWeights = IsNodeOnConstPath(weights_node);
    // WA: For the cases like Const(weights)->Sub(zp)->Deconv. And also for the cases with real runtime weights.
    // Dimensions order of weights blob is IOYX, but
    // the selected format is OIYX by default. So we need to swap I and O dimensions to match the format.
    // For Constant node on input transpose is not needed, because the data is transposed on const node creation
    if ((hasConstWeights && std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_node) == nullptr) || !hasConstWeights) {
        std::string permuteName = layerName + "_cldnn_weights_permute";
        auto weights_rank = op->get_input_shape(1).size();
        std::vector<uint16_t> permute_order(weights_rank);
        std::iota(std::begin(permute_order), std::end(permute_order), 0);
        // Should be 0, 2, 1, 3, 4 {, 5} to swap O and I
        std::swap(permute_order[2], permute_order[1]);
        auto permutePrim = cldnn::permute(permuteName,
                                          weightsName,
                                          permute_order,
                                          op->get_friendly_name());

        p.AddPrimitive(permutePrim);
        p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);

        weightsName = permuteName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};
    const bool weights_have_group_dim = true;

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);

    auto deconvPrim = cldnn::deconvolution(layerName,
                                           inputs[0],
                                           weights,
                                           {},
                                           groups,
                                           strides,
                                           pads_begin,
                                           tensor_from_dims(op->get_output_tensor(0).get_shape()),
                                           weights_have_group_dim,
                                           op->get_friendly_name());

    p.AddPrimitive(deconvPrim);
    p.AddPrimitiveToProfiler(op);
}

static void DeformableConvolutionImpl(Program& p,
                                      const std::shared_ptr<ngraph::Node>& op,
                                      const int64_t groups,
                                      const ov::Strides& strides,
                                      const ov::Strides& dilations,
                                      const ov::CoordinateDiff& padding,
                                      std::int64_t deformableGroupsNum,
                                      bool bilinearInterpolationPad = false) {
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto outDims = op->get_output_shape(0);

    std::vector<cldnn::primitive_id> weights = {inputs[2]};
    // Remove weights from inputs
    inputs.erase(inputs.begin() + 2);
    if (groups == 1) {
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
                                                          bilinearInterpolationPad,
                                                          op->get_friendly_name());
        p.AddPrimitive(defConvPrimInterp);
        p.AddInnerPrimitiveToProfiler(defConvLayerNameInterp, defConvLayerNameConv, op);
        auto defConvPrim = cldnn::deformable_conv(defConvLayerNameConv,
                                                  defConvLayerNameInterp,
                                                  weights,
                                                  {},
                                                  groups,
                                                  tensor_from_dims(outDims),
                                                  op->get_friendly_name());
        p.AddPrimitive(defConvPrim);
        p.AddPrimitiveToProfiler(defConvLayerNameConv, op);
    } else {
        auto convPrim = cldnn::convolution(layerName,
                                           inputs,
                                           weights,
                                           {},
                                           groups,
                                           deformableGroupsNum,
                                           strides,
                                           padding,
                                           dilations,
                                           tensor_from_dims(outDims),
                                           bilinearInterpolationPad,
                                           op->get_friendly_name());

        p.AddPrimitive(convPrim);
        p.AddPrimitiveToProfiler(op);
    }
}

static void CreateDeformableConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::DeformableConvolution>& op) {
    p.ValidateInputs(op, {3});
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    DeformableConvolutionImpl(p, op, op->get_group(), strides, dilations, pads_begin, op->get_deformable_group());
}

static void CreateDeformableConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v8::DeformableConvolution>& op) {
    p.ValidateInputs(op, {3, 4});
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

static void CreateBinaryConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDims = op->get_output_shape(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    cldnn::data_types calc_precision = DataTypeFromPrecision(op->get_output_element_type(0));

    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    auto convPrim = cldnn::binary_convolution(layerName,
                                              inputs[0],
                                              weights,
                                              strides,
                                              pads_begin,
                                              dilations,
                                              tensor_from_dims(outDims),
                                              1,
                                              op->get_pad_value(),
                                              calc_precision,
                                              op->get_friendly_name());

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, GroupConvolution);
REGISTER_FACTORY_IMPL(v1, Convolution);
REGISTER_FACTORY_IMPL(v1, ConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, GroupConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, DeformableConvolution);
REGISTER_FACTORY_IMPL(v8, DeformableConvolution);
REGISTER_FACTORY_IMPL(v1, BinaryConvolution);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
