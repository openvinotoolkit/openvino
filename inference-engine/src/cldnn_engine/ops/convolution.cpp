// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/op/util/op_types.hpp"

#include "api/convolution.hpp"
#include "api/deconvolution.hpp"
#include "api/binary_convolution.hpp"
#include "api/reshape.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

struct ConvoltuionParameters {
    cldnn::tensor stride;
    cldnn::tensor padding;
    cldnn::tensor dilation;
    uint32_t groups;
};

static ConvoltuionParameters GetConvolutionParameters(const ngraph::CoordinateDiff& pads_begin,
                                                      const ngraph::Strides& dilations,
                                                      const ngraph::Strides& strides,
                                                      uint32_t groups) {
    cldnn::tensor stride, padding, dilation;
    if (pads_begin.size() != strides.size() || dilations.size() != strides.size())
        THROW_IE_EXCEPTION << "Strides, Dilations and Pads are supposed to have the same elements count";

    switch (strides.size()) {
        case 3: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[2], strides[1], strides[0]));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[2], -pads_begin[1], -pads_begin[0]));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[2], dilations[1], dilations[0]));
            break;
        }
        case 2: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[1], strides[0], 1));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[1], -pads_begin[0], 0));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[1], dilations[0], 1));
            break;
        }
        case 1: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[0], 1, 1));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[0], 0, 0));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[0], 1, 1));
            break;
        }
        default: THROW_IE_EXCEPTION << "Unsupported convolve parameters size. Only 1d, 2d, and 3d cases are supported";
    }

    return {stride, padding, dilation, groups};
}

void CreateGroupConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::GroupConvolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t groups = op->get_input_shape(1)[0];
    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), groups);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    auto weightsName = inputs[1];

    // WA: For the case with FakeQuantize op on weights that are not folderd by constant propagation pass for some reason.
    // Dimensions order is GOIYZ, but
    // the selected format is OIZYX by default.
    if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1)) == nullptr) {
        std::string reshapeName = layerName + "_cldnn_weights_reshape";
        std::string reorderName = layerName + "_cldnn_weights_reorder";

        auto weights_shape = op->get_input_shape(1);
        std::vector<size_t> new_weights_shape;
        new_weights_shape.push_back(weights_shape[0] * weights_shape[1]); // Merged G and O dims
        for (size_t i = 2; i < weights_shape.size(); i++) {
            new_weights_shape.push_back(weights_shape[i]);
        }
        auto reshapePrim = cldnn::reshape(reshapeName,
                                          weightsName,
                                          CldnnTensorFromIEDims(new_weights_shape));

        p.AddPrimitive(reshapePrim);
        p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

        auto reorderPrim = cldnn::reorder(reorderName,
                                        reshapeName,
                                        DefaultFormatForDims(new_weights_shape.size()),
                                        DataTypeFromPrecision(op->get_input_element_type(1)));

        p.AddPrimitive(reorderPrim);
        p.AddInnerPrimitiveToProfiler(reorderName, layerName, op);

        weightsName = reorderName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};
    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       params.groups,
                                       params.stride,
                                       params.padding,
                                       params.dilation,
                                       CldnnTensorFromIEDims(outDims),
                                       DataTypeFromPrecision(outPrecision));

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::Convolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       params.groups,
                                       params.stride,
                                       params.padding,
                                       params.dilation,
                                       CldnnTensorFromIEDims(outDims),
                                       DataTypeFromPrecision(outPrecision));

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateConvolutionBackpropDataOp(Program& p, const std::shared_ptr<ngraph::op::v1::ConvolutionBackpropData>& op) {
    // 3rd input is an optional output shape
    p.ValidateInputs(op, {2, 3});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            THROW_IE_EXCEPTION << "Unsupported dilation in ConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    auto weightsName = inputs[1];
    // WA: For the case with FakeQuantize op on weights that are not folderd by constant propagation pass for some reason.
    // Dimensions order of weights blob is IOYX, but
    // the selected format is OIYX by default. So we need to swap I and O dimensions to match the format
    if (IsNodeOnConstPath(op->get_input_node_shared_ptr(1))) {
        std::string reshapeName = layerName + "_cldnn_weights_reshape";

        auto weights_shape = op->get_input_shape(1);
        std::swap(weights_shape[0], weights_shape[1]);
        auto reshapePrim = cldnn::reshape(reshapeName,
                                          weightsName,
                                          CldnnTensorFromIEDims(weights_shape));

        p.AddPrimitive(reshapePrim);
        p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

        weightsName = reshapeName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);
    auto deconvPrim = cldnn::deconvolution(layerName,
        inputs[0],
        weights,
        {},
        params.groups,
        params.stride,
        params.padding,
        CldnnTensorFromIEDims(op->get_output_tensor(0).get_shape()));
    p.AddPrimitive(deconvPrim);

    p.AddPrimitiveToProfiler(op);
}

void CreateGroupConvolutionBackpropDataOp(Program& p, const std::shared_ptr<ngraph::op::v1::GroupConvolutionBackpropData>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            THROW_IE_EXCEPTION << "Unsupported dilation in ConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    uint32_t groups = op->get_input_shape(1)[0];
    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), groups);
    std::vector<cldnn::primitive_id> weights = {inputs[1]};

    auto deconvPrim = cldnn::deconvolution(layerName,
        inputs[0],
        weights,
        {},
        params.groups,
        params.stride,
        params.padding,
        CldnnTensorFromIEDims(op->get_output_tensor(0).get_shape()));
    p.AddPrimitive(deconvPrim);

    p.AddPrimitiveToProfiler(op);
}

void CreateDeformableConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::DeformableConvolution>& op) {
    p.ValidateInputs(op, {3});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), op->get_group());
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[2]};
    if (params.groups > 1) {
        auto convPrim = cldnn::convolution(layerName,
                                           inputs[0],
                                           inputs[1],
                                           weights,
                                           {},
                                           params.groups,
                                           op->get_deformable_group(),
                                           params.stride,
                                           params.padding,
                                           params.dilation,
                                           CldnnTensorFromIEDims(outDims));

        p.AddPrimitive(convPrim);
        p.AddPrimitiveToProfiler(op);
    } else {
        std::string defConvLayerNameInterp = layerName + "_interp";
        std::string defConvLayerNameConv = layerName;
        cldnn::tensor kernel;
        auto weights_shape = op->get_input_shape(2);
        size_t sidx = 2 + (params.groups > 1 ? 1 : 0);
        if (weights_shape.size() == 3) {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[sidx + 2],
                                                  weights_shape[sidx + 1],
                                                  weights_shape[sidx + 0]));
        } else {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[sidx + 1],
                                                  weights_shape[sidx + 0],
                                                  1));
        }

        auto defConvPrimInterp = cldnn::deformable_interp(defConvLayerNameInterp,
                                                          inputs[0],
                                                          inputs[1],
                                                          params.groups,
                                                          op->get_deformable_group(),
                                                          params.stride,
                                                          params.padding,
                                                          params.dilation,
                                                          CldnnTensorFromIEDims(outDims),
                                                          kernel);
        p.AddPrimitive(defConvPrimInterp);
        p.AddInnerPrimitiveToProfiler(defConvLayerNameInterp, defConvLayerNameConv, op);
        auto defConvPrim = cldnn::deformable_conv(defConvLayerNameConv,
                                                  defConvLayerNameInterp,
                                                  weights,
                                                  {},
                                                  params.groups,
                                                  CldnnTensorFromIEDims(outDims));
        p.AddPrimitive(defConvPrim);
        p.AddPrimitiveToProfiler(defConvLayerNameConv, op);
    }
}

void CreateBinaryConvolutionOp(Program& p, const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op) {
    p.ValidateInputs(op, {2});
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    cldnn::data_types calc_precision = DataTypeFromPrecision(op->get_output_element_type(0));
    auto convPrim = cldnn::binary_convolution(layerName,
                                              inputs[0],
                                              weights,
                                              params.stride,
                                              params.padding,
                                              params.dilation,
                                              CldnnTensorFromIEDims(outDims),
                                              params.groups,
                                              op->get_pad_value(),
                                              calc_precision);

    p.AddPrimitive(convPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, GroupConvolution);
REGISTER_FACTORY_IMPL(v1, Convolution);
REGISTER_FACTORY_IMPL(v1, ConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, GroupConvolutionBackpropData);
REGISTER_FACTORY_IMPL(v1, DeformableConvolution);
REGISTER_FACTORY_IMPL(v1, BinaryConvolution);

}  // namespace CLDNNPlugin
