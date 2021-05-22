// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/squared_difference.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/op/util/op_types.hpp"

#include "api/data.hpp"

namespace CLDNNPlugin {

struct ConstProperties {
    bool isWeights;
    bool hasGroupDimension;
    bool reversedChannelsOrder;
};

static ConstProperties getConstProperties(const std::shared_ptr<ngraph::op::Constant>& op) {
    for (size_t i = 0; i < op->get_output_size(); i++) {
        auto outTensors = op->get_output_target_inputs(i);
        for (auto& t : outTensors) {
            auto outOp = t.get_node();
            if (dynamic_cast<ngraph::op::v1::Convolution*>(outOp)) {
                return {t.get_index() == 1, false, false};
            } else if (dynamic_cast<ngraph::op::v1::BinaryConvolution*>(outOp)) {
                return {t.get_index() == 1, false, false};
            } else if (auto castedOp = dynamic_cast<ngraph::op::v1::DeformableConvolution*>(outOp)) {
                return {t.get_index() == 2, castedOp->get_group() > 1, false};
            } else if (dynamic_cast<ngraph::op::v1::GroupConvolution*>(outOp)) {
                return {t.get_index() == 1, true, false};
            } else if (dynamic_cast<ngraph::op::v1::ConvolutionBackpropData*>(outOp)) {
                return {t.get_index() == 1, false, true};
            } else if (dynamic_cast<ngraph::op::v1::GroupConvolutionBackpropData*>(outOp)) {
                return {t.get_index() == 1, true, true};
            }
        }
    }
    return {false, false, false};
}

static cldnn::tensor getConstTensor(const ngraph::Shape constDims) {
    cldnn::tensor constTensor;
    switch (constDims.size()) {
    case 6: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[5]), TensorValue(constDims[4]),
                                        TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 5: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[4]), TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 4: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 3: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        1, TensorValue(constDims[2]));
        break;
    case 2: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]), 1, 1);
        break;
    case 1: constTensor = cldnn::tensor(1, TensorValue(constDims[0]), 1, 1);
        break;
    case 0: constTensor = cldnn::tensor(1, 1, 1, 1);
        break;
    default: IE_THROW() << "Invalid constant blob dimensions";
    }
    return constTensor;
}

void CreateConstantOp(Program& p, const std::shared_ptr<ngraph::op::v0::Constant>& op) {
    auto constDims = op->get_shape();
    cldnn::tensor constTensor = getConstTensor(constDims);

    // WA to inconsistency between input and const 1d tensors
    // For Concat along batch we go with batch interpretation
    // For Gather input we go with batch interpretation
    bool needsBatchInterpretation = false;
    if (constDims.size() == 1) {
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);

            for (auto& t : outTensors) {
                auto outOp = t.get_node();
                if (auto castedOp = dynamic_cast<ngraph::op::v0::Concat*>(outOp)) {
                    if (castedOp->get_axis() == 0) {
                        needsBatchInterpretation = true;
                        break;
                    }
                } else if (ngraph::op::is_binary_elementwise_arithmetic(outOp) ||
                           ngraph::op::is_binary_elementwise_logical(outOp) ||
                           ngraph::is_type<ngraph::op::v0::SquaredDifference>(outOp)) {
                    bool all_inputs_1d = true;
                    for (size_t j = 0; j < outOp->get_input_size(); j++) {
                        auto& in_shape = outOp->get_input_shape(j);
                        if (in_shape.size() != 1)
                            all_inputs_1d = false;
                    }
                    needsBatchInterpretation = all_inputs_1d;
                    break;
                } else if (ngraph::is_type<ngraph::op::v1::Gather>(outOp) ||
                           ngraph::is_type<ngraph::op::v1::Split>(outOp) ||
                           ngraph::is_type<ngraph::op::v1::VariadicSplit>(outOp)) {
                    needsBatchInterpretation = true;
                    break;
                }
            }
        }
    }

    if (needsBatchInterpretation) {
        constTensor.batch[0] = constTensor.count();
        constTensor.feature[0] = 1;
    }

    auto constFormat = DefaultFormatForDims(op->get_output_shape(0).size());
    auto prop = getConstProperties(op);

    // If constDims has a dimension = 0, then create tensor with single value
    // TODO: check if dim=0 is a valid case
    if (std::accumulate(constDims.begin(), constDims.end(), 1, std::multiplies<size_t>()) == 0)
        constTensor = cldnn::tensor{1};

    // Swap O and I dimensions to match expected deconvolution weights format
    bool swap_oi = prop.isWeights && prop.reversedChannelsOrder;
    size_t inputFeatureElements = 1;
    size_t outputFeatureElements = 1;
    size_t groups = 1;
    if (swap_oi) {
        size_t expected_min_rank = 2 + (prop.hasGroupDimension ? 1 : 0);
        if (expected_min_rank > constDims.size())
            IE_THROW() << "Invalid constant properties or shape";

        auto newDims = constDims;
        if (prop.hasGroupDimension) {
            std::swap(newDims[2], newDims[1]);
            inputFeatureElements = newDims[2];
            outputFeatureElements = newDims[1];
            groups = newDims[0];
        } else {
            std::swap(newDims[1], newDims[0]);
            inputFeatureElements = newDims[1];
            outputFeatureElements = newDims[0];
            groups = 1;
        }
        constTensor = getConstTensor(newDims);
    }

    cldnn::layout constLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)),
                                              constFormat,
                                              constTensor);

    cldnn::primitive_id initialconstPrimID = layer_type_name_ID(op);
    cldnn::primitive_id constPrimID;
    auto data = op->get_data_ptr<char>();


    auto bufIter = p.blobMemCache.find(std::make_pair(data, constDims));

    if (bufIter != p.blobMemCache.end()) {
        constPrimID = bufIter->second;
    } else {
        auto mem = cldnn::memory::allocate(p.GetEngine(), constLayout, 0, false);
        auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
        auto buf = tmpPointer.data();
        auto bufSize = constLayout.bytes_count();

        // Do actual weights reorder and change O and I channels order
        if (swap_oi) {
            auto elementSize = cldnn::data_type_traits::size_of(constLayout.data_type);
            size_t spatial_dim_off = prop.hasGroupDimension ? 3 : 2;
            size_t featureSize = elementSize;
            for (size_t i = spatial_dim_off; i < constDims.size(); i++) {
                featureSize *= constDims[i];
            }

            for (size_t g = 0; g < groups; g++) {
                for (size_t i = 0; i < inputFeatureElements; i++) {
                    for (size_t o = 0; o < outputFeatureElements; o++) {
                        size_t outputShift = ((g*outputFeatureElements + o)*inputFeatureElements + i)*featureSize;
                        size_t inputShift = ((g*inputFeatureElements + i)*outputFeatureElements + o)*featureSize;

                        for (size_t b = 0; b < featureSize; b++) {
                            buf[outputShift + b] = data[inputShift + b];
                        }
                    }
                }
            }
        } else {
            std::memcpy(&buf[0], &data[0], bufSize);
        }
        p.AddPrimitive(cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[std::make_pair(data, constDims)] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }

    p.AddPrimitiveToProfiler(op, constPrimID);
}

REGISTER_FACTORY_IMPL(v0, Constant);

}  // namespace CLDNNPlugin
