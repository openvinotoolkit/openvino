// Copyright (C) 2020 Intel Corporation
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
                return {true, false, false};
            } else if (dynamic_cast<ngraph::op::v1::BinaryConvolution*>(outOp)) {
                return {true, false, false};
            } else if (auto castedOp = dynamic_cast<ngraph::op::v1::DeformableConvolution*>(outOp)) {
                return {true, castedOp->get_group() > 1, false};
            } else if (dynamic_cast<ngraph::op::v1::GroupConvolution*>(outOp)) {
                return {true, true, false};
            } else if (dynamic_cast<ngraph::op::v1::ConvolutionBackpropData*>(outOp)) {
                return {true, false, true};
            } else if (dynamic_cast<ngraph::op::v1::GroupConvolutionBackpropData*>(outOp)) {
                return {true, true, true};
            }
        }
    }
    return {false, false, false};
}

void CreateConstantOp(Program& p, const std::shared_ptr<ngraph::op::v0::Constant>& op) {
    auto constDims = op->get_shape();
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
    default: THROW_IE_EXCEPTION << "Invalid constant blob dimensions";
    }

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
    if (prop.isWeights) {
        // Deconvolution has reversed channels order (io instead of oi)
        if (prop.reversedChannelsOrder) {
            if (prop.hasGroupDimension) {
                switch (op->get_output_shape(0).size()) {
                    case 5: constFormat = cldnn::format::gioyx; break;
                    case 6: constFormat = cldnn::format::giozyx; break;
                }
            } else {
                switch (op->get_output_shape(0).size()) {
                    case 4: constFormat = cldnn::format::ioyx; break;
                    case 5: constFormat = cldnn::format::iozyx; break;
                }
            }
        } else {
            if (prop.hasGroupDimension) {
                switch (op->get_output_shape(0).size()) {
                    case 5: constFormat = cldnn::format::goiyx; break;
                    case 6: constFormat = cldnn::format::goizyx; break;
                }
            } else {
                switch (op->get_output_shape(0).size()) {
                    case 4: constFormat = cldnn::format::oiyx; break;
                    case 5: constFormat = cldnn::format::oizyx; break;
                }
            }
        }
        std::vector<cldnn::tensor::value_type> dims(constDims.begin(), constDims.end());
        for (size_t i = dims.size(); i < 4; i++) {
            dims.push_back(1);
        }
        constTensor = cldnn::tensor(constFormat, dims);
    }

    // If constDims has a dimension = 0, then create tensor with single value
    // TODO: check if dim=0 is a valid case
    if (std::accumulate(constDims.begin(), constDims.end(), 1, std::multiplies<size_t>()) == 0)
        constTensor = cldnn::tensor{1};

    cldnn::layout constLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)),
                                              constFormat,
                                              constTensor);

    cldnn::primitive_id initialconstPrimID = layer_type_name_ID(op);
    cldnn::primitive_id constPrimID;
    auto data = op->get_data_ptr<char>();

    auto bufIter = p.blobMemCache.find(data);

    if (bufIter != p.blobMemCache.end()) {
        constPrimID = bufIter->second;
    } else {
        auto mem = cldnn::memory::allocate(p.GetEngine(), constLayout, 0, false);
        auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
        auto buf = tmpPointer.data();
        auto bufSize = constLayout.bytes_count();

        std::memcpy(&buf[0], &data[0], bufSize);
        p.AddPrimitive(cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[data] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }

    p.AddPrimitiveToProfiler(op, constPrimID);
}

REGISTER_FACTORY_IMPL(v0, Constant);

}  // namespace CLDNNPlugin
