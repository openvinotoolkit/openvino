// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

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

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

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

struct ConstProperties {
    bool needsBatchInterpretation;
    bool swapOI;
    bool hasGroupDimension;
};

static void createClDnnConstant(Program& p, const ngraph::Shape& constDims, const std::shared_ptr<ngraph::op::v0::Constant>& op, const ConstProperties& props);

static void CreateConstantOp(Program& p, const std::shared_ptr<ngraph::op::v0::Constant>& op) {
    ngraph::Shape constDims = op->get_shape();
    auto constUsers = op->get_output_target_inputs(0);
    size_t numConstUsers = constUsers.size();

    std::unordered_map<std::shared_ptr<ngraph::op::v0::Constant>, ConstProperties> consts = {
        {op, {false, false, false}}
    };

    // handleConvWeights function is executed when one of the constant users is ConvolutionBackpropData or GroupConvolutionBackpropData.
    // In that case, we mark that constant's O and I dimensions need to be swapped.
    auto handleConvWeights = [&op] (ngraph::Node* conv, std::unordered_map<std::shared_ptr<ngraph::op::v0::Constant>, ConstProperties>& consts,
                                 size_t& numConstUsers, bool hasGroupDimension) {
                                 // If constant has multiple users - create its copy and replace 'conv' weights with the copy.
                                 // This is to make sure that dimension change doesn't break other users of the constant node.
                                 // It is a shallow copy, but that's fine since in createClDnnConstant
                                 // every constant created here, gets memcopied to a brand new cldnn::memory.
                                 if (numConstUsers > 1) {
                                     auto constant = std::make_shared<ngraph::op::v0::Constant>(*(op.get()));
                                     conv->input(1).replace_source_output(constant);
                                     consts.insert({constant, {false, true, hasGroupDimension}});
                                     numConstUsers--;
                                 } else {
                                     consts[op].swapOI = true;
                                     consts[op].hasGroupDimension = hasGroupDimension;
                                 }
                             };

    // WA to inconsistency between input and const 1d tensors
    // For Concat along batch we go with batch interpretation
    // For Gather input we go with batch interpretation
    // Also check if constant users is a backprop convolution - in that case O and I need to be swapped.
    for (auto& node : constUsers) {
        auto outOp = node.get_node();
        if (auto castedOp = dynamic_cast<ngraph::op::v0::Concat*>(outOp)) {
            if (castedOp->get_axis() == 0) {
                consts[op].needsBatchInterpretation = constDims.size() == 1;
            }
        } else if (ngraph::op::is_binary_elementwise_arithmetic(outOp) ||
                   ngraph::op::is_binary_elementwise_logical(outOp) ||
                   ngraph::is_type<ngraph::op::v0::SquaredDifference>(outOp)) {
            bool all_inputs_1d = true;
            for (size_t j = 0; j < outOp->get_input_size(); j++) {
                auto& in_shape = outOp->get_input_partial_shape(j);
                if (in_shape.size() > 1)
                    all_inputs_1d = false;
            }
            consts[op].needsBatchInterpretation = all_inputs_1d && constDims.size() == 1;
        } else if (ngraph::is_type<ngraph::op::v1::Gather>(outOp) ||
                   ngraph::is_type<ngraph::op::v1::Split>(outOp) ||
                   ngraph::is_type<ngraph::op::v1::VariadicSplit>(outOp)) {
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        } else if (ngraph::is_type<ngraph::op::v1::ConvolutionBackpropData>(outOp) && node.get_index() == 1) {
            handleConvWeights(outOp, consts, numConstUsers, false);
        } else if (ngraph::is_type<ngraph::op::v1::GroupConvolutionBackpropData>(outOp) && node.get_index() == 1) {
            handleConvWeights(outOp, consts, numConstUsers, true);
        } else if (ngraph::is_type<ngraph::op::v0::PRelu>(outOp) && node.get_index() == 1) {
            // PReLU slope tensor reshape policy
            //
            // 1. 1-dim slope is handled by 'getConstTensor'.
            //   ex) [1] --> [1, 1, 1, 1]
            //       [N] --> [1, N, 1, 1]
            //
            // 2. Multi-dims slope tensor is handled by the numpy broadcasting rule that is defined at
            //    'https://docs.openvino.ai/latest/openvino_docs_ops_broadcast_rules.html'.
            //   ex) [N, 1, 1] --> [1, N, 1, 1]
            //       [N, M, 1] --> [1, N, M, 1]
            auto input_shape = outOp->get_input_partial_shape(0);
            if (constDims.size() != 1 && constDims.size() < input_shape.size()) {
                // Reshape 'constDims' according to the numpy broadcasting rule.
                ngraph::Shape slope_shape(input_shape.size(), 1);
                for (size_t j = 1; j <= constDims.size(); j++)
                    slope_shape[slope_shape.size() - j] = constDims[constDims.size() - j];
                constDims = slope_shape;
            }
        } else if (ngraph::is_type<ngraph::op::v1::GroupConvolution>(outOp) && node.get_index() == 1) {
            auto input_shape = outOp->get_input_shape(0);
            if (constDims.size() == 4 && input_shape.size() == 3) { // In case of weight dim 4 and input dim 3,
                constDims[2] = constDims[3];                        // The weight cldnn tensor adds 1d to the end
                constDims[3] = 1;                                   // as the input cldnn tensor does.
            }
        }
    }

    for (auto& it : consts) {
        createClDnnConstant(p, constDims, it.first, it.second);
    }
}

void createClDnnConstant(Program& p, const ngraph::Shape& constDims, const std::shared_ptr<ngraph::op::v0::Constant>& op, const ConstProperties& props) {
    cldnn::tensor constTensor = getConstTensor(constDims);
    auto constFormat = cldnn::format::get_default_format(constDims.size());

    if (props.needsBatchInterpretation) {
        constTensor.batch[0] = constTensor.count();
        constTensor.feature[0] = 1;
    }

    // If constDims has a dimension = 0, then create tensor with single value
    // TODO: check if dim=0 is a valid case
    if (std::accumulate(constDims.begin(), constDims.end(), size_t(1), std::multiplies<size_t>()) == 0)
        constTensor = cldnn::tensor{1};

    // Swap O and I dimensions to match expected deconvolution weights format
    size_t inputFeatureElements = 1;
    size_t outputFeatureElements = 1;
    size_t groups = 1;
    auto newDims = constDims;
    if (props.swapOI) {
        size_t expected_min_rank = 2 + (props.hasGroupDimension ? 1 : 0);
        if (expected_min_rank > constDims.size())
            IE_THROW() << "Invalid constant properties or shape";

        if (props.hasGroupDimension) {
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

    cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    cldnn::layout constLayout = p.use_new_shape_infer() ? cldnn::layout(newDims, out_dtype, constFormat) :
                                                          cldnn::layout(out_dtype, constFormat, constTensor);

    cldnn::primitive_id initialconstPrimID = layer_type_name_ID(op);
    cldnn::primitive_id constPrimID;
    auto data = op->get_data_ptr<char>();

    auto bufIter = p.blobMemCache.find(std::make_pair(data, newDims));

    if (bufIter != p.blobMemCache.end()) {
        constPrimID = bufIter->second;
        p.primitive_ids[initialconstPrimID] = constPrimID;
        p.profiling_ids.push_back(initialconstPrimID);
    } else {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << initialconstPrimID << ": constant]" << std::endl;
        }
        cldnn::memory::ptr mem = p.GetEngine().allocate_memory(constLayout, false);
        auto& stream = p.GetEngine().get_program_stream();
        cldnn::mem_lock<char> lock{mem, stream};
        auto buf = lock.data();
        auto bufSize = constLayout.bytes_count();

        // Do actual weights reorder and change O and I channels order
        if (props.swapOI) {
            auto elementSize = cldnn::data_type_traits::size_of(constLayout.data_type);
            size_t spatial_dim_off = props.hasGroupDimension ? 3 : 2;
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
        p.add_primitive(*op, cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[std::make_pair(data, newDims)] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }
}

REGISTER_FACTORY_IMPL(v0, Constant);

}  // namespace intel_gpu
}  // namespace ov
