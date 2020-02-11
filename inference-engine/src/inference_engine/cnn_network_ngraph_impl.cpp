// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_ngraph_impl.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <math.h>

#include <cassert>
#include <details/caseless.hpp>
#include <ie_ir_reader.hpp>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <ngraph/descriptor/output.hpp>
#include <ngraph/function.hpp>
#include <ngraph/op/abs.hpp>
#include <ngraph/op/acos.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/asin.hpp>
#include <ngraph/op/atan.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/batch_norm.hpp>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/ceiling.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/cos.hpp>
#include <ngraph/op/cosh.hpp>
#include <ngraph/op/deformable_convolution.hpp>
#include <ngraph/op/deformable_psroi_pooling.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/layers/ctc_greedy_decoder.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/layers/psroi_pooling.hpp>
#include <ngraph/op/experimental/layers/region_yolo.hpp>
#include <ngraph/op/experimental/layers/reorg_yolo.hpp>
#include <ngraph/op/experimental/layers/roi_pooling.hpp>
#include <ngraph/op/experimental/range.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/floor.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/conv_fused.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/fake_quantize.hpp>
#include <ngraph/op/fused/grn.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/normalize_l2.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/fused/squeeze.hpp>
#include <ngraph/op/fused/unsqueeze.hpp>
#include <ngraph/op/fused/hard_sigmoid.hpp>
#include <ngraph/op/gather.hpp>
#include <ngraph/op/get_output_element.hpp>
#include <ngraph/op/less.hpp>
#include <ngraph/op/log.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/min.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/non_max_suppression.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/reduce_mean.hpp>
#include <ngraph/op/reduce_prod.hpp>
#include <ngraph/op/reduce_sum.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/reverse_sequence.hpp>
#include <ngraph/op/select.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/sin.hpp>
#include <ngraph/op/sinh.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/sqrt.hpp>
#include <ngraph/op/subtract.hpp>
#include <ngraph/op/tan.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/topk.hpp>
#include <ngraph/op/and.hpp>
#include <ngraph/op/or.hpp>
#include <ngraph/op/xor.hpp>
#include <ngraph/op/not.hpp>
#include <ngraph/op/reduce_logical_and.hpp>
#include <ngraph/op/reduce_logical_or.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/get_output_element_elimination.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/strides.hpp>
#include <ngraph/variant.hpp>
#include <set>
#include <shape_infer/ie_reshaper.hpp>
#include <string>
#include <transform/transformations/constant_eltwise_reduction.hpp>
#include <transform/transformations/convert_broadcast_to_tiles.hpp>
#include <transform/transformations/convert_convolutions.hpp>
#include <transform/transformations/convert_divide.hpp>
#include <transform/transformations/convert_mod.hpp>
#include <transform/transformations/convert_gather_to_gather_ie.hpp>
#include <transform/transformations/convert_gathertree_to_gathertree_ie.hpp>
#include <transform/transformations/convert_interpolate_to_interp_or_resample.hpp>
#include <transform/transformations/convert_lrn_to_lrn_ie.hpp>
#include <transform/transformations/convert_lstm_cell_to_lstm_cell_ie.hpp>
#include <transform/transformations/convert_matmul_to_fc_or_gemm.hpp>
#include <transform/transformations/convert_minimum_to_power_and_max.hpp>
#include <transform/transformations/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transform/transformations/convert_mul_or_add_finally.hpp>
#include <transform/transformations/convert_negative.hpp>
#include <transform/transformations/convert_nms_to_nms_ie.hpp>
#include <transform/transformations/convert_normalizel2_to_normalize_ie.hpp>
#include <transform/transformations/convert_one_hot_to_one_hot_ie.hpp>
#include <transform/transformations/convert_pad_to_pad_ie.hpp>
#include <transform/transformations/convert_power_to_power_ie.hpp>
#include <transform/transformations/convert_prelu_to_relu_ie.hpp>
#include <transform/transformations/convert_prior_clustered_to_ie_clustered.hpp>
#include <transform/transformations/convert_proposal_to_proposal_ie.hpp>
#include <transform/transformations/convert_prior_to_ie_prior.hpp>
#include <transform/transformations/convert_reduce_to_pooling.hpp>
#include <transform/transformations/convert_strided_slice_to_crop.hpp>
#include <transform/transformations/convert_subtract.hpp>
#include <transform/transformations/convert_selu_to_selu_ie.hpp>
#include <transform/transformations/convert_tile_to_ie_tile.hpp>
#include <transform/transformations/convert_topk_to_topk_ie.hpp>
#include <transform/transformations/convert_depth_to_space.hpp>
#include <transform/transformations/convert_space_to_depth.hpp>
#include <transform/transformations/fusion/batch_norm_decomposition.hpp>  //NOLINT
#include <transform/transformations/fusion/conv_bias_fusion.hpp>
#include <transform/transformations/fusion/fc_bias_fusion.hpp>
#include <transform/transformations/fusion/mul_add_squence_fusion.hpp>
#include <transform/transformations/fusion/mul_add_verification.hpp>
#include <transform/transformations/reshape_constant_folding.hpp>
#include <transform/transformations/reshape_1d_convolutions.hpp>
#include <transform/transformations/pull_transpose_through_fq.hpp>
#include <transform/transformations/convert_strided_slice_to_strided_slice_ie.hpp>
#include <transform/transformations/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp>

#include "debug.h"
#include "graph_tools.hpp"
#include "graph_transformer.h"
#include "ie_util_internal.hpp"
#include "ie_cnn_layer_builder_ngraph.h"
#include "ie_ngraph_utils.hpp"
#include "ie_profiling.hpp"
#include "network_serializer.h"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/fully_connected.hpp"
#include "ngraph_ops/gather_ie.hpp"
#include "ngraph_ops/generic_ie.hpp"
#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph_ops/interp.hpp"
#include "ngraph_ops/lrn_ie.hpp"
#include "ngraph_ops/lstm_cell_ie.hpp"
#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph_ops/pad_ie.hpp"
#include "ngraph_ops/onehot_ie.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/proposal_ie.hpp"
#include "ngraph_ops/quantize_conv_bias_fused.hpp"
#include "ngraph_ops/relu_ie.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tensor_iterator.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/hard_sigmoid_ie.hpp"

using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

static std::shared_ptr<ngraph::Function> copyFunction(const std::shared_ptr<ngraph::Function>& func,
                                                      bool constFolding,
                                                      const std::map<std::string, std::vector<size_t>>& inputShapes) {
    ::ngraph::op::GenericIE::DisableReshape noReshape(func);
    auto original_parameters = func->get_parameters();

    std::vector<::ngraph::element::Type> new_types;
    std::vector<::ngraph::PartialShape> new_shapes;

    for (const auto &parameter : original_parameters) {
        if (inputShapes.find(parameter->get_friendly_name()) != inputShapes.end()) {
            new_shapes.emplace_back(inputShapes.at(parameter->get_friendly_name()));
        } else {
            new_shapes.emplace_back(parameter->get_shape());
        }
        new_types.emplace_back(parameter->get_element_type());
    }
    IE_ASSERT(original_parameters.size() == new_types.size());
    IE_ASSERT(original_parameters.size() == new_shapes.size());

    auto specialized_function = ::ngraph::specialize_function(func, new_types, new_shapes,
                                                              std::vector<void*>(new_shapes.size(), nullptr), constFolding, true);
    // TODO: remove this code after the fix on the nGraph side
    ::ngraph::pass::GetOutputElementElimination goe_elimination;
    for (auto n : specialized_function->get_ops()) {
        goe_elimination.run_on_node(n);
    }
    return specialized_function;
}

// WA: for cnnNetwork ngraph constructor
CNNNetwork::CNNNetwork(const std::shared_ptr<ngraph::Function>& graph) {
#if defined(ENABLE_NGRAPH)
    // Copy nGraph function
    network = std::make_shared<CNNNetworkNGraphImpl>(copyFunction(graph, false, {}));
    actual = network.get();
    if (actual == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
    }
#else
    THROW_IE_EXCEPTION << "InferenceEngine was built without nGraph support!";
#endif
}

void CNNNetworkNGraphImpl::createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName,
                                               DataPtr& ptr) {
    // query shape from ngraph::Parameter output shape and check there are no zeros in it
    SizeVector dims;
    if (output.get_partial_shape().is_static()) {
        dims = output.get_shape();
    }
    for (const auto& dim : dims) {
        if (!dim) THROW_IE_EXCEPTION << outName << " has zero dimension that is not allowable";
    }

    if (ptr) {
        ptr->reshape(dims, ptr->getTensorDesc().getLayout());
    } else {
        const auto precision = details::ngraph::convertPrecision(output.get_element_type());
        const auto layout = TensorDesc::getLayoutByDims(dims);
        ptr.reset(new NGraphData(this, outName, {precision, dims, layout}));
    }
}

std::shared_ptr<ICNNNetwork> CNNNetworkNGraphImpl::getCNNNetwork() {
    if (!cnnNetwork)
        convertToCNNNetworkImpl();
    return cnnNetwork;
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const std::shared_ptr<Function>& nGraph)
    : _ngraph_function(nGraph), _stats(new CNNNetworkStatsImpl()) {
    // Restore usual attributes for ICNNNetwork
    auto keep_input_info = [](CNNNetworkNGraphImpl& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        Precision prc = info->getPrecision();

        // Convert precision into native format (keep element size)
        prc = prc == Precision::Q78
                  ? Precision::I16
                  : prc == Precision::FP16 ? Precision::FP32 : static_cast<Precision::ePrecision>(prc);

        info->setPrecision(prc);
        network.setInputInfo(info);
    };
    ::ngraph::op::GenericIE::addExtension(std::make_shared<ShapeInfer::BuiltInShapeInferHolder>());

    reshape();
    for (const auto& layer : _ngraph_function->get_parameters()) {
        std::string outName = layer->get_friendly_name();
        IE_ASSERT(layer->get_output_size() == 1);  // Parameter as only singly output port

        DataPtr& ptr = _data[outName];
        IE_ASSERT(ptr);  // Data must be allocated after the reshape method

        keep_input_info(*this, ptr);
    }
    for (auto& output : _outputData) {
        // Convert precision into native format. Be consistent with possible convertation to CNNNetwork later.
        if (output.second->getPrecision() != Precision::FP32 &&
            output.second->getPrecision() != Precision::I32) {
            output.second->setPrecision(Precision::FP32);
        }
    }
}

CNNNetworkNGraphImpl::~CNNNetworkNGraphImpl() {
    for (auto& data : _data) {
        if (!data.second) continue;
        if (auto nData = std::dynamic_pointer_cast<NGraphData>(data.second)) {
            nData->reset();
        }
    }
}

void CNNNetworkNGraphImpl::setInputInfo(InputInfo::Ptr data) {
    if (cnnNetwork && !networksEqual) cnnNetwork->setInputInfo(data);
    _inputData[data->name()] = data;
}

DataPtr& CNNNetworkNGraphImpl::getData(const char* name) noexcept {
    if (cnnNetwork) return cnnNetwork->getData(name);
    if (_data.find(name) != _data.end()) {
        return _data[name];
    } else {
        try {
            convertToCNNNetworkImpl();
            return cnnNetwork->getData(name);
        } catch (...) {
            return _data[name];
        }
    }
}

void CNNNetworkNGraphImpl::getName(char* pName, size_t len) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getName(pName, len);
        return;
    }
    // Description buffer will preserve garbage if external pointer not initialized
    if (len < 1) return;
    memset(pName, 0, len);
    DescriptionBuffer(pName, len) << _ngraph_function->get_name();
}

const std::string& CNNNetworkNGraphImpl::getName() const noexcept {
    if (cnnNetwork) {
        return cnnNetwork->getName();
    }
    return _ngraph_function->get_name();
}

InputInfo::Ptr CNNNetworkNGraphImpl::getInput(const std::string& inputName) const noexcept {
    if (cnnNetwork) return cnnNetwork->getInput(inputName);
    auto it = _inputData.find(inputName);
    if (it == _inputData.end()) {
        return nullptr;
    }
    return it->second;
}

Precision CNNNetworkNGraphImpl::getPrecision() const noexcept {
    return Precision::MIXED;
}

void CNNNetworkNGraphImpl::getOutputsInfo(OutputsDataMap& out) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getOutputsInfo(out);
        return;
    }
    out = _outputData;
}

void CNNNetworkNGraphImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getInputsInfo(inputs);
        return;
    }
    inputs = _inputData;
}

size_t CNNNetworkNGraphImpl::layerCount() const noexcept {
    if (cnnNetwork) return cnnNetwork->layerCount();
    return _ngraph_function->get_ops().size();
}

void CNNNetworkNGraphImpl::addLayer(const CNNLayerPtr& layer) noexcept {
    try {
        if (!cnnNetwork) {
            convertToCNNNetworkImpl();
        }
        cnnNetwork->addLayer(layer);
        if (layer)
            networksEqual = false;
    } catch (...) {
        return;
    }
}

void CNNNetworkNGraphImpl::validate(int version) {
    if (cnnNetwork)
        cnnNetwork->validate();
    else
        _ngraph_function->validate_nodes_and_infer_types();
}

StatusCode CNNNetworkNGraphImpl::getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const
    noexcept {
    auto network = cnnNetwork;
    std::shared_ptr<::ngraph::Function> converted_function;

    if (!network) {
        converted_function = cloneFunction();
        convertFunctionToICNNNetwork(converted_function, network);
    }
    if (!network) return GENERAL_ERROR;
    return network->getLayerByName(layerName, out, resp);
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName, size_t outputIndex,
                                           ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(addOutput)
    if (cnnNetwork && !networksEqual) {
        return cnnNetwork->addOutput(layerName, outputIndex, resp);
    }

    try {
        for (const auto layer : _ngraph_function->get_ops()) {
            if (layer->get_friendly_name() == layerName) {
                auto& results = const_cast<::ngraph::ResultVector&>(_ngraph_function->get_results());
                auto result = make_shared<::ngraph::op::Result>(layer->output(outputIndex));
                results.push_back(result);

                std::string outputName = layerName;
                if (layer->outputs().size() != 1) {
                    outputName += "." + std::to_string(outputIndex);
                }
                if (_data.find(outputName) != _data.end()) {
                    addOutput(outputName);
                    if (cnnNetwork)
                        return cnnNetwork->addOutput(layerName, outputIndex, resp);
                } else {
                    reshape();
                    addOutput(outputName);
                }
                return OK;
            }
        }
    } catch (...) {
        return GENERAL_ERROR;
    }
    return DescriptionBuffer(NOT_FOUND, resp) << "Cannot add output! Layer " << layerName << " wasn't found!";
}

void CNNNetworkNGraphImpl::addOutput(const string& dataName) {
    auto it = _data.find(dataName);
    if (it == _data.end()) {
        THROW_IE_EXCEPTION << "data [" << dataName << "] doesn't exist";
    }
    auto data = it->second;
    assert(data->getName() == dataName);
    _outputData[dataName] = data;
}

size_t CNNNetworkNGraphImpl::getBatchSize() const noexcept {
    // TODO Provide adequate implementation.
    // The original code from CNNNetworkImpl just gets the first input and returns the first dimension.
    // This is not correct in general. We can follow the same semantics, but order of inputs should be
    // guaranteed to be the same.
    if (cnnNetwork) {
        return cnnNetwork->getBatchSize();
    }
    auto params = _ngraph_function->get_parameters();
    if (params.empty() || !params[0]->get_partial_shape().is_static()) return 0;

    auto shape = _ngraph_function->get_parameters()[0]->get_shape();

    // WA: for speech recognition layouts (copy-past from CNNNetwork)
    if (shape.size() == 3 || shape.size() == 1) {
        return 1;
    }
    return shape[0];
}

std::shared_ptr<ngraph::Function> CNNNetworkNGraphImpl::cloneFunction(bool constFolding, const std::map<std::string, std::vector<size_t>>& inputShapes) const {
    return copyFunction(_ngraph_function, constFolding, inputShapes);
}

void CNNNetworkNGraphImpl::reshape() {
    ResponseDesc desc;

    // Disable reshape for generic nodes
    ::ngraph::op::GenericIE::DisableReshape noReshape(_ngraph_function);
    StatusCode ret = reshape({}, &desc);
    if (ret != OK)
        THROW_IE_EXCEPTION << desc.msg;
}

StatusCode
CNNNetworkNGraphImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                        ResponseDesc* responseDesc) noexcept {
    IE_PROFILING_AUTO_SCOPE(reshape)
    if (cnnNetwork && !networksEqual)
        return cnnNetwork->reshape(inputShapes, responseDesc);
    try {
        auto params = _ngraph_function->get_parameters();

        for (size_t i = 0; i < params.size(); i++) {
            const auto& param = params[i];
            if (inputShapes.find(param->get_friendly_name()) == inputShapes.end())
                continue;
            ::ngraph::PartialShape shape(inputShapes.at(param->get_friendly_name()));
            auto newParam = std::make_shared<::ngraph::op::Parameter>(param->get_element_type(), shape);
            newParam->set_friendly_name(param->get_friendly_name());
            _ngraph_function->replace_parameter(i, newParam);
        }
        _ngraph_function->validate_nodes_and_infer_types();

        if (cnnNetwork) {
            convertToCNNNetworkImpl();
        } else {
            auto specialized_ngraph_function = cloneFunction(true, inputShapes);
            // Call this transformation because OneHot IE and nGraph have different output precisions
            {
                IE_PROFILING_AUTO_SCOPE(ConvertOneHot);
                bool has_fp16 = has_f16_constants(specialized_ngraph_function);
                ::ngraph::pass::ConvertOneHotToOneHotIE(has_fp16).run_on_function(specialized_ngraph_function);
            }
            specialized_ngraph_function->validate_nodes_and_infer_types();
            std::unordered_set<std::string> opName;

            for (const auto & layer : specialized_ngraph_function->get_ordered_ops()) {
                if (opName.find(layer->get_friendly_name()) != opName.end())
                    THROW_IE_EXCEPTION << "All operations in nGraph function should have unique friendly names!";
                opName.insert(layer->get_friendly_name());
                if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
                    IE_ASSERT(layer->get_inputs().size() == 1);
                    const auto& input = layer->get_inputs()[0];
                    std::string outName = input.get_output().get_node()->get_friendly_name();
                    if (input.get_output().get_node()->get_output_size() != 1)
                        outName += "." + std::to_string(input.get_output().get_index());
                    addOutput(outName);
                }
                for (const auto& output : layer->outputs()) {
                    std::string outName = layer->get_friendly_name();
                    if (layer->outputs().size() != 1)
                        outName += "." + std::to_string(output.get_index());
                    createDataForResult(output, outName, _data[outName]);
                }
            }
        }
    } catch (std::exception& ex) {
#if 0
        for (const auto& op : _ngraph_function->get_ordered_ops()) {
            std::cout << op->get_friendly_name() << " " << op->description() << std::endl << "    Inputs: ";
            for (const auto& in : op->inputs()) {
                auto shape = in.get_shape();
                std::cout << " {";
                for (auto i : shape) {
                    std::cout << i << " ";
                }
                std::cout << "} ";
            }
            std::cout << std::endl << "    Outputs: ";
            for (const auto& in : op->outputs()) {
                auto shape = in.get_shape();
                std::cout << " {";
                for (auto i : shape) {
                    std::cout << i << " ";
                }
                std::cout << "} ";
            }
            std::cout << std::endl;
        }
#endif

        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }

    return OK;
}

StatusCode CNNNetworkNGraphImpl::AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                                              InferenceEngine::ResponseDesc* resp) noexcept {
    if (!cnnNetwork || networksEqual) {
        ::ngraph::op::GenericIE::addExtension(extension);
    }
    return cnnNetwork ? cnnNetwork->AddExtension(extension, resp) : OK;
}

StatusCode CNNNetworkNGraphImpl::serialize(const std::string& xmlPath, const std::string& binPath,
                                           ResponseDesc* resp) const noexcept {
    auto network = cnnNetwork;
    std::shared_ptr<::ngraph::Function> converted_function;

    if (!network) {
        converted_function = cloneFunction();
        convertFunctionToICNNNetwork(converted_function, network);
    }
    if (!network) return GENERAL_ERROR;
    return network->serialize(xmlPath, binPath, resp);
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        if (!cnnNetwork)
            convertToCNNNetworkImpl();
        networksEqual = false;
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

StatusCode CNNNetworkNGraphImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    if (cnnNetwork && !networksEqual)
        return cnnNetwork->setBatchSizeReshape(size, responseDesc);
    try {
        auto original_parameters = _ngraph_function->get_parameters();

        std::map<std::string, std::vector<size_t>> origShapes;
        std::map<std::string, std::vector<size_t>> inShapes;
        for (const auto &parameter : original_parameters) {
            if (parameter->get_partial_shape().is_dynamic())
                THROW_IE_EXCEPTION << "Cannot setBatch! Network contains inputs with dynamic shapes!";
            std::vector<size_t> shape = parameter->get_shape();
            origShapes[parameter->get_friendly_name()] = shape;
            shape[0] = size;
            inShapes[parameter->get_friendly_name()] = shape;
        }
        auto sts = reshape(inShapes, responseDesc);
        if (sts == OK) return OK;
        for (size_t i = 0; i < original_parameters.size(); i++) {
            const auto& param = original_parameters[i];
            if (origShapes.find(param->get_friendly_name()) == origShapes.end())
                continue;
            ::ngraph::PartialShape shape(origShapes.at(param->get_friendly_name()));
            auto newParam = std::make_shared<::ngraph::op::Parameter>(param->get_element_type(), shape);
            newParam->set_friendly_name(param->get_friendly_name());
            _ngraph_function->replace_parameter(i, newParam);
        }
        networksEqual = false;
        convertToCNNNetworkImpl();
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

void CNNNetworkNGraphImpl::convertToCNNNetworkImpl() {
    if (cnnNetwork && !networksEqual)
        return;
    IE_PROFILING_AUTO_SCOPE(convertToCNNNetworkImpl)
    InputsDataMap thisInputDataMap;
    getInputsInfo(thisInputDataMap);
    _converted_function = cloneFunction();
    cnnNetwork.reset();
    convertFunctionToICNNNetwork(_converted_function, cnnNetwork);

    if (!cnnNetwork) THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";

    // update input preprocessing info
    InputsDataMap resultInputDataMap;
    cnnNetwork->getInputsInfo(resultInputDataMap);
    IE_ASSERT(resultInputDataMap.size() == thisInputDataMap.size());
    for (auto i : resultInputDataMap) {
        auto& thisInputData = *thisInputDataMap[i.first];
        i.second->setPrecision(thisInputData.getPrecision());
        i.second->setLayout(thisInputData.getLayout());
        i.second->getPreProcess() = thisInputData.getPreProcess();
    }

    for (const auto& ext : ::ngraph::op::GenericIE::getExtensions()) {
        cnnNetwork->AddExtension(ext, nullptr);
    }
}

void CNNNetworkNGraphImpl::convertFunctionToICNNNetwork(std::shared_ptr<::ngraph::Function>& func,
                                                        std::shared_ptr<details::CNNNetworkImpl>& cnnNetworkImpl) const {
    IE_PROFILING_AUTO_SCOPE(convertFunctionToICNNNetwork)
    const auto createCNNLayer = [](const std::shared_ptr<::ngraph::Node>& node) -> CNNLayerPtr {
        static std::vector<std::shared_ptr<Builder::INodeConverter>> convertors = {
            std::make_shared<Builder::NodeConverter<::ngraph::op::Abs>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Acos>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Add>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Asin>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Atan>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::AvgPool>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::BatchNormInference>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Broadcast>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Clamp>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Concat>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Constant>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ConvolutionIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::DeconvolutionIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::BinaryConvolution>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Cos>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Cosh>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::CropIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Convert>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::CTCGreedyDecoder>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::DetectionOutput>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformableConvolution>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformablePSROIPooling>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Divide>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Reshape>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Eltwise>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Elu>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Erf>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Exp>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::FakeQuantize>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Floor>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Ceiling>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::GatherIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::GatherTree>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::GatherTreeIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Interp>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Interpolate>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Log>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::LRN>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::LRN_IE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::MVN>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::FullyConnected>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::GemmIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::GenericIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::GRN>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::MaxPool>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Maximum>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Minimum>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Multiply>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::NonMaxSuppression>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::NonMaxSuppressionIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeL2>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::OneHotIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PRelu>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PadIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Power>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PowerIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBox>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxClustered>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxClusteredIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Proposal>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ProposalIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Relu>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::SeluIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ReLUIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Range>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ReverseSequence>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMin>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMax>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMean>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceProd>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceSum>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ResampleV2>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::RegionYolo>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ReorgYolo>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ROIPooling>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::PSROIPooling>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ScaleShiftIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::ShapeOf>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Sigmoid>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Sin>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Sign>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Sinh>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::SquaredDifference>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Softmax>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Split>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::VariadicSplit>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::StridedSlice>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::StridedSliceIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Squeeze>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Sqrt>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Subtract>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Tan>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Tanh>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::TileIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::TopK>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::TopKIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Transpose>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::Unsqueeze>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::TensorIterator>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::LSTMCellIE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid_IE>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::LogicalNot>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalAnd>>(),
            std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalOr>>(),
        };

        CNNLayerCreator visitor(node);
        if (node->visit_attributes(visitor)) return visitor.create();

        for (auto& convertor : convertors) {
            if (!convertor->canCreate(node)) continue;
            return convertor->createLayer(node);
        }
        THROW_IE_EXCEPTION << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
    };

    const auto isInternalConstLayer = [](const std::shared_ptr<::ngraph::Node>& constLayer,
                                         const std::shared_ptr<::ngraph::Node>& consumerLayer,
                                         bool keep_constants) -> bool {
        if ((( ::ngraph::as_type_ptr<::ngraph::op::ConvolutionIE>(consumerLayer) ||
               ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
               ::ngraph::as_type_ptr<::ngraph::op::FullyConnected>(consumerLayer)) && !keep_constants) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::DeconvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Elu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::NormalizeIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::PRelu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::Split>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::VariadicSplit>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::ScaleShiftIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Transpose>(consumerLayer)) {
            // Check that all input nodes except zero input are Constants for all ops except DeformableConvolutions
            // for which the input with index 1 is also dynamic
            size_t inputID = ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ? 2 : 1;
            for (; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMCellIE>(consumerLayer)) {
            for (size_t inputID = 3; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        }
        return false;
    };

    // Checks that node is internal layer for all layers from specific function
    const auto isInternalLayer = [=](const std::shared_ptr<::ngraph::Node>& node,
                                     const std::unordered_set<std::string>& names,
                                     bool keep_constant) -> bool {
        if (auto constantNode = ::ngraph::as_type_ptr<::ngraph::op::Constant>(node)) {
            for (const auto& consumerInputPort : constantNode->get_outputs()[0].get_inputs()) {
                const auto& consumerLayer = consumerInputPort->get_node();
                if (names.find(consumerLayer->get_name()) == names.end())
                    continue;
                if (!isInternalConstLayer(constantNode, consumerLayer, keep_constant))
                    return false;
            }
            return true;
        }

        return ::ngraph::as_type_ptr<::ngraph::op::Result>(node) != nullptr;
    };

    const auto keep_input_info = [](std::shared_ptr<details::CNNNetworkImpl>& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        network->setInputInfo(info);
    };

    std::shared_ptr<::ngraph::Function> graph = func;

    // Disable shape inference (WA for generic operations)
    ::ngraph::op::GenericIE::DisableReshape noReshape(graph);

    {
        IE_PROFILING_AUTO_SCOPE(ConvertPriorBox)
        ::ngraph::pass::ConvertPriorBox().run_on_function(graph);
        ::ngraph::pass::ConvertPriorBoxClustered().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding1)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }

    //  RecudeToPooling transformation inserts Multiply operation in case of ReduceSum
    //  so we need to run it before optimizations and lin ops transformations
    {
        IE_PROFILING_AUTO_SCOPE(ConvertReduceToPooling)
        ::ngraph::pass::ConvertReduceToPooling().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertMod)
        ::ngraph::pass::ConvertMod().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertMinimum)
        ::ngraph::pass::ConvertMinimum().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertSubtract)
        ::ngraph::pass::ConvertSubtract().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertDivide)
        ::ngraph::pass::ConvertDivide().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertNegative)
        ::ngraph::pass::ConvertNegative().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertDepthToSpace)
        ::ngraph::pass::ConvertDepthToSpace().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertSpaceToDepth)
        ::ngraph::pass::ConvertSpaceToDepth().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertConvolutions)
        ::ngraph::pass::ConvertConvolutions().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(BatchNormDecomposition)
        ::ngraph::pass::BatchNormDecomposition().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding2)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(MulAddVerification)
        ::ngraph::pass::MulAddVerification().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(MulAddFusion)
        ::ngraph::pass::MulAddFusion().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding3)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertMatMulToFCorGemm)
        ::ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(PullTransposeThroughFQUp)
        ::ngraph::pass::PullTransposeThroughFQUp().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding4)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvFusion)
        ::ngraph::pass::ConvFusion().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(FullyConnectedBiasFusion)
        ::ngraph::pass::FullyConnectedBiasFusion().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding5)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(Reshape1DConvolutions)
        ::ngraph::pass::Reshape1DConvolutions().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertNormalizeL2WithMulToNormalizeIE)
        ::ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertNormalizeL2ToNormalizeIE)
        ::ngraph::pass::ConvertNormalizeL2ToNormalizeIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantEltwiseReduction)
        ::ngraph::pass::ConstantEltwiseReduction().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertMulAddToScaleShiftOrPower)
        ::ngraph::pass::ConvertMulAddToScaleShiftOrPower().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertMulOrAddFinally)
        ::ngraph::pass::ConvertMulOrAddFinally().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ReshapeConstanFolding)
        ::ngraph::pass::ReshapeConstanFolding().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding6)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertBroadcastToTiles)
        ::ngraph::pass::ConvertBroadcastToTiles().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertTileToIETile)
        ::ngraph::pass::ConvertTileToIETile().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertProposalToProposalIE)
        ::ngraph::pass::ConvertProposalToProposalIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertLRNToLRNIE)
        ::ngraph::pass::ConvertLRNToLRNIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertPadToPadIE)
        ::ngraph::pass::ConvertPadToPadIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertHardSigmoidToHardSigmoidIE)
        ::ngraph::pass::ConvertHardSigmoidToHardSigmoidIE().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertLSTMCellToLSTMCellIE)
        ::ngraph::pass::ConvertLSTMCellToLSTMCellIE().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertInterpolateToInterpOrResample)
        ::ngraph::pass::ConvertInterpolateToInterpOrResample().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertStridedSliceToCrop)
        ::ngraph::pass::ConvertStridedSliceToCrop().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertStridedSliceToStridedSliceIE)
        ::ngraph::pass::ConvertStridedSliceToStridedSliceIE().run_on_function(graph);
    }

    {
        IE_PROFILING_AUTO_SCOPE(ConvertPowerToPowerIE)
        ::ngraph::pass::ConvertPowerToPowerIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertPReLUToReLUIE)
        ::ngraph::pass::ConvertPReLUToReLUIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertGatherToGatherIE)
        ::ngraph::pass::ConvertGatherToGatherIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertSeluToSeluIE)
        ::ngraph::pass::ConvertSeluToSeluIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertOneHot)
        bool has_fp16 = has_f16_constants(graph);
        ::ngraph::pass::ConvertOneHotToOneHotIE(has_fp16).run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertGatherTreeToGatherTreeIE)
        ::ngraph::pass::ConvertGatherTreeToGatherTreeIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertTopKToTopKIE)
        ::ngraph::pass::ConvertTopKToTopKIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConvertNMSToNMSIE)
        ::ngraph::pass::ConvertNMSToNMSIE().run_on_function(graph);
    }
    {
        IE_PROFILING_AUTO_SCOPE(ConstantFolding7)
        ::ngraph::pass::ConstantFolding().run_on_function(graph);
    }
    graph->validate_nodes_and_infer_types();

    // Create network
    cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>();
    cnnNetworkImpl->setName(graph->get_friendly_name());
    // In generic case all nGraph functions have MIXED precision
    // Network precision should be deprecated
    cnnNetworkImpl->setPrecision(Precision::MIXED);

    // Collect all names from current graph
    // It is necessary in order to differentiate outputs from constant layers when we share constants
    // (Constant operations contains outputs for converted and original functions)
    std::unordered_set<std::string> op_names;
    for (const auto& layer : graph->get_ops())
        op_names.insert(layer->get_name());

    bool keep_constants(false);
    for (auto & layer : graph->get_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::FakeQuantize>(layer)) {
            keep_constants = true;
            break;
        }
    }

    // Create layers and output data
    for (const auto& layer : graph->get_ops()) {
        if (isInternalLayer(layer, op_names, keep_constants)) continue;

        // TODO: remove this rt info when all blobs will be inputs
        InferenceEngine::Parameter attr(keep_constants);
        auto & rt_info = layer->get_rt_info();
        rt_info["keep_constants"] = attr.asVariant();

        CNNLayerPtr cnnLayer = createCNNLayer(layer);
        for (const auto& rt : layer->get_rt_info()) {
            Parameter param(rt.second);
            if (param.empty()) continue;
            if (details::CaselessEq<std::string>()(rt.first, "affinity")) {
                cnnLayer->affinity = param.as<std::string>();
            } else if (param.is<std::string>()) {
                cnnLayer->params[rt.first] = param.as<std::string>();
            }
        }
        size_t inputCount(0);
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto& input = layer->get_inputs()[i];
            if (isInternalLayer(input.get_output().get_node(), op_names, keep_constants)) continue;
            inputCount++;
        }
        cnnLayer->insData.resize(inputCount);
        for (size_t i = 0; i < layer->get_output_size(); i++) {
            std::string outName = layer->get_friendly_name();
            if (layer->get_output_size() != 1) outName += "." + std::to_string(i);
            DataPtr& ptr = cnnNetworkImpl->getData(outName.c_str());

            SizeVector dims;
            dims = layer->get_output_shape(i);
            for (const auto& dim : dims) {
                if (!dim)
                    THROW_IE_EXCEPTION << cnnLayer->type << " layer " << cnnLayer->name << " has incorrect dimensions in the output data " << i;
            }

            if (!ptr && _data.find(outName) != _data.end()) {
                ptr = _data.at(outName);
                if (auto nData = std::dynamic_pointer_cast<NGraphData>(ptr)) {
                    const auto layout =
                            dims.size() == nData->getTensorDesc().getDims().size() ?
                                nData->getTensorDesc().getLayout() :
                                TensorDesc::getLayoutByDims(dims);

                    nData->reset();
                    nData->reshape(dims, layout);
                }
                cnnNetworkImpl->addData(outName.c_str(), ptr);
            }
            if (!ptr) {
                ptr.reset(new Data(outName, {details::ngraph::convertPrecision(layer->get_output_element_type(i)), dims,
                                             TensorDesc::getLayoutByDims(dims)}));
            }

            ptr->getCreatorLayer() = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto& layer : graph->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_inputs().size() == 1);
            const auto &input = layer->get_inputs()[0];
            std::string outName = input.get_output().get_node()->get_friendly_name();
            if (input.get_output().get_node()->get_output_size() != 1)
                outName += "." + std::to_string(input.get_output().get_index());
            cnnNetworkImpl->addOutput(outName);
            continue;
        }

        uint64_t count_of_skipped = 0;
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto& input = layer->get_inputs()[i];
            if (isInternalConstLayer(input.get_output().get_node(), layer, keep_constants)) {
                count_of_skipped++;
                continue;
            }
            CNNLayerPtr prevCnnLayer;
            StatusCode ret = cnnNetworkImpl->getLayerByName(input.get_output().get_node()->get_friendly_name().c_str(),
                                                            prevCnnLayer, nullptr);
            if (ret != OK)
                THROW_IE_EXCEPTION << "Cannot find layer with name: "
                                   << input.get_output().get_node()->get_friendly_name();

            CNNLayerPtr cnnLayer;
            ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
            if (ret != OK) THROW_IE_EXCEPTION << "Cannot find layer with name: " << layer->get_friendly_name();
            auto inIndex = input.get_index();
            if (cnnLayer->insData.size() <= (inIndex - count_of_skipped) ||
                prevCnnLayer->outData.size() <= input.get_output().get_index() || count_of_skipped > inIndex)
                THROW_IE_EXCEPTION << "Cannot create ICNNNetwork. Network structure is incorrect! "
                    << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                    << cnnLayer->type << " layer " << cnnLayer->name
                    << " cannot be connected with output port " << input.get_output().get_index()
                    << " (max " << prevCnnLayer->outData.size() << ") of " << prevCnnLayer->type
                    << " layer " << prevCnnLayer->name;
            cnnLayer->insData[inIndex - count_of_skipped] = prevCnnLayer->outData[input.get_output().get_index()];
            prevCnnLayer->outData[input.get_output().get_index()]->getInputTo()[cnnLayer->name] = cnnLayer;
        }
    }

    // check all input ports are occupied
    for (const auto& kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr& layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port " << i
                                   << " is not connected to any data";
            }
        }
        layer->validateLayer();
    }

    ICNNNetworkStats *pstats;
    ICNNNetworkStats *lstats;
    ResponseDesc response;
    StatusCode sts = this->getStats(&pstats, &response);
    if (sts != StatusCode::OK) {
        THROW_IE_EXCEPTION << response.msg;
    }
    sts = cnnNetworkImpl->getStats(&lstats, &response);
    if (sts != StatusCode::OK) {
        THROW_IE_EXCEPTION << response.msg;
    }
    lstats->setNodesStats(pstats->getNodesStats());
}

std::shared_ptr<CNNNetworkNGraphImpl> CNNNetworkNGraphImpl::cloneNGraphImpl() const {
    auto result = std::make_shared<CNNNetworkNGraphImpl>(cloneFunction());
    for (const auto& outputInfo : _outputData) {
        result->_outputData[outputInfo.first]->setPrecision(outputInfo.second->getPrecision());
        result->_outputData[outputInfo.first]->setLayout(outputInfo.second->getLayout());
    }
    for (const auto& inputInfo : _inputData) {
        result->_inputData[inputInfo.first]->setPrecision(inputInfo.second->getPrecision());
        result->_inputData[inputInfo.first]->setLayout(inputInfo.second->getLayout());
        result->_inputData[inputInfo.first]->getPreProcess() = inputInfo.second->getPreProcess();
    }
    if (cnnNetwork)
        result->cnnNetwork = cloneNet(*cnnNetwork);
    result->networksEqual = networksEqual;
    result->_stats = _stats;
    return result;
}

void CNNNetworkNGraphImpl::transformConstants() {
    if (!cnnNetwork)
        convertToCNNNetworkImpl();
    // Remove all redundant constant and convert unsupported precisions
    ConstTransformer transformator(cnnNetwork.get());
    transformator.fullTrim();
}

bool details::CNNNetworkNGraphImpl::has_f16_constants(const std::shared_ptr<::ngraph::Function> &function) const {
    for (auto & layer : function->get_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::Constant>(layer) && layer->output(0).get_element_type() == ::ngraph::element::f16) {
            return true;
        }
    }
    return false;
}

void InferenceEngine::details::CNNLayerCreator::on_adapter(const std::string& name,
                                                           ::ngraph::ValueAccessor<void>& adapter) {
    if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::element::Type>>(&adapter)) {
        auto type = static_cast<::ngraph::element::Type&>(*a);
        params[name] = details::ngraph::convertPrecision(type).name();
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::PartialShape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::PartialShape&>(*a);
        for (size_t i = 0; i < static_cast<size_t>(shape.rank()); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(static_cast<size_t>(shape[i]));
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Shape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::Shape&>(*a);
        for (size_t i = 0; i < shape.size(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(static_cast<size_t>(shape[i]));
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Strides>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::Strides&>(*a);
        for (size_t i = 0; i < shape.size(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(static_cast<size_t>(shape[i]));
        }
        params[name] = dims;
    }
}

InferenceEngine::details::CNNLayerCreator::CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node): node(node) {
    addSpecificCreator({"Parameter"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                         const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Input",
            details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        return res;
    });
    // TODO - Remove "GreaterEq" once ngraph transitions to GreaterEqual
    addSpecificCreator({"Eltwise", "Subtract", "Power", "Maximum", "Divide", "Greater", "GreaterEqual", "FloorMod", "LogicalOr", "LogicalAnd", "LogicalXor",
        "GreaterEq", "Less", "LessEqual", "Equal", "NotEqual", "Multiply", "Add"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                                 const std::map<std::string, std::string> params) -> CNNLayerPtr {
            LayerParams attrs = {node->get_friendly_name(), "Eltwise",
                details::ngraph::convertPrecision(node->get_output_element_type(0))};
            auto res = std::make_shared<EltwiseLayer>(attrs);
            res->params = params;
            if (node->description() == "Maximum") {
                res->params["operation"] = "max";
            } else if (node->description() == "Power") {
                res->params["operation"] = "pow";
            } else if (node->description() == "Subtract") {
                res->params["operation"] = "sub";
            } else if (node->description() == "Divide") {
                res->params["operation"] = "div";
            } else if (node->description() == "LessEqual") {
                res->params["operation"] = "less_equal";
            } else if (node->description() == "Less") {
                res->params["operation"] = "less";
            } else if (node->description() == "Equal") {
                res->params["operation"] = "equal";
            } else if (node->description() == "NotEqual") {
                res->params["operation"] = "not_equal";
            } else if (node->description() == "FloorMod") {
                res->params["operation"] = "floor_mod";
            } else if (node->description() == "Multiply") {
                res->params["operation"] = "prod";
            } else if (node->description() == "Add") {
                res->params["operation"] = "sum";
            } else if (node->description() == "Greater") {
                res->params["operation"] = "greater";
            } else if (node->description() == "GreaterEq") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "GreaterEqual") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "LogicalOr") {
                res->params["operation"] = "logical_or";
            } else if (node->description() == "LogicalAnd") {
                res->params["operation"] = "logical_and";
            } else if (node->description() == "LogicalXor") {
                res->params["operation"] = "logical_xor";
            } else if (node->description() == "Eltwise") {
                auto castedLayer = std::dynamic_pointer_cast<::ngraph::op::Eltwise>(node);
                if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << attrs.type << " layer " << attrs.name;
                std::string type;
                switch (castedLayer->eltwise_type) {
                case ELTWISE_TYPE::Sum:
                    type = "sum";
                    break;
                case ELTWISE_TYPE::Prod:
                    type = "prod";
                    break;
                default:
                    THROW_IE_EXCEPTION << "Not supported eltwise type!";
                }

                res->params["operation"] = type;
            }
            return res;
        });
    addSpecificCreator({"Concat"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ConcatLayer>(attrs);
        res->params = params;
        return res;
    });
    addSpecificCreator({"AvgPool", "MaxPool"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                  const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Pooling",
            details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<PoolingLayer>(attrs);
        res->params = params;
        if (res->params.find("auto_pad") != res->params.end() &&
            details::CaselessEq<std::string>()(res->params["auto_pad"], "EXPLICIT"))
            res->params.erase("auto_pad");

        if (res->params.find("exclude_pad") != res->params.end()) {
            res->params["exclude-pad"] = res->params["exclude_pad"];
            res->params.erase("exclude_pad");
        }

        if (node->description() == "MaxPool") {
            res->params["pool-method"] = "max";
        } else if (node->description() == "AvgPool") {
            res->params["pool-method"] = "avg";
        }
        return res;
    });
    addSpecificCreator({"Select"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SelectLayer>(attrs);
        res->params = params;
        return res;
    });
}

CNNLayerPtr InferenceEngine::details::CNNLayerCreator::create() {
    auto one_from = [](const std::string& desc, const std::vector<std::string>& descs) -> bool {
        for (const auto& d : descs) {
            if (details::CaselessEq<std::string>()(d, desc)) return true;
        }
        return false;
    };
    LayerParams attrs = {node->get_friendly_name(), node->description(),
                         details::ngraph::convertPrecision(node->get_output_element_type(0))};
    if (creators.find(node->description()) != creators.end())
        return creators[node->description()](node, params);

    auto res = std::make_shared<CNNLayer>(attrs);
    res->params = params;
    return res;
}
