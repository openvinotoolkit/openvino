// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>
#include "cnn_network_ngraph_impl.hpp"
#include <memory>
#include <map>
#include <set>
#include <string>
#include <cassert>
#include <shape_infer/ie_reshaper.hpp>
#include "debug.h"
#include "graph_tools.hpp"
#include <vector>
#include <math.h>
#include "network_serializer.h"
#include "ie_profiling.hpp"
#include <ngraph/pass/visualize_tree.hpp>
#include "ie_ngraph_utils.hpp"
#include <cpp/ie_cnn_network.h>

#include <ngraph/pass/constant_folding.hpp>

#include "ngraph_ops/dummy.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph_ops/matmul_bias.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/quantize_conv_bias_fused.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/interp.hpp"
#include "cnn_network_impl.hpp"

#include <ngraph/op/add.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/dot.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/conv_fused.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/fused/squeeze.hpp>
#include <ngraph/op/fused/unsqueeze.hpp>
#include <ngraph/op/get_output_element.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/subtract.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/topk.hpp>
#include <ngraph.hpp>

#include <transform/transformations/conv_bias_fusion.hpp>
#include <transform/transformations/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transform/transformations/convert_mul_or_add_finally.hpp>
#include <transform/transformations/matmul_bias_fusion.hpp>
#include <transform/transformations/quantizeconv_dequantize_fusion.hpp>
#include <transform/transformations/constant_eltwise_reduction.hpp>
#include <transform/transformations/convert_quantize_conv_elimination.hpp>
#include <transform/transformations/convert_broadcast_to_tiles.hpp>
#include <transform/transformations/convert_tile_to_ie_tile.hpp>
#include <transform/transformations/reshape_constant_folding.hpp>
#include <transform/transformations/convert_prior_clustered_to_ie_clustered.hpp>
#include <transform/transformations/convert_pror_to_ie_prior.hpp>
#include <transform/transformations/convert_interpolate_to_interp_or_resample.hpp>
#include <transform/transformations/convert_strided_slice_to_crop.hpp>

#include <ie_ir_reader.hpp>


using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

// WA: for cnnNetwork ngraph constructor
CNNNetwork::CNNNetwork(const std::shared_ptr<ngraph::Function>& graph) {
    network = std::make_shared<CNNNetworkNGraphImpl>(graph);
    actual = network.get();
    if (actual == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
    }
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const std::shared_ptr<Function>& nGraph)
    : _ngraph_function(nGraph), _stats(new CNNNetworkStatsImpl()) {
    // Restore usual attributes for ICNNNetwork
    // TODO: Move this code to corresponding function that should deliver required information
    // directly from ngraph_function, not cashing it here.
    auto keep_input_info = [](CNNNetworkNGraphImpl& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        Precision prc = info->getPrecision();

        // Convert precision into native format (keep element size)
        prc = prc == Precision::Q78 ? Precision::I16 :
            prc == Precision::FP16 ? Precision::FP32 :
            static_cast<Precision::ePrecision>(prc);

        info->setPrecision(prc);
        network.setInputInfo(info);
    };

    auto create_data_for_output = [](const ::ngraph::descriptor::Output& output, const std::string& outName, DataPtr& ptr) {
        assert(!ptr);  // no data allocated for this parameter because we are meeting it for the first time

        // query shape from ngraph::Parameter output shape and check there are no zeros in it
        SizeVector dims = output.get_shape();
        for (const auto& dim : dims) {
            if (!dim)
                THROW_IE_EXCEPTION << outName << " has zero dimension that is not allowable";
        }

        // assign the same output shape for data object that will be used in input preprocessing info
        ptr.reset(new Data(outName, { details::ngraph::convertPrecision(output.get_element_type()),
                    dims, TensorDesc::getLayoutByDims(dims) }));
    };

    // TODO: check if can just iterate over all parameters of a function without iterating over all nodes
    for (const auto &layer : _ngraph_function->get_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
            std::string outName = layer->get_friendly_name();
            assert(layer->get_output_size() == 1);  // Parameter as only singly output port

            DataPtr& ptr = getData(outName.c_str());  // FIXME: don't use obsolete getData
            assert(!ptr);  // no data allocated for this parameter because we are meeting it for the first time

            create_data_for_output(layer->get_outputs()[0], outName, ptr);
            if (ptr->getCreatorLayer().lock())
                THROW_IE_EXCEPTION << "two layers set to the same output [" << outName << "]";

            keep_input_info(*this, ptr);
        } else if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_inputs().size() == 1);
            const auto& input = layer->get_inputs()[0];
            std::string outName = input.get_output().get_node()->get_friendly_name();
            if (input.get_output().get_node()->get_output_size() != 1)
                outName += "." + std::to_string(input.get_output().get_index());
            create_data_for_output(layer->get_outputs()[0], outName, getData(outName.c_str()));
            addOutput(outName);
        }
    }
}

CNNNetworkNGraphImpl::~CNNNetworkNGraphImpl() {
    for (auto& data : _data) {
        if (!data.second)
            continue;
        for (auto& input : data.second->getInputTo()) {
            if (!input.second)
                continue;
            input.second.reset();
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
    if (cnnNetwork)
        return cnnNetwork->getInput(inputName);
    auto it = _inputData.find(inputName);
    if (it == _inputData.end()) {
        return nullptr;
    }
    return it->second;
}

Precision CNNNetworkNGraphImpl::getPrecision() const noexcept {
    return convertToCNNNetworkImpl()->getPrecision();
}

void CNNNetworkNGraphImpl::getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept {
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

size_t CNNNetworkNGraphImpl::layerCount()  const noexcept {
    if (cnnNetwork)
        return cnnNetwork->layerCount();
    return _ngraph_function->get_ops().size();
}

void CNNNetworkNGraphImpl::addLayer(const CNNLayerPtr& layer) noexcept {
    convertToCNNNetworkImpl()->addLayer(layer);
}

void CNNNetworkNGraphImpl::validate(int version) {
    if (cnnNetwork)
        cnnNetwork->validate();
    else
        _ngraph_function->validate_nodes_and_infer_types();
}

StatusCode CNNNetworkNGraphImpl::getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept {
    return convertToCNNNetworkImpl()->getLayerByName(layerName, out, resp);
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept {
    return convertToCNNNetworkImpl()->addOutput(layerName, outputIndex, resp);
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
    return 0;
}

StatusCode
CNNNetworkNGraphImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                        ResponseDesc* responseDesc) noexcept {
    if (cnnNetwork)
        return cnnNetwork->reshape(inputShapes, responseDesc);
    auto params = _ngraph_function->get_parameters();
    for (const auto& item : inputShapes) {
        size_t idx = 0;
        for (; idx < params.size(); idx++) {
            if (params[idx]->get_friendly_name() == item.first) {
                ::ngraph::PartialShape shape(item.second);
                auto param = std::make_shared<::ngraph::op::Parameter>(params[idx]->get_element_type(), shape);
                _ngraph_function->replace_parameter(idx, param);
                break;
            }
        }
        if (idx == params.size())
            return GENERAL_ERROR;
    }
    _ngraph_function->validate_nodes_and_infer_types();
    return OK;
}

StatusCode
CNNNetworkNGraphImpl::AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                             InferenceEngine::ResponseDesc* resp) noexcept {
    return convertToCNNNetworkImpl()->AddExtension(extension, resp);
}

StatusCode CNNNetworkNGraphImpl::serialize(const std::string &xmlPath, const std::string &binPath, ResponseDesc* resp) const noexcept {
    return convertToCNNNetworkImpl()->serialize(xmlPath, binPath, resp);
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    return convertToCNNNetworkImpl()->setBatchSize(size, responseDesc);
}

StatusCode CNNNetworkNGraphImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    return convertToCNNNetworkImpl()->setBatchSizeReshape(size, responseDesc);
}

std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> CNNNetworkNGraphImpl::convertToCNNNetworkImpl() const {
    if (!cnnNetwork) {
        auto result = convertFunctionToICNNNetwork(_ngraph_function);
        auto cnnResult = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(result);
        if (!cnnResult)
            THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";
        return cnnResult;
    }

    return cnnNetwork;
}

std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> CNNNetworkNGraphImpl::convertToCNNNetworkImpl() {
    if (!cnnNetwork) {
        // TODO Implement accurate translation considering already gathered structures in _data.
        auto result = convertFunctionToICNNNetwork(_ngraph_function);

        // update input preprocessing info
        InputsDataMap resultInputDataMap;
        result->getInputsInfo(resultInputDataMap);
        InputsDataMap thisInputDataMap;
        getInputsInfo(thisInputDataMap);
        IE_ASSERT(resultInputDataMap.size() == thisInputDataMap.size());
        for (auto i : resultInputDataMap) {
            auto& thisInputData = *thisInputDataMap[i.first];
            i.second->setPrecision(thisInputData.getPrecision());
            i.second->setLayout(thisInputData.getLayout());
            i.second->getPreProcess() = thisInputData.getPreProcess();
        }

        cnnNetwork = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(result);

        if (!cnnNetwork)
            THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";
    }

    return cnnNetwork;
}
