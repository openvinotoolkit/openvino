// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_ngraph_impl.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <math.h>

#include <cassert>
#include <details/caseless.hpp>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/get_output_element_elimination.hpp>
#include <set>
// #include <shape_infer/ie_reshaper.hpp>
#include <string>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp>

#include "ngraph_ops/eltwise.hpp"
#include "graph_tools.hpp"
#include "exec_graph_info.hpp"
#include "graph_transformer.h"
#include "ie_util_internal.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_profiling.hpp"
#include "network_serializer.h"
#include "generic_ie.hpp"
#include "convert_function_to_cnn_network.hpp"
#include <shape_infer/built-in/ie_built_in_holder.hpp>

using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

static std::shared_ptr<ngraph::Function> copyFunction(const std::shared_ptr<const ngraph::Function>& func,
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
            new_shapes.emplace_back(parameter->get_partial_shape());
        }
        new_types.emplace_back(parameter->get_element_type());
    }
    IE_ASSERT(original_parameters.size() == new_types.size());
    IE_ASSERT(original_parameters.size() == new_shapes.size());

    // TODO: remove const cast if specialize function works with constant ngraph function
    auto specialized_function = ::ngraph::specialize_function(std::const_pointer_cast<ngraph::Function>(func), new_types, new_shapes,
                                                              std::vector<void*>(new_shapes.size(), nullptr), constFolding, true);
    // TODO: remove this code after the fix on the nGraph side
    ::ngraph::pass::GetOutputElementElimination goe_elimination;
    for (auto n : specialized_function->get_ops()) {
        goe_elimination.run_on_node(n);
    }
    specialized_function->set_friendly_name(func->get_friendly_name());
    return specialized_function;
}

// WA: for cnnNetwork ngraph constructor
CNNNetwork::CNNNetwork(const std::shared_ptr<const ngraph::Function>& graph) {
    if (graph == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized: 'graph' object is empty";
    }

    // Copy nGraph function
    if (graph == nullptr)
        THROW_IE_EXCEPTION << "Cannot create CNNNetwork from empty nGraph function!";
    network = std::make_shared<CNNNetworkNGraphImpl>(copyFunction(graph, false, {}));
    actual = network.get();
    if (actual == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
    }
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
        const auto precision = details::convertPrecision(output.get_element_type());
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
    : _ngraph_function(nGraph) {
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

    // Add shape infer method for old operations which are not included to opset1, opset2 and opset3
    ::ngraph::op::GenericIE::addExtension(_ngraph_function, std::make_shared<ShapeInfer::BuiltInShapeInferHolder>());

    reshape();
    for (const auto& layer : _ngraph_function->get_parameters()) {
        std::string outName = layer->get_friendly_name();
        IE_ASSERT(layer->get_output_size() == 1);  // Parameter as only singly output port

        DataPtr& ptr = _data[outName];
        IE_ASSERT(ptr);  // Data must be allocated after the reshape method

        keep_input_info(*this, ptr);
    }
    for (auto& output : _outputData) {
        // Convert precision into native format. Be consistent with possible conversion to CNNNetwork later.
        if (output.second->getPrecision() == Precision::I64) {
            output.second->setPrecision(Precision::I32);
        } else if (output.second->getPrecision() != Precision::FP32 &&
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
    if (cnnNetwork) cnnNetwork->setInputInfo(data);
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

DataPtr& CNNNetworkNGraphImpl::getData(const std::string& name) {
    IE_SUPPRESS_DEPRECATED_START
    return getData(name.c_str());
    IE_SUPPRESS_DEPRECATED_END
}

void CNNNetworkNGraphImpl::getName(char* pName, size_t len) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getName(pName, len);
        return;
    }
    // Description buffer will preserve garbage if external pointer not initialized
    if (len < 1) return;
    memset(pName, 0, len);
    DescriptionBuffer(pName, len) << _ngraph_function->get_friendly_name();
}

const std::string& CNNNetworkNGraphImpl::getName() const noexcept {
    if (cnnNetwork) {
        return cnnNetwork->getName();
    }
    return _ngraph_function->get_friendly_name();
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
    if (!cnnNetwork) {
        const_cast<CNNNetworkNGraphImpl *>(this)->convertToCNNNetworkImpl();
    }
    if (!cnnNetwork) return GENERAL_ERROR;
    return cnnNetwork->getLayerByName(layerName, out, resp);
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName, size_t outputIndex,
                                           ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(addOutput)
    if (cnnNetwork) {
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
    if (cnnNetwork)
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
                ::ngraph::pass::ConvertOneHotToOneHotIE().run_on_function(specialized_ngraph_function);
            }
            specialized_ngraph_function->validate_nodes_and_infer_types();

#if 0
            for (const auto &op : specialized_ngraph_function->get_ordered_ops()) {
                cout << "[ " <<  op->description() << " ] " << op->get_friendly_name() << endl;
                cout << "    Inputs: ";
                for (const auto &in : op->inputs()) {
                    cout << "[" << in.get_element_type().get_type_name() << "]";
                    if (in.get_partial_shape().is_dynamic()) {
                        cout << "dyn_shape";
                    } else {
                        cout << "{";
                        bool first = true;
                        for (auto i : in.get_shape()) {
                            if (!first) cout << ",";
                            cout << i;
                            first = false;
                        }
                        cout << "} ";
                    }
                }
                cout << endl << "    Outputs: ";
                for (const auto &in : op->outputs()) {
                    cout << "[" << in.get_element_type().get_type_name() << "]";
                    if (in.get_partial_shape().is_dynamic()) {
                        cout << "dyn_shape";
                    } else {
                        cout << "{";
                        bool first = true;
                        for (auto i : in.get_shape()) {
                            if (!first) cout << ",";
                            cout << i;
                            first = false;
                        }
                        cout << "} ";
                    }
                }
                cout << endl;
            }
#endif
            std::unordered_set<std::string> opName;
            for (const auto & layer : specialized_ngraph_function->get_ordered_ops()) {
                if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
                    IE_ASSERT(layer->get_inputs().size() == 1);
                    const auto& input = layer->get_inputs()[0];
                    std::string outName = input.get_output().get_node()->get_friendly_name();
                    if (input.get_output().get_node()->get_output_size() != 1)
                        outName += "." + std::to_string(input.get_output().get_index());
                    addOutput(outName);
                    continue;
                }
                if (opName.find(layer->get_friendly_name()) != opName.end())
                    THROW_IE_EXCEPTION << "All operations in nGraph function should have unique friendly names!";
                opName.insert(layer->get_friendly_name());
                for (const auto& output : layer->outputs()) {
                    std::string outName = layer->get_friendly_name();
                    if (layer->outputs().size() != 1)
                        outName += "." + std::to_string(output.get_index());
                    createDataForResult(output, outName, _data[outName]);
                }
            }
        }
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }

    return OK;
}

StatusCode CNNNetworkNGraphImpl::AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                                              InferenceEngine::ResponseDesc* resp) noexcept {
    if (!cnnNetwork) {
        ::ngraph::op::GenericIE::addExtension(_ngraph_function, extension);
    }
    return cnnNetwork ? cnnNetwork->AddExtension(extension, resp) : OK;
}

StatusCode CNNNetworkNGraphImpl::serialize(const std::string& xmlPath, const std::string& binPath,
                                           ResponseDesc* resp) const noexcept {
    auto network = cnnNetwork;
    if (!network) {
        // TODO: once Serialization::Serialize supports true IR v10
        // remove this conversion and WA for execution graph
        try {
            bool isExecutionGraph = true;
            for (const auto & op : _ngraph_function->get_ops()) {
                auto & rtInfo = op->get_rt_info();
                if (rtInfo.find(ExecGraphInfoSerialization::PERF_COUNTER) == rtInfo.end()) {
                    isExecutionGraph = false;
                    break;
                }
            }
            if (isExecutionGraph) {
                Serialization::Serialize(xmlPath, binPath, (InferenceEngine::ICNNNetwork&)*this);
                return OK;
            }
        } catch (const InferenceEngineException& e) {
            return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
        } catch (const std::exception& e) {
            return DescriptionBuffer(UNEXPECTED, resp) << e.what();
        } catch (...) {
            return DescriptionBuffer(UNEXPECTED, resp);
        }

        auto graph = cloneFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(graph);

        ::ngraph::pass::CommonOptimizations().run_on_function(graph);
        ::ngraph::pass::ConvertOpSet3ToOpSet2().run_on_function(graph);
        ::ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(graph);
        ::ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(graph);
        network = InferenceEngine::details::convertFunctionToICNNNetwork(graph, *this);
    }
    if (!network) return GENERAL_ERROR;
    return network->serialize(xmlPath, binPath, resp);
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        if (size == getBatchSize())
            return OK;
        if (!cnnNetwork)
            convertToCNNNetworkImpl();
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

StatusCode CNNNetworkNGraphImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    if (cnnNetwork)
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
        convertToCNNNetworkImpl();
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

void CNNNetworkNGraphImpl::convertToCNNNetworkImpl() {
    IE_PROFILING_AUTO_SCOPE(convertToCNNNetworkImpl)
    if (cnnNetwork)
        return;
    auto graph = cloneFunction();
    // Disable shape inference (WA for generic operations)
    ::ngraph::op::GenericIE::DisableReshape noReshape(graph);

    ::ngraph::pass::CommonOptimizations().run_on_function(graph);
    ::ngraph::pass::ConvertOpSet3ToOpSet2().run_on_function(graph);
    ::ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(graph);
    ::ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(graph);
    cnnNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(graph, *this);
}
