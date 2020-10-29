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

#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp>

#include "ngraph_ops/eltwise.hpp"
#include "graph_tools.hpp"
#include "graph_transformer.h"
#include "ie_util_internal.hpp"
#include "ie_cnn_layer_builder_ngraph.h"
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
    return specialized_function;
}

// WA: for cnnNetwork ngraph constructor
CNNNetwork::CNNNetwork(const std::shared_ptr<const ngraph::Function>& graph) {
    // Copy nGraph function
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

    // Add shape infer method for old operations which are not included to opset1 and opset2
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
        auto graph = cloneFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(graph);

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

    ::ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(graph);
    ::ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(graph);
    cnnNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(graph, *this);
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
    return result;
}

void CNNNetworkNGraphImpl::transformConstants() {
    if (!cnnNetwork)
        convertToCNNNetworkImpl();
    // Remove all redundant constant and convert unsupported precisions
    ConstTransformer transformator(cnnNetwork.get());
    transformator.fullTrim();
}

void InferenceEngine::details::CNNLayerCreator::on_adapter(const std::string& name,
                                                           ::ngraph::ValueAccessor<void>& adapter) {
    if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::element::Type>>(&adapter)) {
        auto type = static_cast<::ngraph::element::Type&>(*a);
        params[name] = details::ngraph::convertPrecision(type).name();
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::PartialShape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::PartialShape&>(*a);
        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i].get_length());
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Shape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::Shape&>(*a);
        for (size_t i = 0; i < shape.size(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i]);
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Strides>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::Strides&>(*a);
        for (size_t i = 0; i < shape.size(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i]);
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
    addSpecificCreator({"BinaryConvolution"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BinaryConvolutionLayer>(attrs);

        // todo: investigate difference between ngraph parameters for BinConvolution and the implementation above
        // this leads to accuracy issue for Precollected_ONNX_ResNet50_88percentinto1bit e2e test
        // res->params = params;

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(node);

        std::string value;
        for (const auto& val : castedLayer->get_pads_begin()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_begin"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_pads_end()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_end"] = value;

        switch (castedLayer->get_auto_pad()) {
            case ::ngraph::op::PadType::SAME_UPPER:
                res->params["auto_pad"] = "same_upper";
                break;
            case ::ngraph::op::PadType::SAME_LOWER:
                res->params["auto_pad"] = "same_lower";
                break;
            case ::ngraph::op::PadType::VALID:
                res->params["auto_pad"] = "valid";
                break;
            default:
                break;
        }

        value.clear();
        for (const auto& val : castedLayer->get_strides()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["strides"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_dilations()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["dilations"] = value;

        // Restore kernel size and output
        const auto& shape = castedLayer->get_input_shape(1);
        res->params["output"] = Builder::asString(shape[0]);

        value.clear();
        for (size_t i = 2; i < shape.size(); i++) {
            if (!value.empty()) value += ",";
            value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = value;

        switch (castedLayer->get_mode()) {
            case ::ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT:
                res->params["mode"] = "xnor-popcount";
        }

        auto weights_shape = castedLayer->input(1).get_source_output().get_shape();
        res->params["input"] = Builder::asString(weights_shape[1]);
        res->params["pad_value"] = Builder::asString(castedLayer->get_pad_value());

        Builder::NodeConverter<::ngraph::op::Constant> converter;

        const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"SpaceToBatch"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToBatchLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"BatchToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::ngraph::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BatchToSpaceLayer>(attrs);
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
