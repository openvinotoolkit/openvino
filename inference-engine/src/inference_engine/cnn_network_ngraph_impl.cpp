// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_ngraph_impl.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <math.h>

#include <ie_memcpy.h>
#include <blob_factory.hpp>


#include <cassert>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <ngraph/ngraph.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <set>
#include <string>

#include <transformations/utils/utils.hpp>
#include <transformations/smart_reshape/set_batch_size.hpp>
#include <transformations/smart_reshape/smart_reshape.hpp>
#include "transformations/serialize.hpp"

// TODO: remove this pass usage
#include <legacy/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>

#include "ie_ngraph_utils.hpp"
#include "exec_graph_info.hpp"
#include "ie_itt.hpp"

using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

void CNNNetworkNGraphImpl::createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName,
                                               DataPtr& ptr) {
    const auto isCompatible = [](size_t size, const Layout& l) -> bool {
        switch (size) {
        case 0:
            return l == Layout::SCALAR;
        case 1:
            return l == Layout::C;
        case 2:
            return l == Layout::CN || l == Layout::HW || l == Layout::NC;
        case 3:
            return l == Layout::CHW || l == Layout::HWC;
        case 4:
            return l == Layout::NCHW || l == Layout::NHWC;
        case 5:
            return l == Layout::NCDHW || l == Layout::NDHWC;
        default:
            return false;
        }
    };
    // query shape from ngraph::Parameter output shape and check there are no zeros in it
    SizeVector dims;
    if (output.get_partial_shape().is_static()) {
        dims = output.get_shape();
    }
    for (const auto& dim : dims) {
        if (!dim)
            IE_THROW() << outName << " has zero dimension which is not allowed";
    }

    if (ptr) {
        const auto origLayout = ptr->getTensorDesc().getLayout();
        const auto layout = isCompatible(dims.size(), origLayout) ? origLayout : TensorDesc::getLayoutByDims(dims);
        ptr->reshape(dims, layout);
    } else {
        const auto layout = TensorDesc::getLayoutByDims(dims);
        const auto precision = details::convertPrecision(output.get_element_type());
        ptr.reset(new Data(outName, {precision, dims, layout}));
    }
}

void CNNNetworkNGraphImpl::validateFunctionNames() const {
    // nGraph function parameters and pre-Results operations should have unique names
    std::unordered_set<std::string> unique_names;
    for (const auto& param : _ngraph_function->get_parameters()) {
        if (unique_names.count(param->get_friendly_name())) {
            IE_THROW() << "Function contains several inputs with one friendly name!";
        }
        unique_names.insert(param->get_friendly_name());
    }
    for (const auto& result : _ngraph_function->get_results()) {
        const auto& parent = result->get_input_node_shared_ptr(0);
        auto name = parent->get_friendly_name();
        if (parent->get_output_size() > 1) {
            name += "." + std::to_string(result->get_input_source_output(0).get_index());
        }
        if (unique_names.count(name) && !ngraph::op::is_parameter(parent)) {
            IE_THROW() << "Function contains several inputs and outputs with one friendly name!";
        }
        unique_names.insert(name);
    }
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(
    const std::shared_ptr<Function>& nGraph,
    const std::vector<IExtensionPtr>& exts)
    : _ngraph_function(nGraph), _ie_extensions(exts) {
    // Restore usual attributes for CNNNetwork
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

    validateFunctionNames();

    reshape();
    for (const auto& layer : _ngraph_function->get_parameters()) {
        std::string outName = layer->get_friendly_name();
        IE_ASSERT(layer->get_output_size() == 1);  // Parameter as only singly output port

        // map original names to OpenVINO name
        for (const auto& name : layer->get_output_tensor(0).get_names()) {
            _tensorNames[name] = outName;
        }

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

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const CNNNetwork& network) {
    IE_SUPPRESS_DEPRECATED_START
    const ICNNNetwork& iNetwork = network;
    IE_SUPPRESS_DEPRECATED_END
    const auto net = dynamic_cast<const CNNNetworkNGraphImpl*>(&iNetwork);
    if (network.getFunction() == nullptr || !net) {
        IE_THROW() << "Cannot create CNNNetwork with nGraph from legacy network format!";
    }

    _ngraph_function = ngraph::clone_function(*network.getFunction());
    validateFunctionNames();
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();

    _tensorNames = net->_tensorNames;

    for (const auto& outputInfo : outputs) {
        const auto& name = outputInfo.second->getName();
        DataPtr output = std::make_shared<Data>(name, outputInfo.second->getTensorDesc());
        _outputData[name] = output;
        _data[name] = output;
    }
    for (const auto& inputInfo : inputs) {
        InputInfo::Ptr info = std::make_shared<InputInfo>();
        const auto& name = inputInfo.second->getInputData()->getName();
        DataPtr input = std::make_shared<Data>(name, inputInfo.second->getInputData()->getTensorDesc());
        _data[name] = input;
        info->setInputData(input);
        info->getPreProcess() = inputInfo.second->getPreProcess();
        info->setPrecision(inputInfo.second->getPrecision());
        info->setLayout(inputInfo.second->getLayout());
        _inputData[name] = info;
    }
}

void CNNNetworkNGraphImpl::setInputInfo(InputInfo::Ptr data) {
    _inputData[data->name()] = data;
}

const std::string& CNNNetworkNGraphImpl::getName() const noexcept {
    return _ngraph_function->get_friendly_name();
}

InputInfo::Ptr CNNNetworkNGraphImpl::getInput(const std::string& inputName) const noexcept {
    auto it = _inputData.find(inputName);
    if (it == _inputData.end()) {
        return nullptr;
    }
    return it->second;
}

void CNNNetworkNGraphImpl::getOutputsInfo(OutputsDataMap& out) const noexcept {
    out = _outputData;
}

void CNNNetworkNGraphImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    inputs = _inputData;
}

size_t CNNNetworkNGraphImpl::layerCount() const noexcept {
    return _ngraph_function->get_ops().size();
}

void CNNNetworkNGraphImpl::validate(int version) {
    _ngraph_function->validate_nodes_and_infer_types();
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName, size_t outputIndex,
                                           ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::addOutput");

    try {
        for (const auto & layer : _ngraph_function->get_ops()) {
            // Result can have the same name as previous operation
            if (layer->get_friendly_name() == layerName && !std::dynamic_pointer_cast<ngraph::op::Result>(layer)) {
                // Check that output port exists
                if (layer->outputs().size() <= outputIndex) {
                    return DescriptionBuffer(OUT_OF_BOUNDS, resp)
                    << "port index " << outputIndex << " exceeds the number of layer outputs " << layer->outputs().size();
                }
                std::string outputName = layerName;
                if (layer->outputs().size() != 1) {
                    outputName += "." + std::to_string(outputIndex);
                }

                // Check that we don't have a result for the output port
                for (const auto& port : layer->output(outputIndex).get_target_inputs()) {
                    if (dynamic_cast<ngraph::op::Result*>(port.get_node()))
                        return OK;
                }
                auto result = make_shared<::ngraph::op::Result>(layer->output(outputIndex));
                result->set_friendly_name(outputName);
                _ngraph_function->add_results({result});
                // Check that we cannot add Result to layer with non unique friendly name
                try {
                    validateFunctionNames();
                } catch (...) {
                    _ngraph_function->remove_result(result);
                    throw;
                }

                if (_outputData.count(outputName) == 0) {
                    reshape();
                }
                return OK;
            }
        }
    } catch (...) {
        return GENERAL_ERROR;
    }
    return DescriptionBuffer(NOT_FOUND, resp) << "Cannot add output! Layer " << layerName << " wasn't found!";
}

void CNNNetworkNGraphImpl::addOutput(const ::ngraph::Output<::ngraph::Node> & output) {
    auto dataName = ngraph::op::util::create_ie_output_name(output);
    DataPtr data;
    if (_data.count(dataName))
        data = _data[dataName];
    createDataForResult(output, dataName, data);
    _data[dataName] = data;
    _outputData[dataName] = data;

    // Save original framework names
    for (const auto& name : output.get_tensor().get_names()) {
        _tensorNames[name] = dataName;
    }
}

size_t CNNNetworkNGraphImpl::getBatchSize() const noexcept {
    // TODO Provide adequate implementation.
    // The original code from CNNNetworkImpl just gets the first input and returns the first dimension.
    // This is not correct in general. We can follow the same semantics, but order of inputs should be
    // guaranteed to be the same.
    auto params = _ngraph_function->get_parameters();
    sort(params.begin(), params.end(), [](std::shared_ptr<ngraph::Node> lhs, std::shared_ptr<ngraph::Node> rhs) {
        return lhs->get_friendly_name() < rhs->get_friendly_name();
    });

    for (const auto& param : params) {
        if (param->get_partial_shape().rank().is_dynamic())
            continue;
        auto pshape = param->get_partial_shape();
        auto rank = pshape.rank().get_length();
        // WA: for speech recognition and scalar layouts (copy-past from CNNNetwork)
        if ((rank == 2 || rank > 3) && pshape[0].is_static()) {
            return pshape[0].get_length();
        }
    }
    return 1;
}

void CNNNetworkNGraphImpl::reshape() {
    reshape({});
}

StatusCode
CNNNetworkNGraphImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                              ResponseDesc* responseDesc) noexcept {
    if (inputShapes.empty()) return OK;

    const auto & params = _ngraph_function->get_parameters();

    // Check that we need to do reshape only if input shapes will be changed
    bool needReshape = false;
    for (const auto & param : params) {
        const auto it = inputShapes.find(param->get_friendly_name());
        if (it == inputShapes.end()) {
            continue;
        }
        if (param->get_partial_shape().is_dynamic() || param->get_shape() != it->second) {
            needReshape = true;
            break;
        }
    }

    if (!needReshape) return OK;

    // save original parameters shape
    std::map<std::string, ngraph::PartialShape> originalInputShapes;
    for (const auto & param : params) {
        originalInputShapes[param->get_friendly_name()] = param->get_partial_shape();
    }

    try {
        ngraph::pass::Manager ssr_manager;
        ssr_manager.register_pass<ngraph::pass::SmartReshape>();
        ssr_manager.run_passes(_ngraph_function);

        std::map<std::string, ngraph::PartialShape> reshapeShapes;
        for (const auto & item : inputShapes) {
            reshapeShapes[item.first] = ngraph::PartialShape(item.second);
        }
        reshape(reshapeShapes);
    } catch (std::exception& ex) {
        reshape(originalInputShapes);
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }

    return OK;
}

void
CNNNetworkNGraphImpl::reshape(const std::map<std::string, ngraph::PartialShape>& inputShapes) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::reshape");

    auto params = _ngraph_function->get_parameters();

    bool parameter_replaced = false;
    for (size_t i = 0; i < params.size(); i++) {
        const auto& param = params[i];
        if (inputShapes.find(param->get_friendly_name()) == inputShapes.end())
            continue;
        ::ngraph::PartialShape shape(inputShapes.at(param->get_friendly_name()));
        auto newParam = std::make_shared<::ngraph::op::Parameter>(param->get_element_type(), shape);
        newParam->set_friendly_name(param->get_friendly_name());
        _ngraph_function->replace_parameter(i, newParam);
        parameter_replaced = true;
    }
    if (parameter_replaced)
        _ngraph_function->validate_nodes_and_infer_types();

    const auto& results = _ngraph_function->get_results();
    bool outputs_are_static = all_of(
            begin(results), end(results),
            [](const std::shared_ptr<ngraph::Node>& n){ return n->get_output_partial_shape(0).is_static(); });

    {
        shared_ptr<Function> specialized_ngraph_function = nullptr;
        if (outputs_are_static) {
            specialized_ngraph_function = _ngraph_function;
        } else {
            specialized_ngraph_function = ngraph::clone_function(*_ngraph_function);
            {
                OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::ConvertToLegacy");
                ::ngraph::pass::Manager manager;
                // resolves dynamism by replacing dynamic operation with static version
                manager.register_pass<::ngraph::pass::ConvertNMS5ToLegacyMatcher>(false);
                manager.register_pass<::ngraph::pass::DisableConvertConstantFoldingOnConstPath>();
                manager.register_pass<::ngraph::pass::ConstantFolding>();
                // OneHotToLegacy changes output precision
                manager.register_pass<::ngraph::pass::ConvertOneHotToOneHotIEMatcher>()->detect_output_type(
                        specialized_ngraph_function);
                manager.run_passes(specialized_ngraph_function);
            }
            specialized_ngraph_function->validate_nodes_and_infer_types();
        }

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
        for (const auto &result : specialized_ngraph_function->get_results()) {
            addOutput(result->input_value(0));
        }

        for (const auto &parameter : specialized_ngraph_function->get_parameters()) {
            const auto &outName = parameter->get_friendly_name();
            if (opName.find(outName) != opName.end()) {
                IE_THROW() << "All operations in nGraph function should have unique friendly names!";
            }
            opName.insert(outName);
            createDataForResult(parameter, outName, _data[outName]);
        }
    }
}

StatusCode CNNNetworkNGraphImpl::serialize(const std::string& xmlPath,
                                           const std::string& binPath,
                                           ResponseDesc* resp) const noexcept {
    try {
        std::map<std::string, ngraph::OpSet> custom_opsets;
        for (const auto& extension : _ie_extensions) {
            auto opset = extension->getOpSets();
            custom_opsets.insert(begin(opset), end(opset));
        }
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::Serialize>(
            xmlPath, binPath, ngraph::pass::Serialize::Version::IR_V10,
            custom_opsets);
        manager.run_passes(_ngraph_function);
    } catch (const Exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkNGraphImpl::serialize(std::ostream& xmlBuf,
                                           std::ostream& binBuf,
                                           ResponseDesc* resp) const noexcept {
    try {
        std::map<std::string, ngraph::OpSet> custom_opsets;
        for (const auto& extension : _ie_extensions) {
            auto opset = extension->getOpSets();
            custom_opsets.insert(begin(opset), end(opset));
        }
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::Serialize>(
            xmlBuf, binBuf, ngraph::pass::Serialize::Version::IR_V10,
            custom_opsets);
        manager.run_passes(_ngraph_function);
    } catch (const Exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkNGraphImpl::serialize(std::ostream& xmlBuf,
                                           Blob::Ptr& binBlob,
                                           ResponseDesc* resp) const noexcept {
    try {
        std::map<std::string, ngraph::OpSet> custom_opsets;
        for (const auto& extension : _ie_extensions) {
            auto opset = extension->getOpSets();
            custom_opsets.insert(begin(opset), end(opset));
        }

        std::stringstream binBuf;
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::Serialize>(
            xmlBuf, binBuf, ngraph::pass::Serialize::Version::IR_V10,
            custom_opsets);
        manager.run_passes(_ngraph_function);

        std::streambuf* pbuf = binBuf.rdbuf();
        unsigned long bufSize = binBuf.tellp();

        TensorDesc tensorDesc(Precision::U8, { bufSize }, Layout::C);
        binBlob = make_shared_blob<uint8_t>(tensorDesc);
        binBlob->allocate();
        pbuf->sgetn(binBlob->buffer(), bufSize);
    } catch (const Exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkNGraphImpl::getOVNameForTensor(std::string& ov_name, const std::string& orig_name, ResponseDesc* resp) const noexcept {
    if (_tensorNames.find(orig_name) == _tensorNames.end())
        return DescriptionBuffer(NOT_FOUND, resp) << "Framework tensor with name \"" << orig_name << "\" was not mapped to OpenVINO data!";
    ov_name = _tensorNames.at(orig_name);
    return OK;
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        if (getBatchSize() == size) return OK;
        auto original_parameters = _ngraph_function->get_parameters();
        if (original_parameters.empty()) return DescriptionBuffer(GENERAL_ERROR, responseDesc) << "Cannot set batch! Function doesn't contain parameters!";

        stringstream ss;
        ss << " Please use reshape method instead. Original parameter shapes are: ";
        for (size_t i = 0; i < original_parameters.size(); ++i) {
            if (i) ss << ", ";
            ss << "\"" << original_parameters[i]->get_friendly_name() << "\": " << original_parameters[i]->get_partial_shape();
        }

        // ill-formed logic from the past setBatchSize (we keep it for backward-compatibility)
        const auto first_parameter = *std::min_element(original_parameters.begin(), original_parameters.end(),
            [](std::shared_ptr<ngraph::Node> lhs, std::shared_ptr<ngraph::Node> rhs){return lhs->get_friendly_name() < rhs->get_friendly_name();});
        const auto first_parameter_pshape = first_parameter->get_partial_shape();
        if (first_parameter_pshape.is_dynamic()) return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) <<
            "Cannot set batch! Function contains parameter with partially defined shape!" << ss.str();
        const auto first_parameter_rank = first_parameter_pshape.rank().get_length();
        if (first_parameter_rank == 0 || first_parameter_rank == 1 || first_parameter_rank == 3) return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) <<
            "Cannot set batch! Function contains 0D/1D/3D parameter with unknown batch dimension placement." << ss.str();

        std::map<std::string, std::vector<size_t>> inShapes;
        for (const auto &parameter : original_parameters) {
            const auto & pshape = parameter->get_partial_shape();
            if (pshape.is_dynamic()) return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) <<
                "Cannot set batch! Function contains parameter with partially defined shape!" << ss.str();
            const auto & rank = pshape.rank().get_length();
            if (rank == 0) return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) <<
                "Cannot set batch! Function contains 0D/1D/3D parameter with unknown batch dimension placement." << ss.str();
            auto shape = parameter->get_shape();
            shape[0] = {static_cast<size_t>(std::ceil(size * static_cast<float>(shape[0]) / static_cast<float>(getBatchSize())))};
            inShapes[parameter->get_friendly_name()] = shape;
        }
        ngraph::pass::Manager ssr_manager;
        ssr_manager.register_pass<ngraph::pass::SetBatchSize>();
        ssr_manager.run_passes(_ngraph_function);

        return reshape(inShapes, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}
