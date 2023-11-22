// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_ngraph_impl.hpp"

#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "blob_factory.hpp"
#include "cpp/ie_cnn_network.h"
#include "ie_common.h"
#include "ie_memcpy.h"
#include "ie_ngraph_utils.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/smart_reshape/set_batch_size.hpp"
#include "transformations/smart_reshape/smart_reshape.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

void CNNNetworkNGraphImpl::createDataForResult(const ::ngraph::Output<::ngraph::Node>& output,
                                               const std::string& outName,
                                               DataPtr& ptr) {
    const auto isCompatible = [](int64_t size, const Layout& l) -> bool {
        switch (size) {
        case -1:
            return l == Layout::BLOCKED;
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
    auto shape = output.get_partial_shape();
    SizeVector dims(1, 0);
    if (shape.rank().is_static()) {
        dims.resize(shape.size(), 0);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i].get_max_length() != -1)  // dimension has an estimation
                dims[i] = shape[i].get_max_length();
        }
    }
    // query shape from ngraph::Parameter output shape and check there are no zeros in it
    for (const auto& dim : shape) {
        if (dim.is_static() && dim.get_length() == 0)
            IE_THROW() << outName << " has zero dimension which is not allowed";
    }

    auto rank = shape.rank().is_static() ? shape.rank().get_length() : -1;
    const Layout rankLayout = rank < 0 ? Layout::BLOCKED : TensorDesc::getLayoutByRank(rank);
    if (ptr) {
        const auto origLayout = ptr->getTensorDesc().getLayout();
        const auto layout = isCompatible(rank, origLayout) ? origLayout : rankLayout;
        ptr->reshape(dims, layout);
    } else {
        const auto precision = details::convertPrecision(output.get_element_type());
        ptr.reset(new Data(outName, {precision, dims, rankLayout}));
    }
}

void CNNNetworkNGraphImpl::validateFunctionNames() const {
    // nGraph function parameters and pre-Results operations should have unique names
    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>> unique_names;
    for (const auto& param : _ngraph_function->get_parameters()) {
        if (unique_names.count(param->get_friendly_name())) {
            IE_THROW() << "Function contains several inputs with one friendly name!";
        }
        unique_names.insert({param->get_friendly_name(), param});
    }
    for (const auto& result : _ngraph_function->get_results()) {
        const auto& parent = result->get_input_node_shared_ptr(0);
        auto name = parent->get_friendly_name();
        if (parent->get_output_size() > 1) {
            name += "." + std::to_string(result->get_input_source_output(0).get_index());
        }
        if (unique_names.count(name) && !ov::op::util::is_parameter(parent) && parent != unique_names.at(name)) {
            IE_THROW() << "Function contains several inputs and outputs with one friendly name: " << name;
        }
        unique_names.insert({name, parent});
    }
}

ngraph::element::Type details::toLegacyType(const ngraph::element::Type& ngraph_type, bool input) {
    if (input) {
        return ngraph_type == ngraph::element::f16 ? ngraph::element::f32 : ngraph_type;
    } else {
        if (ngraph_type == ngraph::element::i64 || ngraph_type == ngraph::element::u64 ||
            ngraph_type == ngraph::element::i32 || ngraph_type == ngraph::element::u32) {
            return ngraph::element::i32;
        } else if (ngraph_type == ngraph::element::i8 || ngraph_type == ngraph::element::u8 ||
                   ngraph_type == ngraph::element::i16 || ngraph_type == ngraph::element::u16) {
            return ngraph::element::i8;
        } else if (ngraph_type != ngraph::element::f32) {
            return ngraph::element::f32;
        }
    }

    return ngraph_type;
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const std::shared_ptr<Function>& nGraph,
                                           const std::vector<IExtensionPtr>& exts,
                                           bool newAPI)
    : _ngraph_function(nGraph),
      _ie_extensions(exts),
      _new_api(newAPI) {
    {
        ov::pass::Manager m;
        using namespace ov::pass;
        REGISTER_PASS(m, EliminateScatterUpdate)
        REGISTER_PASS(m, RemoveConcatZeroDimInput)
        REGISTER_PASS(m, RemoveMultiSubGraphOpDanglingParamsResults)
        REGISTER_PASS(m, FoldSubgraphEmptyInputs)
        m.run_passes(_ngraph_function);
    }
    // Restore usual attributes for CNNNetwork
    auto keep_input_info = [=](CNNNetworkNGraphImpl& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);

        if (!_new_api) {
            Precision prc = info->getPrecision();

            // Convert precision into native format (keep element size)
            prc = prc == Precision::Q78
                      ? Precision::I16
                      : prc == Precision::FP16 ? Precision::FP32 : static_cast<Precision::ePrecision>(prc);

            info->setPrecision(details::convertPrecision(toLegacyType(details::convertPrecision(prc), true)));
        }

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

    if (!_new_api) {
        for (auto& output : _outputData) {
            // Convert precision into native format. Be consistent with possible conversion to CNNNetwork later.
            output.second->setPrecision(details::convertPrecision(
                toLegacyType(details::convertPrecision(output.second->getPrecision()), false)));
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
        const auto& inData = inputInfo.second->getInputData();
        DataPtr input = std::make_shared<Data>(name, inData->getTensorDesc());
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

size_t CNNNetworkNGraphImpl::layerCount() const {
    return _ngraph_function->get_ops().size();
}

void CNNNetworkNGraphImpl::validate(int version) {
    _ngraph_function->validate_nodes_and_infer_types();
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName,
                                           size_t outputIndex,
                                           ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "CNNNetworkNGraphImpl::addOutput");

    try {
        for (const auto& layer : _ngraph_function->get_ops()) {
            // Result can have the same name as previous operation
            if (layer->get_friendly_name() == layerName && !std::dynamic_pointer_cast<ngraph::op::Result>(layer)) {
                // Check that output port exists
                if (layer->outputs().size() <= outputIndex) {
                    return DescriptionBuffer(OUT_OF_BOUNDS, resp)
                           << "port index " << outputIndex << " exceeds the number of layer outputs "
                           << layer->outputs().size();
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

void CNNNetworkNGraphImpl::addOutput(const ::ngraph::Output<::ngraph::Node>& output) {
    auto dataName = ov::op::util::create_ie_output_name(output);
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

size_t CNNNetworkNGraphImpl::getBatchSize() const {
    // TODO Provide adequate implementation.
    // The original code from CNNNetworkImpl just gets the first input and returns the first dimension.
    // This is not correct in general. We can follow the same semantics, but order of inputs should be
    // guaranteed to be the same.
    auto params = _ngraph_function->get_parameters();
    sort(params.begin(), params.end(), [](std::shared_ptr<ngraph::Node> lhs, std::shared_ptr<ngraph::Node> rhs) {
        return lhs->get_friendly_name() < rhs->get_friendly_name();
    });

    for (const auto& param : params) {
        if (param->get_output_partial_shape(0).rank().is_dynamic())
            continue;
        auto pshape = param->get_output_partial_shape(0);
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

StatusCode CNNNetworkNGraphImpl::reshape(const std::map<std::string, ngraph::PartialShape>& inputShapes,
                                         ResponseDesc* responseDesc) noexcept {
    try {
        if (inputShapes.empty())
            return OK;

        const auto& params = _ngraph_function->get_parameters();

        // Check that we need to do reshape only if input shapes will be changed
        bool needReshape = false;
        for (const auto& param : params) {
            const auto it = inputShapes.find(param->get_friendly_name());
            if (it == inputShapes.end()) {
                continue;
            }
            if (param->get_output_partial_shape(0).is_dynamic() || param->get_output_partial_shape(0) != it->second) {
                needReshape = true;
                break;
            }
        }

        if (!needReshape)
            return OK;

        // save original parameters shape
        std::map<std::string, ngraph::PartialShape> originalInputShapes;
        for (const auto& param : params) {
            originalInputShapes[param->get_friendly_name()] = param->get_output_partial_shape(0);
        }

        try {
            ngraph::pass::Manager ssr_manager;
            using namespace ov::pass;
            REGISTER_PASS(ssr_manager, SmartReshape)
            ssr_manager.run_passes(_ngraph_function);

            reshape(inputShapes);
        } catch (std::exception& ex) {
            reshape(originalInputShapes);
            return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
        }

        return OK;
    } catch (const InferenceEngine::GeneralError& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    } catch (const ov::Exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    } catch (const std::runtime_error& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    } catch (const std::out_of_range& ex) {
        return DescriptionBuffer(OUT_OF_BOUNDS, responseDesc) << ex.what();
    } catch (...) {
        return GENERAL_ERROR;
    }
}

StatusCode CNNNetworkNGraphImpl::reshape(const std::map<std::string, SizeVector>& inputShapes,
                                         ResponseDesc* responseDesc) noexcept {
    std::map<std::string, ngraph::PartialShape> shapes;
    for (const auto& shape : inputShapes)
        shapes[shape.first] = ngraph::PartialShape(shape.second);
    return reshape(shapes, responseDesc);
}

#if 0
namespace {
void collect_dynamism_signature(const std::shared_ptr<ov::Model>& ov_model,
                                std::map<std::string, std::map<std::string, size_t>>& signatures,
                                bool obfuscate) {
    for (const auto& op : ov_model->get_ordered_ops()) {
        const auto& type_name = string(op->get_type_info().name) + "_" + op->get_type_info().version_id;

        std::stringstream shape_representation;
        for (const auto& input : op->input_values()) {
            bool first = true;
            shape_representation << "{";
            for (const auto& dimension : input.get_partial_shape()) {
                if (!first)
                    shape_representation << ",";
                first = false;

                if (obfuscate)
                    shape_representation << (dimension.is_dynamic() ? "D" : "S");
                else
                    shape_representation << dimension;
            }
            shape_representation << "} ";
        }
        shape_representation << "-> ";
        for (const auto& output : op->outputs()) {
            bool first = true;
            shape_representation << "{";
            for (const auto& dimension : output.get_partial_shape()) {
                if (!first)
                    shape_representation << ",";
                first = false;

                if (obfuscate)
                    shape_representation << (dimension.is_dynamic() ? "D" : "S");
                else
                    shape_representation << dimension;
            }
            shape_representation << "} ";
        }
        signatures[type_name][shape_representation.str()]++;

        // collect dynamism signature for sub-graphs of multi-subgraph operation
        if (const auto multi_sub_graph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            int num_subgraphs = static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
            for (int i = 0; i < num_subgraphs; i++)
                collect_dynamism_signature(multi_sub_graph_op->get_function(i), signatures, obfuscate);
        }
    }
}
}  // namespace
#endif

void CNNNetworkNGraphImpl::reshape(const std::map<std::string, ngraph::PartialShape>& inputShapes) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "CNNNetworkNGraphImpl::reshape");

    auto params = _ngraph_function->get_parameters();

    bool parameter_replaced = false;
    for (auto& param : params) {
        if (inputShapes.find(param->get_friendly_name()) == inputShapes.end())
            continue;
        param->set_partial_shape(inputShapes.at(param->get_friendly_name()));
        parameter_replaced = true;
    }
    if (parameter_replaced)
        _ngraph_function->validate_nodes_and_infer_types();

#if 0
        bool obfuscate = true;  // set to false to get exact dimensions
        std::map<std::string, std::map<std::string, size_t>> signatures;

        collect_dynamism_signature(_ngraph_function, signatures, obfuscate);

        for (const auto& item : signatures)
            for (const auto& shape_to_count : item.second)
                std::cout << item.first << " " << shape_to_count.second << "x " << shape_to_count.first << std::endl;
#endif

    std::unordered_set<std::string> opName;
    for (const auto& result : _ngraph_function->get_results()) {
        addOutput(result->input_value(0));
    }

    for (const auto& parameter : _ngraph_function->get_parameters()) {
        const auto& outName = parameter->get_friendly_name();
        if (opName.find(outName) != opName.end()) {
            IE_THROW() << "All operations in nGraph function should have unique friendly names!";
        }
        opName.insert(outName);
        createDataForResult(parameter, outName, _data[outName]);
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
        using namespace ov::pass;
        REGISTER_PASS(manager, Serialize, xmlPath, binPath, custom_opsets, ov::pass::Serialize::Version::IR_V10)
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

StatusCode CNNNetworkNGraphImpl::serialize(std::ostream& xmlBuf, std::ostream& binBuf, ResponseDesc* resp) const
    noexcept {
    try {
        std::map<std::string, ngraph::OpSet> custom_opsets;
        for (const auto& extension : _ie_extensions) {
            auto opset = extension->getOpSets();
            custom_opsets.insert(begin(opset), end(opset));
        }
        ngraph::pass::Manager manager;
        using namespace ov::pass;
        REGISTER_PASS(manager, Serialize, xmlBuf, binBuf, custom_opsets, ov::pass::Serialize::Version::IR_V10)
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

StatusCode CNNNetworkNGraphImpl::serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob, ResponseDesc* resp) const
    noexcept {
    try {
        std::map<std::string, ngraph::OpSet> custom_opsets;
        for (const auto& extension : _ie_extensions) {
            auto opset = extension->getOpSets();
            custom_opsets.insert(begin(opset), end(opset));
        }

        std::stringstream binBuf;
        ngraph::pass::Manager manager;
        using namespace ov::pass;
        REGISTER_PASS(manager, Serialize, xmlBuf, binBuf, custom_opsets, ov::pass::Serialize::Version::IR_V10)
        manager.run_passes(_ngraph_function);

        std::streambuf* pbuf = binBuf.rdbuf();
        unsigned long bufSize = static_cast<unsigned long>(binBuf.tellp());

        TensorDesc tensorDesc(Precision::U8, {bufSize}, Layout::C);
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

StatusCode CNNNetworkNGraphImpl::getOVNameForTensor(std::string& ov_name,
                                                    const std::string& orig_name,
                                                    ResponseDesc* resp) const noexcept {
    if (_tensorNames.find(orig_name) == _tensorNames.end())
        return DescriptionBuffer(NOT_FOUND, resp)
               << "Framework tensor with name \"" << orig_name << "\" was not mapped to OpenVINO data!";
    ov_name = _tensorNames.at(orig_name);
    return OK;
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        if (getBatchSize() == size)
            return OK;
        auto original_parameters = _ngraph_function->get_parameters();
        if (original_parameters.empty())
            return DescriptionBuffer(GENERAL_ERROR, responseDesc)
                   << "Cannot set batch! Function doesn't contain parameters!";

        stringstream ss;
        ss << " Please use reshape method instead. Original parameter shapes are: ";
        for (size_t i = 0; i < original_parameters.size(); ++i) {
            if (i)
                ss << ", ";
            ss << "\"" << original_parameters[i]->get_friendly_name()
               << "\": " << original_parameters[i]->get_output_partial_shape(0);
        }

        // ill-formed logic from the past setBatchSize (we keep it for backward-compatibility)
        const auto first_parameter =
            *std::min_element(original_parameters.begin(),
                              original_parameters.end(),
                              [](std::shared_ptr<ngraph::Node> lhs, std::shared_ptr<ngraph::Node> rhs) {
                                  return lhs->get_friendly_name() < rhs->get_friendly_name();
                              });
        const auto first_parameter_pshape = first_parameter->get_output_partial_shape(0);
        if (first_parameter_pshape.is_dynamic())
            return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc)
                   << "Cannot set batch! Function contains parameter with partially defined shape!" << ss.str();
        const auto first_parameter_rank = first_parameter_pshape.rank().get_length();
        if (first_parameter_rank == 0 || first_parameter_rank == 1 || first_parameter_rank == 3)
            return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc)
                   << "Cannot set batch! Function contains 0D/1D/3D parameter with unknown batch dimension placement."
                   << ss.str();

        std::map<std::string, std::vector<size_t>> inShapes;
        for (const auto& parameter : original_parameters) {
            const auto& pshape = parameter->get_output_partial_shape(0);
            if (pshape.is_dynamic())
                return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc)
                       << "Cannot set batch! Function contains parameter with partially defined shape!" << ss.str();
            const auto& rank = pshape.rank().get_length();
            if (rank == 0)
                return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc)
                       << "Cannot set batch! Function contains 0D/1D/3D parameter with unknown batch dimension "
                          "placement."
                       << ss.str();
            auto shape = parameter->get_shape();
            shape[0] = {static_cast<size_t>(
                std::ceil(size * static_cast<float>(shape[0]) / static_cast<float>(getBatchSize())))};
            inShapes[parameter->get_friendly_name()] = shape;
        }
        ov::pass::Manager ssr_manager;
        using namespace ov::pass;
        REGISTER_PASS(ssr_manager, SetBatchSize)
        ssr_manager.run_passes(_ngraph_function);

        return reshape(inShapes, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}
