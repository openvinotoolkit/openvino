// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "ngraph/ops.hpp"
#include "ngraph_ops/nms_ie_internal.hpp"
#include "cldnn_itt.h"
#include "cldnn/runtime/debug_configuration.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

const cldnn::primitive_id Program::m_preProcessTag("_cldnn_input_preprocess");
const cldnn::primitive_id Program::m_meanValuesTag("_cldnn_mean_values");
const cldnn::primitive_id Program::m_preCustomLayerTag("_cldnn_custom_preprocess");
const cldnn::primitive_id Program::m_postCustomLayerTag("_cldnn_custom_postprocess");
Program::factories_map_t Program::factories_map = {};

std::string layer_type_lower(const ngraph::Node* op) {
    std::string layerType = op->get_type_name();
    std::transform(layerType.begin(), layerType.end(), layerType.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return layerType;
}

std::string layer_type_name_ID(const ngraph::Node* op) {
    return layer_type_lower(op) + ":" + op->get_friendly_name();
}

std::string layer_type_lower(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_lower(op.get());
}

std::string layer_type_name_ID(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_name_ID(op.get());
}

void Program::ChangeInputBatch(int batch) {
    m_curBatch = batch;
}

void Program::ValidateInputs(const std::shared_ptr<ngraph::Node>& op, std::vector<size_t> validInputsCount) {
    for (auto ic : validInputsCount) {
        if (op->get_input_size() == ic) {
            return;
        }
    }

    IE_THROW() << "Invalid inputs count (" << op->get_input_size() << ") in "
                       << op->get_friendly_name() << " (" << op->get_type_name()
                       << " op::v" << op->get_type_info().version << ")";
}

bool Program::CanProcessDynBatch(std::vector<std::shared_ptr<ngraph::Node>> ops, InferenceEngine::InputsDataMap networkInputs) const {
    if (networkInputs.empty())
        return false;

    for (auto op : ops) {
        // TODO: do we have any other exception cases?
        if (std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(op)) {
            if (op->get_input_shape(0)[0] == op->get_output_shape(0)[0])
                continue;
        }

        // List of the operations which can lead to invalid dynamic batch processing
        if (std::dynamic_pointer_cast<ngraph::op::internal::NonMaxSuppressionIEInternal>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v5::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v4::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v3::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v1::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::PSROIPooling>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::ROIPooling>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::PriorBox>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::DetectionOutput>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v4::Proposal>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Proposal>(op)) {
            return false;
        }

        auto customLayer = m_config.customLayers.find(op->get_type_name());
        if (customLayer != m_config.customLayers.end()) {
            return false;
        }
    }

    return true;
}

Program::Program(InferenceEngine::CNNNetwork& network, std::shared_ptr<cldnn::engine> engine, const Config& config, bool createTopologyOnly)
    : m_config(config)
    , m_engine(engine)
    , m_curBatch(-1)
    , queryMode(false) {
    // Extract inputs/outputs info from CNNNetwork
    auto networkInputs = network.getInputsInfo();
    auto networkOutputs = network.getOutputsInfo();

    auto func = network.getFunction();
    if (!func) {
        IE_THROW() << "Function pointer inside CNNNetwork is nullptr";
    }

    auto ops = func->get_ordered_ops();

    if (m_config.max_dynamic_batch > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(ops, networkInputs)) {
            IE_THROW() << "Such topology cannot be compiled for dynamic batch!";
        }
    }

    int m_bv_sz = GetMaxBatchSizeForSingleProgram();

    m_max_batch = config.max_dynamic_batch;

    if (config.max_dynamic_batch > 1) {
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            inputLayouts.clear();
            outputDims.clear();
            primitiveIDs.clear();
            blobMemCache.clear();

            ChangeInputBatch(1U << static_cast<unsigned>(b));
            m_programs.insert(m_programs.begin(), BuildProgram(ops, networkInputs, networkOutputs, createTopologyOnly));
        }
    } else {
        m_programs.emplace_back(BuildProgram(ops, networkInputs, networkOutputs, createTopologyOnly));
    }
}

int Program::GetMaxBatchSizeForSingleProgram() {
    if (m_config.max_dynamic_batch > 1) {
        // calculate number of networks necessary based on binary log
        unsigned int tmp = m_config.max_dynamic_batch;
        unsigned int mask = 1U << 31;
        unsigned int ldigit = 31;

        while (!(tmp & mask)) {
            mask >>= 1;
            ldigit--;
        }

        return ldigit + 1;
    }

    return 0;
}

std::shared_ptr<cldnn::program> Program::GetCompiledProgram(int program_id) {
    if (program_id >= m_programs.size())
        IE_THROW() << "Invalid program ID";

    return m_programs[program_id];
}

void Program::PrepareBuild(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    m_topology.reset(new cldnn::topology());
    m_networkInputs = networkInputs;
    m_networkOutputs = networkOutputs;
}

void Program::CleanupBuild() {
    m_topology.reset();
    m_networkInputs.clear();
    m_networkOutputs.clear();
}

std::shared_ptr<cldnn::program> Program::BuildProgram(const std::vector<std::shared_ptr<ngraph::Node>>& ops,
                                                      InferenceEngine::InputsDataMap networkInputs,
                                                      InferenceEngine::OutputsDataMap networkOutputs,
                                                      bool createTopologyOnly) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Program::BuildProgram");
    cldnn::build_options options;
    GPU_DEBUG_GET_INSTANCE(debug_config);

    if (!m_config.graph_dumps_dir.empty()) {
        options.set_option(cldnn::build_option::graph_dumps_dir(m_config.graph_dumps_dir));
    }

    GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
        options.set_option(cldnn::build_option::graph_dumps_dir(debug_config->dump_graphs));
    }

    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::tuning_config(m_config.tuningConfig));

    PrepareBuild(networkInputs, networkOutputs);
    for (const auto& op : ops) {
        CreateSingleLayerPrimitive(*m_topology, op);
    }
    if (createTopologyOnly) {
        return {};
    } else {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Program::CreateProgram");
        auto program = std::make_shared<cldnn::program>(*m_engine, *m_topology, options);
        CleanupBuild();

        return program;
    }
}

bool Program::IsOpSupported(const InferenceEngine::CNNNetwork& network, const std::shared_ptr<ngraph::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Program::IsOpSupported");
    cldnn::topology topology;
    try {
        // Query mode disables checks that input primitives are created,
        // as IsOpSupported method is called for each operation separately
        // So we just ensure that inputs count is valid for given operation
        EnableQueryMode();
        // Creating topology object for each operation is supposed to be more time-consuming than
        // simple check by op type, but it has 2 big advantages:
        // 1. Code reuse. We don't need to have separate white-list of supported operations or
        //    add any ugly macro/templates to apply single function to multiple cases.
        // 2. We also check parameters of each operation, which means we have more
        //    reliable results of QueryNetwork call.
        PrepareBuild(network.getInputsInfo(), network.getOutputsInfo());
        CreateSingleLayerPrimitive(topology, op);
        CleanupBuild();
        DisableQueryMode();
    } catch (std::exception&) {
        // Exception means that an operation or some of it's parameters are not supported
        CleanupBuild();
        return false;
    }

    return true;
}

void Program::CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Program::CreateSingleLayerPrimitive");
    InitProfileInfo(op->get_friendly_name(), op->get_type_name());

    bool is_created = false;
    const ngraph::NodeTypeInfo* op_type_info = &op->get_type_info();
    while (op_type_info != nullptr) {
        auto customLayer = m_config.customLayers.find(op->get_type_name());
        if (customLayer != m_config.customLayers.end()) {
            CreateCustomOp(*this, op, customLayer->second);
            return;
        }

        auto factory_it = factories_map.find(*op_type_info);
        if (factory_it != factories_map.end()) {
            factory_it->second(*this, op);
            is_created = true;
            break;
        }
        op_type_info = op_type_info->parent;
    }

    if (!is_created) {
        IE_THROW() << "Operation: " << op->get_friendly_name()
                           << " of type " << op->get_type_name()
                           << "(op::v" << op->get_type_info().version << ") is not supported";
    }
}

std::vector<cldnn::primitive_id> Program::GetInputPrimitiveIDs(const std::shared_ptr<ngraph::Node>& op) const {
    if (!op) {
        return {};
    }

    std::vector<cldnn::primitive_id> inputPrimitives;
    for (size_t i = 0; i < op->get_input_size(); i++) {
        auto prevOp = op->get_input_node_ptr(i);
        std::string prevName = layer_type_name_ID(prevOp);
        if (prevOp->get_output_size() > 1) {
            prevName += "." + std::to_string(op->get_input_source_output(i).get_index());
        }

        if (!queryMode) {
            if (primitiveIDs.find(prevName) == primitiveIDs.end()) {
                IE_THROW() << "Input " << prevName << " hasn't been found in primitiveIDs map";
            }
            inputPrimitives.push_back(primitiveIDs.at(prevName));
        } else {
            inputPrimitives.push_back(prevName);
        }
    }
    return inputPrimitives;
}

void Program::AddPrimitiveToProfiler(const std::shared_ptr<ngraph::Node>& op,
                                     cldnn::primitive_id customOutputId) {
    auto id = layer_type_name_ID(op);
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddPrimitiveToProfiler(cldnn::primitive_id id, const std::shared_ptr<ngraph::Node>& op,
                                     cldnn::primitive_id customOutputId) {
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                          const std::shared_ptr<ngraph::Node>& op) {
    InitProfileInfo(id, layer_type_lower(op), false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, parentId);
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = id;
    profilingIDs.push_back(id);
}

void Program::InitProfileInfo(const std::string& layerName,
                              const std::string& layerType,
                              bool isCPU,
                              InferenceEngine::InferenceEngineProfileInfo::LayerStatus status, std::string parentId) {
    std::string layer_type_lower = layerType;
    for (auto& c : layer_type_lower)
        c = tolower(c);

    std::string name = layerName;
    if (name.find(layer_type_lower + ":") != std::string::npos) {
        name = layerName.substr(layerName.find(":") + 1, layerName.length());
    }

    perfMap[layer_type_lower + ":" + name].first = name;
    auto& perfEntry = perfMap[layer_type_lower + ":" + name].second;
    perfEntry.layerType = layerType;
    perfEntry.status = status;
    perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
    perfEntry.isCPU = isCPU;
    perfEntry.parentPrimitive = parentId;
}

// TODO: Does it make sense to add such method to ngraph core?
bool IsNodeOnConstPath(const std::shared_ptr<ngraph::Node>& node) {
    std::list<std::shared_ptr<ngraph::Node>> nodes_to_process = { node };
    while (!nodes_to_process.empty()) {
        auto current_node = nodes_to_process.front();
        nodes_to_process.pop_front();

        for (size_t i = 0; i < current_node->get_input_size(); i++) {
            auto input_node = current_node->get_input_node_shared_ptr(i);

            // If input is constant, then drop if from the processing list
            if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_node) != nullptr)
                continue;

            // If the node doesn't have any parents and it's not a constant, then we deal with dynamic path
            if (input_node->get_input_size() == 0) {
                return false;
            }

            nodes_to_process.insert(nodes_to_process.end(), input_node);
        }
    }

    return true;
}

}  // namespace CLDNNPlugin
