// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "ngraph/ops.hpp"
#include "ngraph_ops/nms_ie_internal.hpp"
#include "openvino/core/graph_util.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace runtime {
namespace intel_gpu {

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

auto getParamName = [](const std::shared_ptr<ov::Node>& param) -> std::string {
    const auto& names = param->get_output_tensor(0).get_names();
    if (!names.empty())
        return *names.begin();
    else
        return param->get_friendly_name();
};

//  detect the only supported dynamic shape case -
//  exactly one dimension is dynamic in input params with defined min/max interval
bool Program::IsDynBatchModel(const std::shared_ptr<ov::Model>& model,
                              std::map<std::string, ov::PartialShape>& shapes,
                              std::map<std::string, std::pair<int64_t, int64_t>>& batch_dim) {
    for (const auto& param : model->get_parameters()) {
        auto pname = getParamName(param);
        batch_dim[pname] = { -1, -1 };
        if (param->get_output_partial_shape(0).rank().is_dynamic()) {
            return false;
        }
        ov::PartialShape pshape = param->get_output_partial_shape(0);
        int dynCount = 0;
        int64_t batch_idx = -1;
        for (size_t i = 0; i < pshape.size(); i++) {
            if (pshape[i].is_dynamic()) {
                dynCount++;
                if (batch_idx < 0) {
                    batch_idx = i;
                }
            }
        }
        switch (dynCount) {
            case 1:
                // exactly one dynamic dim
                {
                    int64_t max_b = pshape[batch_idx].get_max_length();
                    if (max_b > 1) {
                        batch_dim[pname].first = batch_idx;
                        batch_dim[pname].second = max_b;
                        pshape[batch_idx] = 1;
                    }
                }
            case 0:
                // no dynamic dims - possible legacy case
                shapes[pname] = pshape;
                break;
            default:
                break;
        }
    }
    if (batch_dim.empty())
        return false;
    bool dyn_shape_batch_found = false;
    // detect 1st dyn dim, mark it and continue
    auto bitr = batch_dim.begin();
    dyn_shape_batch_found = bitr->second.first >= 0;
    auto batch_val_1st = bitr->second.second;
    bitr++;
    for (; bitr != batch_dim.end(); bitr++) {
        if (bitr->second.first >= 0) {
            if (bitr->second.second != batch_val_1st) {
                dyn_shape_batch_found = false;
                break;
            } else {
                dyn_shape_batch_found = true;
            }
        }
    }
    return dyn_shape_batch_found;
}

Program::Program(InferenceEngine::CNNNetwork& network, std::shared_ptr<cldnn::engine> engine, const Config& config,
    bool createTopologyOnly, bool partialBuild)
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

    bool dyn_shape_batch_found = false;
    std::map<std::string, ngraph::PartialShape> shapes;
    std::map<std::string, std::pair<int64_t, int64_t>> batch_dim;
    if (m_config.enableDynamicBatch) {
        // in case of legacy dynamic batch,
        // we assume 4D input with 0 batch dim
        auto param = func->get_parameters().front();
        auto pname = getParamName(param);
        shapes[pname] = param->get_output_partial_shape(0);
        batch_dim[pname].first = 0;
        batch_dim[pname].second = m_config.max_dynamic_batch;
    } else {
        dyn_shape_batch_found = IsDynBatchModel(func, shapes, batch_dim);
        if (dyn_shape_batch_found) {
            m_config.max_dynamic_batch = batch_dim.begin()->second.second;
        } else {
            if (!batch_dim.empty() && shapes.empty()) {
                // more than on dynamic dim or dynamic rank
                IE_THROW() << "Only dynamic batch is supported!";
            }
        }
    }

    int m_bv_sz = GetMaxBatchSizeForSingleProgram();
    m_max_batch = m_config.max_dynamic_batch;

    if (dyn_shape_batch_found || config.max_dynamic_batch > 1) {
        // compile log2 networks to serve dynamic batch requests
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            inputLayouts.clear();
            outputDims.clear();
            primitiveIDs.clear();
            blobMemCache.clear();

            auto new_batch = 1U << static_cast<unsigned>(b);
            ChangeInputBatch(new_batch);

            // clone the source model, find the batch dim
            // and reshape the model to next batch size
            auto new_func = ov::clone_model(*func);
            std::map<ov::Output<ov::Node>, ngraph::PartialShape> new_shapes;
            for (const auto& param : new_func->get_parameters()) {
                ov::PartialShape pshape = param->get_output_partial_shape(0);

                auto pname = getParamName(param);
                auto batch_idx = batch_dim[pname].first;

                if (batch_idx >= 0) {
                    auto pshape = shapes[pname];
                    pshape[batch_idx] = new_batch;
                    new_shapes[param->output(0)] = pshape;
                }
            }
            new_func->reshape(new_shapes);
            {
                auto deviceInfo = engine->get_device_info();
                TransformationsPipeline transformations(config, deviceInfo);
                transformations.apply(new_func);
            }

            // reshape network input/output maps accordingly
            // for correct network compilation
            for (auto& new_input : new_func->inputs()) {
                auto iname = new_input.get_node()->get_friendly_name();
                auto it = networkInputs.find(iname);
                if (it != networkInputs.end()) {
                    auto shape = new_input.get_shape();
                    auto l = it->second->getTensorDesc().getLayout();
                    it->second->getInputData()->reshape(shape, l);
                }
            }

            for (auto& new_output : new_func->outputs()) {
                auto iname = new_output.get_node_shared_ptr()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
                auto it = networkOutputs.find(iname);
                if (it != networkOutputs.end()) {
                    auto shape = new_output.get_shape();
                    auto l = it->second->getTensorDesc().getLayout();
                    it->second->reshape(shape, l);
                }
            }
            m_programs.insert(m_programs.begin(), BuildProgram(new_func->get_ordered_ops(), networkInputs, networkOutputs,
                createTopologyOnly, partialBuild));
        }
        {
            // recompute maximal dynamic batch inputs/outputs for infer request
            // and store them into internal maps
            // same operations as above, but for maximum batch
            auto new_func = ov::clone_model(*func);
            std::map<ov::Output<ov::Node>, ngraph::PartialShape> new_shapes;
            for (const auto& param : new_func->get_parameters()) {
                ov::PartialShape pshape = param->get_output_partial_shape(0);

                auto pname = getParamName(param);
                auto batch_idx = batch_dim[pname].first;

                if (batch_idx >= 0) {
                    auto pshape = shapes[pname];
                    pshape[batch_idx] = m_max_batch;
                    new_shapes[param->output(0)] = pshape;
                }
            }
            new_func->reshape(new_shapes);

            for (auto& new_input : new_func->inputs()) {
                auto iname = new_input.get_node()->get_friendly_name();
                auto it = networkInputs.find(iname);
                if (it != networkInputs.end()) {
                    auto shape = new_input.get_shape();
                    auto l = it->second->getTensorDesc().getLayout();
                    it->second->getInputData()->reshape(shape, l);
                }
            }

            for (auto& new_output : new_func->outputs()) {
                auto iname = new_output.get_node_shared_ptr()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
                auto it = networkOutputs.find(iname);
                if (it != networkOutputs.end()) {
                    auto shape = new_output.get_shape();
                    auto l = it->second->getTensorDesc().getLayout();
                    SizeVector old_shape = it->second->getTensorDesc().getDims();
                    it->second->reshape(shape, l);
                    // detect changed output batch dimension
                    SizeVector new_shape = it->second->getTensorDesc().getDims();
                    for (int64_t i = 0; i < old_shape.size(); i++) {
                        if (old_shape[i] != new_shape[i]) {
                            m_output_batch_dim[iname] = i;
                            break;
                        }
                    }
                }
            }
            m_networkInputs = networkInputs;
            m_networkOutputs = networkOutputs;
            m_input_batch_dim = batch_dim;
        }
    } else {
        m_programs.emplace_back(BuildProgram(ops, networkInputs, networkOutputs, createTopologyOnly, partialBuild));
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
                                                      bool createTopologyOnly, bool partialBuild) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Program::BuildProgram");
    cldnn::build_options options;

    if (!m_config.graph_dumps_dir.empty()) {
        options.set_option(cldnn::build_option::graph_dumps_dir(m_config.graph_dumps_dir));
    }

    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::tuning_config(m_config.tuningConfig));
    if (partialBuild) {
        options.set_option(cldnn::build_option::partial_build_program(true));
    }
    PrepareBuild(networkInputs, networkOutputs);
    for (const auto& op : ops) {
        CreateSingleLayerPrimitive(*m_topology, op);
    }
    if (createTopologyOnly) {
        return {};
    } else {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Program::CreateProgram");
        cldnn::program::ptr program;
        try {
            program = cldnn::program::build_program(*m_engine, *m_topology, options);
        } catch (std::exception& e) {
            IE_THROW() << "cldnn program build failed! " << e.what();
        }
        CleanupBuild();

        return program;
    }
}

bool Program::IsOpSupported(const InferenceEngine::CNNNetwork& network, const std::shared_ptr<ngraph::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Program::IsOpSupported");
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
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Program::CreateSingleLayerPrimitive");
    InitProfileInfo(op->get_friendly_name(), op->get_type_name());

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "Process " << "op::v" << op->get_type_info().version << "::" << op->get_type_name() << " operation "
                       << "(friendly_name=" << op->get_friendly_name() << ")" << std::endl;
    }

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
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddPrimitiveToProfiler(cldnn::primitive_id id, const std::shared_ptr<ngraph::Node>& op,
                                     cldnn::primitive_id customOutputId) {
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                          const std::shared_ptr<ngraph::Node>& op) {
    InitProfileInfo(id, layer_type_lower(op), false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, parentId);
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

void Program::AddVariableStateInfo(const std::string& variable_id, const cldnn::layout& layout) {
    auto it = m_variablesStateInfo.find(variable_id);
    if (it != m_variablesStateInfo.end())
        it->second.insert(layout);
    else
        m_variablesStateInfo.insert({variable_id, { layout }});
}

// TODO: Does it make sense to add such method to ngraph core?
bool IsNodeOnConstPath(const std::shared_ptr<ngraph::Node>& node) {
    std::set<std::shared_ptr<ngraph::Node>> nodes_processed = {};
    std::function<bool(const std::shared_ptr<ngraph::Node>&)> is_const_node = [&nodes_processed, &is_const_node](const std::shared_ptr<ngraph::Node>& node) {
        if (nodes_processed.count(node)) return true;
        nodes_processed.insert(node);
        // If input is constant, then drop it from the processing list
        if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node) != nullptr)
            return true;
        // If the node doesn't have any parents and it's not a constant, then we deal with dynamic path
        if (node->get_input_size() == 0)
            return false;
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto input_node = node->get_input_node_shared_ptr(i);
            if (!is_const_node(input_node))
                return false;
        }
        return true;
    };
    return is_const_node(node);
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
