// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/graph_util.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#ifdef __linux__
# include <dlfcn.h>
#endif

#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif


using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

const cldnn::primitive_id ProgramBuilder::m_preProcessTag("_cldnn_input_preprocess");
const cldnn::primitive_id ProgramBuilder::m_meanValuesTag("_cldnn_mean_values");
const cldnn::primitive_id ProgramBuilder::m_preCustomLayerTag("_cldnn_custom_preprocess");
const cldnn::primitive_id ProgramBuilder::m_postCustomLayerTag("_cldnn_custom_postprocess");
ProgramBuilder::factories_map_t ProgramBuilder::factories_map = {};
std::mutex ProgramBuilder::m_mutex = {};

std::string layer_type_lower(const ov::Node* op) {
    std::string layerType = op->get_type_name();
    std::transform(layerType.begin(), layerType.end(), layerType.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return layerType;
}

std::string layer_type_name_ID(const ov::Node* op) {
    return layer_type_lower(op) + ":" + op->get_friendly_name();
}

std::string layer_type_lower(const std::shared_ptr<ov::Node>& op) {
    return layer_type_lower(op.get());
}

std::string layer_type_name_ID(const std::shared_ptr<ov::Node>& op) {
    return layer_type_name_ID(op.get());
}

void ProgramBuilder::ChangeInputBatch(int batch) {
    m_curBatch = batch;
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
bool ProgramBuilder::IsDynBatchModel(const std::shared_ptr<ov::Model>& model,
                              std::map<std::string, ov::PartialShape>& shapes,
                              std::map<std::string, std::pair<int64_t, int64_t>>& batch_dim) {
    for (const auto& param : model->get_parameters()) {
        auto pname = getParamName(param);
        batch_dim[pname] = { -1, -1 };
        if (param->get_output_partial_shape(0).rank().is_dynamic()) {
            return false;
        }
        ov::PartialShape pshape = param->get_output_partial_shape(0);
        bool only_batch_dynamic = pshape.size() && pshape[0].is_dynamic();
        for (size_t i = 1; i < pshape.size(); i++) {
            if (pshape[i].is_dynamic()) {
                // only support 0th dimension for legacy dynamic batch
                return false;
            }
        }
        if (only_batch_dynamic) {
            int64_t max_b = pshape[0].get_max_length();
            if (max_b > 1) {
                batch_dim[pname].first = 0;
                batch_dim[pname].second = max_b;
                pshape[0] = 1;
            } else {
                // unbounded dynamic shape should be handled with new dynamic shape path
                return false;
            }
        }
        shapes[pname] = pshape;
    }
    if (batch_dim.empty())
        return false;

    bool dyn_shape_batch_found = false;
    // detect 1st dyn dim, mark it and continue
    auto bitr = batch_dim.begin();
    dyn_shape_batch_found = (bitr->second.first == 0);
    auto batch_val_1st = bitr->second.second;
    bitr++;
    for (; bitr != batch_dim.end(); bitr++) {
        if (bitr->second.first == 0) {
            if (bitr->second.second != batch_val_1st) {
                dyn_shape_batch_found = false;
                break;
            } else {
                dyn_shape_batch_found = true;
            }
        } else {
            return false;
        }
    }
    return dyn_shape_batch_found;
}

ProgramBuilder::ProgramBuilder(InferenceEngine::CNNNetwork& network, cldnn::engine& engine, const ExecutionConfig& config,
    bool createTopologyOnly, bool partialBuild,
    InferenceEngine::InputsDataMap* inputs, InferenceEngine::OutputsDataMap* outputs,
    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor, bool innerProgram)
    : m_curBatch(-1)
    , m_config(config)
    , m_engine(engine)
    , queryMode(false)
    , m_task_executor(task_executor) {
    if (m_task_executor == nullptr)
        m_task_executor = cldnn::program::make_task_executor(m_config);
    // Extract inputs/outputs info from CNNNetwork
    auto networkInputs = (inputs != nullptr) ? *inputs : network.getInputsInfo();
    auto networkOutputs = (outputs != nullptr) ? *outputs : network.getOutputsInfo();

    auto func = network.getFunction();
    if (!func) {
        OPENVINO_THROW("Function pointer inside CNNNetwork is nullptr");
    }

    // locate global custom kernel config
    // and auto-load kernels from it
#ifdef _WIN32
    CHAR mpath[MAX_PATH + 1];
    HMODULE nModule;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)CustomLayer::LoadFromFile,
        &nModule);
    GetModuleFileName(nModule, mpath, sizeof(mpath));
#elif __linux__
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(CustomLayer::LoadFromFile), &dl_info);
    const char* mpath = dl_info.dli_fname;
#else
#error "Intel GPU plugin: unknown target system"
#endif
    std::string configFile(mpath);
    std::size_t dir_split_pos = configFile.find_last_of("/\\");
    std::string config_path;

    if (dir_split_pos != std::string::npos) {
        // path contains directory
        config_path = configFile.substr(0, dir_split_pos);
    }
    config_path += "/cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml";

    CustomLayer::LoadFromFile(config_path, m_custom_layers, true);
    auto custom_layers_config = m_config.get_property(ov::intel_gpu::config_file);
    CustomLayer::LoadFromFile(custom_layers_config, m_custom_layers, custom_layers_config.empty());

    auto ops = func->get_ordered_ops();

    bool dyn_shape_batch_found = false;
    std::map<std::string, ov::PartialShape> shapes;
    std::map<std::string, std::pair<int64_t, int64_t>> batch_dim;

    dyn_shape_batch_found = IsDynBatchModel(func, shapes, batch_dim);
    if (dyn_shape_batch_found) {
        m_config.set_property(ov::intel_gpu::max_dynamic_batch(batch_dim.begin()->second.second));
    }

    int m_bv_sz = GetMaxBatchSizeForSingleProgram();
    m_max_batch = static_cast<int>(m_config.get_property(ov::intel_gpu::max_dynamic_batch));

    // Do not apply dynamic batch for inner program (only single batch is allowed)
    if (!innerProgram && (dyn_shape_batch_found || m_max_batch > 1)) {
        // compile log2 networks to serve dynamic batch requests
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            inputLayouts.clear();
            outputDims.clear();
            primitive_ids.clear();
            blobMemCache.clear();

            auto new_batch = 1U << static_cast<unsigned>(b);
            ChangeInputBatch(new_batch);

            // clone the source model, find the batch dim
            // and reshape the model to next batch size
            auto new_func = func->clone();
            std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
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
                auto deviceInfo = engine.get_device_info();
                TransformationsPipeline transformations(m_config, deviceInfo);
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
            auto new_func = func->clone();
            std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
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
                    for (int64_t i = 0; i < static_cast<int64_t>(old_shape.size()); i++) {
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
        m_programs.emplace_back(BuildProgram(ops, networkInputs, networkOutputs, createTopologyOnly, partialBuild, innerProgram));
    }
}

ProgramBuilder::ProgramBuilder(cldnn::engine& engine, const ExecutionConfig& config,
                 InferenceEngine::InputsDataMap* inputs, InferenceEngine::OutputsDataMap* outputs)
        : m_max_batch(1)
        , m_curBatch(-1)
        , m_config(config)
        , m_engine(engine)
        , queryMode(false) {
    m_task_executor = cldnn::program::make_task_executor(m_config);
    if (inputs != nullptr)
        m_networkInputs = *inputs;
    if (outputs != nullptr)
        m_networkOutputs = *outputs;
}

int ProgramBuilder::GetMaxBatchSizeForSingleProgram() {
    auto max_dynamic_batch = m_config.get_property(ov::intel_gpu::max_dynamic_batch);
    if (max_dynamic_batch > 1) {
        // calculate number of networks necessary based on binary log
        unsigned int tmp = static_cast<unsigned int>(max_dynamic_batch);
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

std::shared_ptr<cldnn::program> ProgramBuilder::GetCompiledProgram(int program_id) {
    if (program_id >= static_cast<int32_t>(m_programs.size()))
        OPENVINO_THROW("Invalid program ID");

    return m_programs[program_id];
}

void ProgramBuilder::PrepareBuild(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    m_topology.reset(new cldnn::topology());
    m_networkInputs = networkInputs;
    m_networkOutputs = networkOutputs;
}

void ProgramBuilder::CleanupBuild() {
    m_topology.reset();
    m_networkInputs.clear();
    m_networkOutputs.clear();
    #if defined(__unix__) && !defined(__ANDROID__)
    //  NOTE: In linux, without malloc_trim, an amount of the memory used by compilation is not being returned to system thought they are freed.
    //  (It is at least 500 MB when we perform parallel compilation)
    //  It is observed that freeing the memory manually with malloc_trim saves significant amount of the memory.
    //  Also, this is not happening in Windows.
    //  So, added malloc_trim for linux build until we figure out a better solution.
    malloc_trim(0);
    #endif
}

std::shared_ptr<cldnn::program> ProgramBuilder::BuildProgram(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                                      InferenceEngine::InputsDataMap networkInputs,
                                                      InferenceEngine::OutputsDataMap networkOutputs,
                                                      bool createTopologyOnly, bool partialBuild, bool innerProgram) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::BuildProgram");
    // std::cout << "BuildProgram " << createTopologyOnly << ", " << partialBuild << ", " << innerProgram << std::endl;
    // In the case of inner program, allow_new_shape_infer flag is setted by outside of program.
    // So, do not check allow_new_shape_infer for inner program build
    for (const auto& op : ops) {
        if (requires_new_shape_infer(*op)) {
            allow_new_shape_infer = true;
            break;
        }
    }

    if (innerProgram) {
        allow_new_shape_infer = (m_config.get_property(ov::intel_gpu::allow_new_shape_infer) || allow_new_shape_infer);
    }

    m_config.set_property(ov::intel_gpu::partial_build_program(partialBuild));
    m_config.set_property(ov::intel_gpu::optimize_data(true));
    m_config.set_property(ov::intel_gpu::allow_new_shape_infer(allow_new_shape_infer));

    PrepareBuild(networkInputs, networkOutputs);
    {
        GPU_DEBUG_DEFINE_MEM_LOGGER("CreateSingleLayerPrimitives");
        for (const auto& op : ops) {
            CreateSingleLayerPrimitive(*m_topology, op);
        }
    }
    if (createTopologyOnly) {
        return {};
    } else {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::CreateProgram");
        cldnn::program::ptr program;
        try {
            program = cldnn::program::build_program(m_engine, *m_topology, m_config, get_task_executor());
        } catch (std::exception& e) {
            OPENVINO_ASSERT(false, "GPU program build failed!\n", e.what());
        }
        CleanupBuild();

        return program;
    }
}

bool ProgramBuilder::IsOpSupported(const InferenceEngine::CNNNetwork& network, const std::shared_ptr<ov::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::IsOpSupported");
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
        allow_new_shape_infer = requires_new_shape_infer(*op);
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

void ProgramBuilder::CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ov::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::CreateSingleLayerPrimitive");
    GPU_DEBUG_LOG << "Process " << "op::v" << op->get_type_info().version_id << "::" << op->get_type_name() << " operation "
                  << "(friendly_name=" << op->get_friendly_name() << ")" << std::endl;

    bool is_created = false;
    const ov::NodeTypeInfo* op_type_info = &op->get_type_info();
    while (op_type_info != nullptr) {
        auto customLayer = m_custom_layers.find(op->get_type_name());
        if (customLayer != m_custom_layers.end()) {
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
        OPENVINO_THROW("Operation: ", op->get_friendly_name(),
                       " of type ", op->get_type_name(),
                       "(", op->get_type_info().version_id, ") is not supported");
    }
}

std::vector<cldnn::input_info> ProgramBuilder::GetInputInfo(const std::shared_ptr<ov::Node>& op) const {
    if (!op) {
        return {};
    }

    // Currently multiple outputs are supported only in the dynamic shape case,
    // So the output index of the dependency is not processed
    std::vector<cldnn::input_info> inputInfo;
    for (size_t i = 0; i < op->get_input_size(); i++) {
        auto prevOp = op->get_input_node_ptr(i);
        std::string prevName = layer_type_name_ID(prevOp);
        bool is_legacy_multiple_outputs = !allow_new_shape_infer
                                          // Note:: Currently Split/Variadic Split are divided to multiple crops
                                          || ov::is_type<ov::op::v1::Split>(prevOp)
                                          || ov::is_type<ov::op::v1::VariadicSplit>(prevOp);
        if (prevOp->get_output_size() > 1 && is_legacy_multiple_outputs) {
            prevName += ".out" + std::to_string(op->get_input_source_output(i).get_index());
        }

        if (!queryMode) {
            if (primitive_ids.find(prevName) == primitive_ids.end()) {
                OPENVINO_THROW("Input ", prevName, " hasn't been found in primitive_ids map");
            }
            inputInfo.push_back(
                cldnn::input_info(primitive_ids.at(prevName), is_legacy_multiple_outputs ? 0: static_cast<int>(op->get_input_source_output(i).get_index())));
        } else {
            inputInfo.push_back(cldnn::input_info(prevName, is_legacy_multiple_outputs ? 0 : static_cast<int>(op->get_input_source_output(i).get_index())));
        }
    }
    return inputInfo;
}

void ProgramBuilder::init_profile_info(const cldnn::primitive& prim) {
    perfMap[prim.id].first = prim.id;
    auto& perfEntry = perfMap[prim.id].second;
    perfEntry.layerType = prim.origin_op_type_name;
    perfEntry.status = InferenceEngine::InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
    perfEntry.isCPU = false;
    perfEntry.parentPrimitive = prim.origin_op_name;
}

void ProgramBuilder::AddVariableStateInfo(const std::string& variable_id, const cldnn::layout& layout) {
    auto it = m_variablesStateInfo.find(variable_id);
    if (it != m_variablesStateInfo.end())
        it->second.insert(layout);
    else
        m_variablesStateInfo.insert({variable_id, { layout }});
}

void ProgramBuilder::add_primitive(const ov::Node& op, std::shared_ptr<cldnn::primitive> prim, std::vector<std::string> aliases) {
    OPENVINO_ASSERT(m_topology != nullptr, "[GPU] Invalid ProgramBuilder builder state: topology is nullptr");

    prim->origin_op_name = op.get_friendly_name();
    prim->origin_op_type_name = op.get_type_name();

    bool should_profile = prim->type != cldnn::mutable_data::type_id() &&
                          prim->type != cldnn::data::type_id();

    auto prim_id = prim->id;
    auto id = layer_type_name_ID(&op);
    primitive_ids[id] = prim_id;

    bool multi_output_case = ends_with(prim_id, ".out0") && prim_id.length() > 5 && prim_id.substr(0, prim_id.length() - 5) == id;
    if (id != prim_id) {
        primitive_ids[prim_id] = prim_id;

        if (!multi_output_case)
            prim->origin_op_type_name = prim->type_string();
    }

    if (this->m_config.get_property(ov::enable_profiling) && should_profile) {
        profiling_ids.push_back(prim_id);
        init_profile_info(*prim);
    }

    for (auto& alias : aliases) {
        primitive_ids[alias] = prim_id;
    }

    m_topology->add_primitive(prim);
}

bool ProgramBuilder::requires_new_shape_infer(const ov::Node& op) const {
    if (op.is_dynamic()) {
        return true;
    }

    if (ov::is_type<op::FullyConnectedCompressed>(&op))
        return true;

    for (size_t i = 0; i < op.get_output_size(); i++) {
        if (op.get_output_partial_shape(i).size() > 6)
            return true;
    }

    for (size_t i = 0; i < op.get_input_size(); i++) {
        if (op.get_input_partial_shape(i).size() > 6)
            return true;
    }

    return false;
}

// TODO: Does it make sense to add such method to ov core?
bool IsNodeOnConstPath(const std::shared_ptr<ov::Node>& node) {
    std::set<std::shared_ptr<ov::Node>> nodes_processed = {};
    std::function<bool(const std::shared_ptr<ov::Node>&)> is_const_node = [&nodes_processed, &is_const_node](const std::shared_ptr<ov::Node>& node) {
        if (nodes_processed.count(node)) return true;
        nodes_processed.insert(node);
        // If input is constant, then drop it from the processing list
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(node) != nullptr)
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

void validate_inputs_count(const std::shared_ptr<ov::Node>& op, std::vector<size_t> valid_inputs_count) {
    for (auto ic : valid_inputs_count) {
        if (op->get_input_size() == ic) {
            return;
        }
    }

    OPENVINO_THROW("Invalid inputs count (", op->get_input_size(), ") in )",
                   op->get_friendly_name(), " (", op->get_type_name(),
                   " ", op->get_type_info().version_id, ")");
}

}  // namespace intel_gpu
}  // namespace ov
