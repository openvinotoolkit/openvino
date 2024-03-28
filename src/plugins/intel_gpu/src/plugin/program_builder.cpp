// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/loop.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#ifdef __linux__
# include <dlfcn.h>
#endif

#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif


namespace ov {
namespace intel_gpu {

const cldnn::primitive_id ProgramBuilder::m_preProcessTag("_cldnn_input_preprocess");
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

ProgramBuilder::ProgramBuilder(std::shared_ptr<ov::Model> model, cldnn::engine& engine, const ExecutionConfig& config,
                               bool create_topology_only, bool partial_build,
                               std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                               std::shared_ptr<cldnn::ICompilationContext> compilation_context,
                               bool is_inner_program)
    : m_model(model)
    , m_config(config)
    , m_engine(engine)
    , queryMode(false)
    , m_task_executor(task_executor)
    , m_compilation_context(compilation_context)
    , m_is_inner_program(is_inner_program) {
    if (m_task_executor == nullptr)
        m_task_executor = cldnn::program::make_task_executor(m_config);

    if (m_compilation_context == nullptr) {
        m_compilation_context = cldnn::program::make_compilation_context(m_config);
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

    auto ops = model->get_ordered_ops();

    m_program = build(ops, create_topology_only, partial_build, is_inner_program);
}

ProgramBuilder::ProgramBuilder(cldnn::engine& engine, const ExecutionConfig& config)
        : m_config(config)
        , m_engine(engine)
        , queryMode(false) {
    m_task_executor = cldnn::program::make_task_executor(m_config);
}

std::shared_ptr<cldnn::program> ProgramBuilder::get_compiled_program() const {
    return m_program;
}

void ProgramBuilder::prepare_build() {
    m_topology.reset(new cldnn::topology());
}

void ProgramBuilder::cleanup_build() {
    m_topology.reset();
    #if defined(__unix__) && !defined(__ANDROID__)
    //  NOTE: In linux, without malloc_trim, an amount of the memory used by compilation is not being returned to system thought they are freed.
    //  (It is at least 500 MB when we perform parallel compilation)
    //  It is observed that freeing the memory manually with malloc_trim saves significant amount of the memory.
    //  Also, this is not happening in Windows.
    //  So, added malloc_trim for linux build until we figure out a better solution.
    malloc_trim(0);
    #endif
}

std::shared_ptr<cldnn::program> ProgramBuilder::build(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                               bool create_topology_only, bool partial_build, bool is_inner_program) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::build");
    // In the case of inner program, allow_new_shape_infer flag is setted by outside of program.
    // So, do not check allow_new_shape_infer for inner program build
    for (const auto& op : ops) {
        if (requires_new_shape_infer(op)) {
            allow_new_shape_infer = true;
            break;
        }
    }

    if (is_inner_program) {
        allow_new_shape_infer = (m_config.get_property(ov::intel_gpu::allow_new_shape_infer) || allow_new_shape_infer);
    }

    m_config.set_property(ov::intel_gpu::partial_build_program(partial_build));
    m_config.set_property(ov::intel_gpu::optimize_data(true));
    m_config.set_property(ov::intel_gpu::allow_new_shape_infer(allow_new_shape_infer));

    prepare_build();
    {
        GPU_DEBUG_DEFINE_MEM_LOGGER("CreateSingleLayerPrimitives");
        for (const auto& op : ops) {
            CreateSingleLayerPrimitive(*m_topology, op);
        }
    }
    if (create_topology_only) {
        return {};
    } else {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::CreateProgram");
        cldnn::program::ptr program;
        try {
            program = cldnn::program::build_program(m_engine,
                                                    *m_topology,
                                                    m_config,
                                                    get_task_executor(),
                                                    get_compilation_context(),
                                                    false,
                                                    false,
                                                    is_inner_program);
        } catch (std::exception& e) {
            OPENVINO_ASSERT(false, "[GPU] ProgramBuilder build failed!\n", e.what());
        }
        cleanup_build();

        return program;
    }
}

bool ProgramBuilder::is_op_supported(const std::shared_ptr<ov::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::is_op_supported");
    cldnn::topology topology;
    try {
        // Query mode disables checks that input primitives are created,
        // as is_op_supported method is called for each operation separately
        // So we just ensure that inputs count is valid for given operation
        EnableQueryMode();
        // Creating topology object for each operation is supposed to be more time-consuming than
        // simple check by op type, but it has 2 big advantages:
        // 1. Code reuse. We don't need to have separate white-list of supported operations or
        //    add any ugly macro/templates to apply single function to multiple cases.
        // 2. We also check parameters of each operation, which means we have more
        //    reliable results of QueryNetwork call.
        prepare_build();
        allow_new_shape_infer = requires_new_shape_infer(op);
        CreateSingleLayerPrimitive(topology, op);
        cleanup_build();
        DisableQueryMode();
    } catch (std::exception&) {
        // Exception means that an operation or some of it's parameters are not supported
        cleanup_build();
        return false;
    }

    return true;
}

void ProgramBuilder::CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ov::Node>& op) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "ProgramBuilder::CreateSingleLayerPrimitive");
    GPU_DEBUG_LOG << "Process " << "op::" << op->get_type_info().version_id << "::" << op->get_type_name() << " operation "
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

        const std::string paged_attention_type = "PagedAttentionExtension";
        if (op->get_type_name() == paged_attention_type) {
            CreatePagedAttention(*this, op);
            return;
        }
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
        // Note: Currently Split/Variadic Split are divided to multiple crops
        // LSTMCell contains its own body network, and each output has a unique pid
        // But there is no need to maintain output port index for the next node e.g. Result
        bool is_legacy_multiple_outputs = !allow_new_shape_infer
                                          || ov::is_type<ov::op::v1::Split>(prevOp)
                                          || ov::is_type<ov::op::v1::VariadicSplit>(prevOp)
                                          || ov::is_type<ov::op::v4::LSTMCell>(prevOp);
        if (prevOp->get_output_size() > 1 && is_legacy_multiple_outputs) {
            prevName += ".out" + std::to_string(op->get_input_source_output(i).get_index());
        }

        if (ov::is_type<op::Placeholder>(prevOp)) {
            inputInfo.push_back(cldnn::input_info{});
            continue;
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
    perfEntry.status = ov::ProfilingInfo::Status::EXECUTED;
    perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
    perfEntry.isCPU = false;
    perfEntry.parentPrimitive = prim.origin_op_name;
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

bool ProgramBuilder::requires_new_shape_infer(const std::shared_ptr<ov::Node>& op) const {
    if (op->is_dynamic()) {
        return true;
    }

    if (ov::is_type<ov::op::v5::Loop>(op)) {
        const auto body_function = std::static_pointer_cast<ov::op::v5::Loop>(op)->get_function();
        if (body_function->is_dynamic())
            return true;
    }
    // When input node has dynamic shape with 4 dimension, this function return false
    // because op.is_dynamic() which only checks input shapes return false.
    // So, in the case of input data, we need to check output shape.
    for (size_t i = 0; i < op->get_output_size(); i++) {
        if (op->get_output_partial_shape(i).is_dynamic())
            return true;
    }

    if (ov::is_type<op::FullyConnectedCompressed>(op))
        return true;

    for (size_t i = 0; i < op->get_output_size(); i++) {
        if (op->get_output_partial_shape(i).size() > 6)
            return true;
    }

    for (size_t i = 0; i < op->get_input_size(); i++) {
        if (op->get_input_partial_shape(i).size() > 6)
            return true;
    }

    return false;
}

int64_t ProgramBuilder::get_parameter_index(const std::shared_ptr<ov::op::v0::Parameter>& parameter) const {
    return m_model->get_parameter_index(parameter);
}

int64_t ProgramBuilder::get_result_index(const ov::Output<ov::Node>& value) const {
    return  m_model->get_result_index(value);
}

int64_t ProgramBuilder::get_result_index(const ov::Output<const ov::Node>& value) const {
    return m_model->get_result_index(value);
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
