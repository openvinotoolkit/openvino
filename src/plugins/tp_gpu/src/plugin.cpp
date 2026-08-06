// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "compiled_model.hpp"
#include "graph_rewriter.hpp"
#include "tp_l0_shared_context.hpp"
#include "tp_gpu/properties.hpp"
#include "intel_gpu/runtime/collective_comm_registry.hpp"
#include "tp_gpu/tp_device_coordinator.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/zero_api.hpp"

namespace ov {
namespace tp_gpu {

namespace {
inline void ze_throw_on_error(ze_result_t r, const char* what) {
    if (r != ZE_RESULT_SUCCESS) {
        OPENVINO_THROW("[TP_GPU] L0 call ", what, " failed: 0x", std::hex, r);
    }
}

/// \brief Resolves the per-rank device list.
///
/// The number of ranks is whatever this returns: `DEVICE_IDS` wins when given,
/// otherwise `TP_SIZE` ranks are mapped onto GPU.0 .. GPU.{TP_SIZE - 1}. When
/// both are set they must agree.
std::vector<std::string> get_device_names(const ov::AnyMap& config) {
    std::vector<std::string> device_names;

    auto it_devices = config.find(device_ids.name());
    if (it_devices != config.end()) {
        device_names = it_devices->second.as<std::vector<std::string>>();
    }

    size_t devices_count = device_names.size();
    auto it_size = config.find(tp_size.name());
    if (it_size != config.end()) {
        devices_count = it_size->second.as<uint32_t>();
    }

    OPENVINO_ASSERT(it_devices != config.end() || it_size != config.end(),
                    "[TP_GPU] Neither TP_SIZE nor DEVICE_IDS was set, at least one of them must be specified");

    if (device_names.empty()) {
        device_names.reserve(devices_count);
        for (uint32_t i = 0; i < devices_count; ++i) {
            device_names.push_back("GPU." + std::to_string(i));
        }
    } else {
        OPENVINO_ASSERT(device_names.size() == devices_count,
                        "[TP_GPU] ", device_ids.name(), " lists ", device_names.size(),
                        " devices, which does not match ", tp_size.name(), "=", devices_count);
    }

    OPENVINO_ASSERT(device_names.size() >= 2,
                    "[TP_GPU] Need at least 2 devices, got ", device_names.size());

    return device_names;
}

/// \brief Reads the collective timeout, in milliseconds.
///
/// Zero disables the bound entirely, which is only useful when stepping
/// through a collective under a debugger.
std::chrono::milliseconds get_collective_timeout(const ov::AnyMap& config) {
    auto it = config.find(communication_timeout_ms.name());
    if (it == config.end()) {
        return std::chrono::milliseconds{5000};
    }
    return std::chrono::milliseconds{it->second.as<uint32_t>()};
}

/// \brief Drops the keys owned by this plugin.
///
/// Whatever is left in the config is forwarded to the GPU plugin verbatim, and
/// it rejects properties it does not recognize.
void erase_tp_keys(ov::AnyMap& config) {
    config.erase(tp_size.name());
    config.erase(device_ids.name());
    config.erase(communication_timeout_ms.name());
}
}  // namespace
#define ZE_THROW_ON_ERROR(expr, what) ::ov::tp_gpu::ze_throw_on_error((expr), (what))

Plugin::Plugin() {
    set_device_name("TP_GPU");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    auto config = properties;
    config.insert(m_config.begin(), m_config.end());

    // ---- Extract TP configuration ----
    auto device_names = get_device_names(config);
    const auto tp_degree = static_cast<uint32_t>(device_names.size());
    const auto collective_timeout = get_collective_timeout(config);
    erase_tp_keys(config);

    // ---- Build execution plan ----
    //
    // 1. Analyze the model to identify shardable MatMuls.
    // 2. Create shared coordinator object for cross-GPU AllReduce.
    // 3. For each rank: shard weights, insert TPAllReduce ops, compile on the rank's GPU.
    //    The GPU plugin handles TPAllReduce as a CPU-fallback primitive that
    //    synchronizes via the shared coordinator object.

    // ---- Disable dynamic quantization for TP ----
    // DQ (DynamicQuantize) + oneDNN FC produces shape-dependent results:
    // TP shards FC weights, changing K (row-parallel) and N (column-parallel)
    // dimensions.  oneDNN's internal BRGEMM kernel uses different tiling for
    // different shapes, altering the i8*i4 accumulation order.  At seq_len >= 80
    // the BRGEMM JIT kernel switches, causing max_abs_diff to jump from ~0.04
    // to >1.0 -- unacceptable for LLM inference.
    //
    // Disabling DQ (group_size=0) while keeping oneDNN enabled gives f16 FC
    // with ~0.04 max_abs_diff, which is acceptable.
    auto dq_key = ov::hint::dynamic_quantization_group_size.name();
    if (config.find(dq_key) == config.end()) {
        config[dq_key] = (uint64_t)0;
    }

    auto sharding_plan = GraphRewriter::analyze(model);
    int num_collectives = GraphRewriter::count_collectives(sharding_plan);

    if (std::getenv("TP_PROF") != nullptr) {
        const auto column_count = sharding_plan.linears.size() - static_cast<size_t>(num_collectives);
        const auto biased = std::count_if(sharding_plan.linears.begin(), sharding_plan.linears.end(),
                                          [](const ShardingPlan::LinearDesc& l) { return l.has_bias; });
        std::cerr << "[TP] Sharding plan: layers=" << sharding_plan.num_layers
                  << " linears=" << sharding_plan.linears.size()
                  << " (column=" << column_count << " row=" << num_collectives
                  << " biased=" << biased << ")"
                  << " heads=" << sharding_plan.num_heads
                  << " kv_heads=" << sharding_plan.num_kv_heads
                  << " head_dim=" << sharding_plan.head_dim
                  << " hidden=" << sharding_plan.hidden_size
                  << " intermediate=" << sharding_plan.intermediate_size << std::endl;
    }

    // Attention is split by KV head.  When they do not divide evenly the first
    // ranks take one extra each, and since every layer ends in a collective the
    // slowest rank paces the whole model -- worth saying out loud, because the
    // model still runs and the cost is invisible otherwise.
    if (const int remainder = sharding_plan.num_kv_heads % static_cast<int>(tp_degree)) {
        const int base = sharding_plan.num_kv_heads / static_cast<int>(tp_degree);
        const int imbalance_pct = 100 / (base + 1);
        if (imbalance_pct > 10) {
            std::cerr << "[TP_GPU] Warning: " << sharding_plan.num_kv_heads
                      << " KV heads do not divide evenly across " << tp_degree << " ranks ("
                      << remainder << " rank(s) get " << (base + 1) << ", the rest " << base
                      << "), about " << imbalance_pct << "% load imbalance." << std::endl;
        }
    }

    // ---- Build a single L0 context spanning every rank's GPU ----
    //
    // Cross-device L0 USM copies require all participating devices to belong
    // to the same ze_context_handle_t.  We discover each rank's (driver,
    // device) handles via the intel_gpu plugin's default RemoteContext
    // properties, then create one shared L0 context covering all of them and
    // hand it back to each rank's compile_model() call as a non-owning
    // ContextType::ZE remote context.
    //
    // Every failure here is fatal by design: without the shared context the
    // collectives would have to fall back to host-staged reductions, which is
    // slow enough to look like a hang on an LLM.  A clear error beats silent
    // degradation.
    ZE_THROW_ON_ERROR(ov::zeInit(0), "zeInit");

    std::vector<ze_driver_handle_t> drivers(tp_degree);
    std::vector<ze_device_handle_t> rank_devs(tp_degree);

    for (uint32_t rank = 0; rank < tp_degree; ++rank) {
        auto def_ctx = get_core()->get_default_context(device_names[rank]);
        const auto& props = def_ctx->get_property();

        auto it_drv = props.find(ov::intel_gpu::ze_driver_handle.name());
        auto it_dev = props.find(ov::intel_gpu::ze_device_handle.name());
        OPENVINO_ASSERT(it_drv != props.end() && it_dev != props.end(),
                        "[TP_GPU] device ", device_names[rank],
                        " did not expose Level-Zero (driver, device) handles. "
                        "intel_gpu plugin must be built with GPU_RT_TYPE=L0.");

        drivers[rank] = reinterpret_cast<ze_driver_handle_t>(it_drv->second.as<ov::intel_gpu::gpu_handle_param>());
        rank_devs[rank] = reinterpret_cast<ze_device_handle_t>(it_dev->second.as<ov::intel_gpu::gpu_handle_param>());
    }

    for (uint32_t rank = 1; rank < tp_degree; ++rank) {
        OPENVINO_ASSERT(drivers[rank] == drivers[0],
                        "[TP_GPU] all ranks must share a single L0 driver, but ", device_names[rank],
                        " belongs to a different one than ", device_names[0]);
    }

    OPENVINO_ASSERT(ov::zeContextCreateEx != nullptr,
                    "[TP_GPU] L0 loader does not export zeContextCreateEx, which is required to "
                    "share one context across devices. Update the Level-Zero loader.");

    ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t shared_ctx_h = nullptr;
    ZE_THROW_ON_ERROR(ov::zeContextCreateEx(drivers[0], &ctx_desc,
                                            static_cast<uint32_t>(rank_devs.size()),
                                            rank_devs.data(), &shared_ctx_h),
                      "zeContextCreateEx");

    auto shared_l0_ctx = std::make_shared<TPL0SharedContext>();
    shared_l0_ctx->driver = drivers[0];
    shared_l0_ctx->devices = rank_devs;
    shared_l0_ctx->context = shared_ctx_h;

    std::vector<ov::SoPtr<ov::IRemoteContext>> rank_ctx(tp_degree);
    for (uint32_t rank = 0; rank < tp_degree; ++rank) {
        ov::AnyMap rank_params{
            {ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::ZE},
            {ov::intel_gpu::ze_context.name(), static_cast<ov::intel_gpu::gpu_handle_param>(shared_ctx_h)},
            {ov::intel_gpu::ze_device_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(rank_devs[rank])},
            {ov::intel_gpu::ze_driver_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(drivers[0])},
        };
        rank_ctx[rank] = get_core()->create_context(device_names[rank], rank_params);
    }

    if (std::getenv("TP_PROF") != nullptr) {
        std::cerr << "[TP] Created shared L0 context across " << tp_degree << " devices (ctx=" << shared_ctx_h
                  << ")" << std::endl;
    }

    // Device-side AllReduce coordinator on top of the shared L0 context.
    TPDeviceCoordinatorPtr coordinator;
    if (num_collectives > 0) {
        coordinator = std::make_shared<TPDeviceCoordinator>(shared_l0_ctx,
                                                           static_cast<int>(tp_degree),
                                                           num_collectives,
                                                           collective_timeout);
    }

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(tp_degree);

    for (uint32_t rank = 0; rank < tp_degree; ++rank) {
        auto rank_model = GraphRewriter::rewrite(model, sharding_plan, rank, tp_degree);

        if (std::getenv("TP_PROF") != nullptr) {
            std::cerr << "[TP] Rank " << rank << ": " << rank_model->get_ordered_ops().size() << " ops, "
                      << num_collectives << " AllReduce points" << std::endl;
        }

        rank_compiled[rank] = get_core()->compile_model(rank_model, rank_ctx[rank], config);
    }

    if (coordinator) {
        // Hand the collective state over once the networks exist.  Doing it here
        // rather than through compile_model properties keeps the graph free of
        // runtime pointers and makes the same path work after import_model.
        //
        // One registry per stream worker, shared by every rank of that worker.
        // Only a single worker exists until ov::num_streams is supported.
        auto registry = std::make_shared<ov::intel_gpu::CollectiveCommRegistry>();
        registry->set_group(0, coordinator);
        std::vector<ov::intel_gpu::CollectiveCommRegistryPtr> registry_set{registry};

        for (auto& compiled : rank_compiled) {
            compiled->set_property({{ov::intel_gpu::collective_comm_registry_set.name(), registry_set}});
        }
    }

    return std::make_shared<CompiledModel>(model, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(device_names),
                                           std::move(shared_l0_ctx),
                                           std::move(coordinator));
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    m_config.insert(properties.begin(), properties.end());
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::tp_gpu::tp_size.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::tp_gpu::device_ids.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::tp_gpu::communication_timeout_ms.name(), ov::PropertyMutability::RW},
        };
    } else if (name == ov::device::full_name.name()) {
        return std::string("TP_GPU");
    } else if (name == ov::device::capabilities.name()) {
        return std::vector<std::string>{};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{};
    } else if (name == ov::tp_gpu::communication_timeout_ms.name()) {
        auto it = m_config.find(name);
        return it != m_config.end() ? it->second : ov::Any{uint32_t{5000}};
    } else if (name == ov::tp_gpu::tp_size.name() || name == ov::tp_gpu::device_ids.name()) {
        auto it = m_config.find(name);
        OPENVINO_ASSERT(it != m_config.end(), "[TP_GPU] Property ", name, " was not set");
        return it->second;
    }
    OPENVINO_THROW("[TP_GPU] Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model != nullptr, "[TP_GPU] query_model: model is null");

    auto config = properties;
    config.insert(m_config.begin(), m_config.end());

    const auto device_names = get_device_names(config);
    erase_tp_keys(config);

    // Every rank compiles the same op set, so whatever the GPU plugin supports
    // on one rank is what TP_GPU supports as a whole. The collectives inserted
    // later are invisible at query time -- they do not exist in the user model.
    auto supported = get_core()->query_model(model, device_names.front(), config);
    for (auto& entry : supported) {
        entry.second = get_device_name();
    }
    return supported;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&, const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&, const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

}  // namespace tp_gpu
}  // namespace ov
