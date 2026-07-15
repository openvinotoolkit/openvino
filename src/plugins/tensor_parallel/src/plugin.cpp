// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>
#include <string>
#include <vector>

#include "compiled_model.hpp"
#include "graph_rewriter.hpp"
#include "tp_l0_shared_context.hpp"
#include "tensor_parallel/tp_coordination.hpp"
#include "tensor_parallel/tp_device_coordinator.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/zero_api.hpp"

namespace ov {
namespace tp {

namespace {
inline void ze_throw_on_error(ze_result_t r, const char* what) {
    if (r != ZE_RESULT_SUCCESS) {
        OPENVINO_THROW("[TENSOR_PARALLEL] L0 call ", what, " failed: 0x", std::hex, r);
    }
}
}  // namespace
#define ZE_THROW_ON_ERROR(expr, what) ::ov::tp::ze_throw_on_error((expr), (what))

Plugin::Plugin() {
    set_device_name("TENSOR_PARALLEL");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    auto config = properties;
    config.insert(m_config.begin(), m_config.end());

    // ---- Extract TP configuration ----
    uint32_t tp_degree = 2;
    std::vector<std::string> device_names;

    auto it_degree = config.find("TENSOR_PARALLEL_DEGREE");
    if (it_degree != config.end()) {
        tp_degree = it_degree->second.as<uint32_t>();
        config.erase(it_degree);
    }

    auto it_devices = config.find("TENSOR_PARALLEL_DEVICES");
    if (it_devices != config.end()) {
        device_names = it_devices->second.as<std::vector<std::string>>();
        config.erase(it_devices);
    }

    // Auto-detect GPU devices if not specified
    if (device_names.empty()) {
        for (uint32_t i = 0; i < tp_degree; ++i) {
            device_names.push_back("GPU." + std::to_string(i));
        }
    }

    OPENVINO_ASSERT(device_names.size() >= 2,
                    "[TENSOR_PARALLEL] Need at least 2 devices, got ", device_names.size());
    OPENVINO_ASSERT(device_names.size() == tp_degree,
                    "[TENSOR_PARALLEL] Device count (", device_names.size(),
                    ") must match TP degree (", tp_degree, ")");

    // ---- Build execution plan ----
    //
    // 1. Analyze the model to identify shardable MatMuls.
    // 2. Create shared coordination object for cross-GPU AllReduce.
    // 3. For each rank: shard weights, insert TPAllReduce ops, compile on the rank's GPU.
    //    The GPU plugin handles TPAllReduce as a CPU-fallback primitive that
    //    synchronizes via the shared coordination object.

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

    auto coordination = std::make_shared<TPCoordination>(tp_degree, num_collectives);

    // ---- Build a single L0 context spanning every rank's GPU ----
    //
    // Cross-device L0 USM copies require all participating devices to belong
    // to the same ze_context_handle_t.  We discover each rank's (driver,
    // device) handles via the intel_gpu plugin's default RemoteContext
    // properties, then create one shared L0 context covering all of them and
    // hand it back to each rank's compile_model() call as a non-owning
    // ContextType::ZE remote context.
    TPL0SharedContextPtr shared_l0_ctx;
    std::vector<ov::SoPtr<ov::IRemoteContext>> rank_ctx(tp_degree);
    bool shared_ctx_ok = false;
    try {
        // Touch the L0 loader early; if zero_loader/zeInit fails we fall back
        // to the legacy per-rank path which still works for single-GPU cases.
        ZE_THROW_ON_ERROR(ov::zeInit(0), "zeInit");

        std::vector<ze_driver_handle_t>  drivers(tp_degree);
        std::vector<ze_device_handle_t>  rank_devs(tp_degree);

        for (uint32_t rank = 0; rank < tp_degree; ++rank) {
            auto def_ctx = get_core()->get_default_context(device_names[rank]);
            const auto& props = def_ctx->get_property();

            auto it_drv = props.find(ov::intel_gpu::ze_driver_handle.name());
            auto it_dev = props.find(ov::intel_gpu::ze_device_handle.name());
            OPENVINO_ASSERT(it_drv != props.end() && it_dev != props.end(),
                            "[TENSOR_PARALLEL] device ", device_names[rank],
                            " did not expose Level-Zero (driver, device) handles. "
                            "intel_gpu plugin must be built with GPU_RT_TYPE=L0.");

            drivers[rank]   = reinterpret_cast<ze_driver_handle_t>(it_drv->second.as<ov::intel_gpu::gpu_handle_param>());
            rank_devs[rank] = reinterpret_cast<ze_device_handle_t>(it_dev->second.as<ov::intel_gpu::gpu_handle_param>());
        }

        for (uint32_t rank = 1; rank < tp_degree; ++rank) {
            OPENVINO_ASSERT(drivers[rank] == drivers[0],
                            "[TENSOR_PARALLEL] all ranks must share a single L0 driver");
        }

        OPENVINO_ASSERT(ov::zeContextCreateEx != nullptr,
                        "[TENSOR_PARALLEL] L0 loader does not export zeContextCreateEx");

        ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ze_context_handle_t shared_ctx_h = nullptr;
        ZE_THROW_ON_ERROR(
            ov::zeContextCreateEx(drivers[0], &ctx_desc,
                                  static_cast<uint32_t>(rank_devs.size()),
                                  rank_devs.data(), &shared_ctx_h),
            "zeContextCreateEx");

        shared_l0_ctx = std::make_shared<TPL0SharedContext>();
        shared_l0_ctx->driver  = drivers[0];
        shared_l0_ctx->devices = rank_devs;
        shared_l0_ctx->context = shared_ctx_h;

        for (uint32_t rank = 0; rank < tp_degree; ++rank) {
            ov::AnyMap rank_params{
                {ov::intel_gpu::context_type.name(),     ov::intel_gpu::ContextType::ZE},
                {ov::intel_gpu::ze_context.name(),       static_cast<ov::intel_gpu::gpu_handle_param>(shared_ctx_h)},
                {ov::intel_gpu::ze_device_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(rank_devs[rank])},
                {ov::intel_gpu::ze_driver_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(drivers[0])},
            };
            rank_ctx[rank] = get_core()->create_context(device_names[rank], rank_params);
        }
        shared_ctx_ok = true;
        std::cerr << "[TP] Created shared L0 context across " << tp_degree
                  << " devices (ctx=" << shared_ctx_h << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[TP] Shared L0 context unavailable: " << e.what()
                  << " -- falling back to per-rank contexts." << std::endl;
        shared_l0_ctx.reset();
        shared_ctx_ok = false;
    }

    // Build the device-side AllReduce coordinator on top of the shared L0
    // context. Failure here is non-fatal: we simply fall through to the
    // legacy CPU-staged AllReduce path.
    TPDeviceCoordinatorPtr device_coordinator;
    if (shared_ctx_ok && num_collectives > 0) {
        try {
            device_coordinator = std::make_shared<TPDeviceCoordinator>(
                shared_l0_ctx, static_cast<int>(tp_degree), num_collectives);
        } catch (const std::exception& e) {
            std::cerr << "[TP] Device coordinator init failed: " << e.what()
                      << " -- AllReduce will use the CPU-staged fallback." << std::endl;
            device_coordinator.reset();
        }
    }

    // Hand the device coordinator to the coordination object so that the
    // CPU-side `tp_allreduce` primitive picked up by intel_gpu can choose
    // the device fast-path when its inputs/outputs live in USM-device.
    if (device_coordinator) {
        coordination->set_device_coordinator(device_coordinator);
    }

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(tp_degree);

    for (uint32_t rank = 0; rank < tp_degree; ++rank) {
        auto rank_model = GraphRewriter::rewrite(model, sharding_plan, rank, tp_degree, coordination);

        std::cerr << "[TP] Rank " << rank << ": " << rank_model->get_ordered_ops().size()
                  << " ops, " << num_collectives << " AllReduce points" << std::endl;

        if (shared_ctx_ok) {
            rank_compiled[rank] = get_core()->compile_model(
                rank_model, rank_ctx[rank], config);
        } else {
            rank_compiled[rank] = get_core()->compile_model(
                rank_model, device_names[rank], config);
        }
    }

    return std::make_shared<CompiledModel>(model, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(device_names),
                                           std::move(shared_l0_ctx),
                                           std::move(device_coordinator));
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
        };
    } else if (name == ov::device::full_name.name()) {
        return std::string("TENSOR_PARALLEL");
    } else if (name == ov::device::capabilities.name()) {
        return std::vector<std::string>{};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{};
    }
    OPENVINO_THROW("[TENSOR_PARALLEL] Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
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

}  // namespace tp
}  // namespace ov
