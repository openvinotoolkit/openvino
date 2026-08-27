// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <algorithm>
#include <cstring>
#include <istream>
#include <memory>
#include <streambuf>
#include <string>
#include <vector>

#include "compiled_model.hpp"
#include "graph_rewriter.hpp"
#include "tp_blob.hpp"
#include "tp_l0_shared_context.hpp"
#include "tp_gpu/properties.hpp"
#include "intel_gpu/runtime/collective_comm_registry.hpp"
#include "tp_gpu/tp_device_coordinator.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/zero_api.hpp"

namespace ov {
namespace tp_gpu {

namespace {
/// Sanity bounds for values read back from a compiled blob. They are not
/// hardware limits, just the point past which the stream is certainly not a
/// TP_GPU blob any more.
constexpr uint32_t kMaxWorldSize = 64;
constexpr uint32_t kMaxCollectives = 1u << 16;
constexpr uint32_t kMaxShardedStates = 1u << 16;
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

/// \brief Same as `get_device_names`, but tolerates a config that says nothing
/// about the topology.
///
/// On import the topology comes from the blob; the config only gets a say when
/// the caller explicitly asked for one, and then it has to agree.
std::vector<std::string> get_requested_device_names(const ov::AnyMap& config) {
    if (config.count(tp_size.name()) == 0 && config.count(device_ids.name()) == 0) {
        return {};
    }
    return get_device_names(config);
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

/// \brief Read-only, seekable stream over an already materialized blob.
///
/// `import_model(ov::Tensor)` hands us the whole blob in memory. The rank
/// readers want an `std::istream`, and the TP container needs to seek between
/// rank boundaries, so a plain buffer view is enough -- no copy.
class MemoryInputBuffer : public std::streambuf {
public:
    MemoryInputBuffer(const char* data, std::size_t size) {
        auto* begin = const_cast<char*>(data);
        setg(begin, begin, begin + size);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::in) == 0) {
            return pos_type(off_type(-1));
        }
        off_type target = off;
        if (dir == std::ios_base::cur) {
            target += gptr() - eback();
        } else if (dir == std::ios_base::end) {
            target += egptr() - eback();
        }
        if (target < 0 || target > egptr() - eback()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), eback() + target, egptr());
        return pos_type(target);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};
}  // namespace
#define ZE_THROW_ON_ERROR(expr, what) ::ov::tp_gpu::ze_throw_on_error((expr), (what))

Plugin::Plugin() {
    set_device_name("TP_GPU");
}

Plugin::SharedL0Setup Plugin::create_shared_l0(const std::vector<std::string>& device_names) const {
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
    const auto tp_degree = static_cast<uint32_t>(device_names.size());
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

    SharedL0Setup setup;
    setup.shared = std::make_shared<TPL0SharedContext>();
    setup.shared->driver = drivers[0];
    setup.shared->devices = rank_devs;
    setup.shared->context = shared_ctx_h;

    setup.rank_ctx.resize(tp_degree);
    for (uint32_t rank = 0; rank < tp_degree; ++rank) {
        ov::AnyMap rank_params{
            {ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::ZE},
            {ov::intel_gpu::ze_context.name(), static_cast<ov::intel_gpu::gpu_handle_param>(shared_ctx_h)},
            {ov::intel_gpu::ze_device_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(rank_devs[rank])},
            {ov::intel_gpu::ze_driver_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(drivers[0])},
        };
        setup.rank_ctx[rank] = get_core()->create_context(device_names[rank], rank_params);
    }

    if (std::getenv("TP_PROF") != nullptr) {
        std::cerr << "[TP] Created shared L0 context across " << tp_degree << " devices (ctx=" << shared_ctx_h
                  << ")" << std::endl;
    }

    return setup;
}

/// \brief Hands the collective state to the rank networks.
///
/// Done after the networks exist rather than through compile_model properties:
/// that keeps runtime pointers out of the graph and out of the cache hash, and
/// makes the very same path work for an imported model.
///
/// One registry per stream worker, shared by every rank of that worker. Only a
/// single worker exists until ov::num_streams is supported.
static void attach_collective_registry(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled,
                                       const TPDeviceCoordinatorPtr& coordinator) {
    auto registry = std::make_shared<ov::intel_gpu::CollectiveCommRegistry>();
    registry->set_group(0, coordinator);
    std::vector<ov::intel_gpu::CollectiveCommRegistryPtr> registry_set{registry};

    for (const auto& compiled : rank_compiled) {
        compiled->set_property({{ov::intel_gpu::collective_comm_registry_set.name(), registry_set}});
    }
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

    auto sharding_plan = GraphRewriter::analyze(model);
    int num_collectives = GraphRewriter::count_collectives(sharding_plan, static_cast<int>(tp_degree));

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
    auto setup = create_shared_l0(device_names);

    // Device-side AllReduce coordinator on top of the shared L0 context.
    TPDeviceCoordinatorPtr coordinator;
    if (num_collectives > 0) {
        coordinator = std::make_shared<TPDeviceCoordinator>(setup.shared,
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

        rank_compiled[rank] = get_core()->compile_model(rank_model, setup.rank_ctx[rank], config);
    }

    if (coordinator) {
        attach_collective_registry(rank_compiled, coordinator);
    }

    return std::make_shared<CompiledModel>(model, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(device_names),
                                           std::move(setup.shared),
                                           std::move(coordinator),
                                           GraphRewriter::sharded_state_ids(model, sharding_plan));
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    m_config.insert(properties.begin(), properties.end());
}

std::vector<std::string> Plugin::query_devices(const ov::AnyMap& arguments) const {
    ov::AnyMap merged = arguments;
    merged.insert(m_config.begin(), m_config.end());

    if (auto it = merged.find(device_ids.name()); it != merged.end()) {
        auto names = it->second.as<std::vector<std::string>>();
        if (!names.empty()) {
            return names;
        }
    }
    if (auto it = merged.find(tp_size.name()); it != merged.end()) {
        const auto count = it->second.as<uint32_t>();
        std::vector<std::string> names;
        names.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            names.push_back("GPU." + std::to_string(i));
        }
        if (!names.empty()) {
            return names;
        }
    }
    // Nothing to go on yet -- the caller is asking about the plugin, not about
    // a particular run. The GPU plugin's own default device answers for it.
    return {"GPU"};
}

std::vector<ov::PropertyName> Plugin::gpu_caching_properties(const ov::AnyMap& arguments) const {
    return get_core()->get_property(query_devices(arguments).front(), ov::internal::caching_properties);
}

ov::Any Plugin::aggregate_rank_property(const std::string& name, const ov::AnyMap& arguments) const {
    // A TP blob is a container of one intel_gpu blob per rank, so anything
    // that makes a single rank's blob unique makes the whole thing unique.
    // Values are joined across ranks so a heterogeneous set of GPUs cannot
    // collide with a homogeneous one in the cache hash.
    const auto devices = query_devices(arguments);
    std::string joined;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) {
            joined += ";";
        }
        joined += get_core()->get_property(devices[i], name, {}).as<std::string>();
    }
    return joined;
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
        return std::vector<std::string>{ov::device::capability::EXPORT_IMPORT};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
        };
    } else if (name == ov::internal::caching_properties.name()) {
        auto properties = gpu_caching_properties(arguments);
        // The rank topology is ours alone: the same model on 2 and on 4 ranks
        // produces different blobs that the GPU properties cannot tell apart.
        properties.emplace_back(ov::tp_gpu::tp_size.name(), ov::PropertyMutability::RW);
        properties.emplace_back(ov::tp_gpu::device_ids.name(), ov::PropertyMutability::RW);
        return properties;
    } else if (name == ov::tp_gpu::communication_timeout_ms.name()) {
        auto it = m_config.find(name);
        return it != m_config.end() ? it->second : ov::Any{uint32_t{5000}};
    } else if (name == ov::tp_gpu::tp_size.name()) {
        // Reported as "unset" instead of refused: the cache hash queries every
        // caching property before any topology has been chosen, and a throw
        // there would disable caching outright.
        auto it = m_config.find(name);
        return it != m_config.end() ? it->second : ov::Any{uint32_t{0}};
    } else if (name == ov::tp_gpu::device_ids.name()) {
        auto it = m_config.find(name);
        return it != m_config.end() ? it->second : ov::Any{std::vector<std::string>{}};
    }

    // Everything the GPU plugin folds into its cache hash has to be answerable
    // here too, otherwise Core cannot build a compile config for TP_GPU.
    if (ov::util::contains(gpu_caching_properties(arguments), name)) {
        return aggregate_rank_property(name, arguments);
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_blob(std::istream& blob, const ov::AnyMap& properties) const {
    auto config = properties;
    config.insert(m_config.begin(), m_config.end());

    const auto collective_timeout = get_collective_timeout(config);
    const auto requested_devices = get_requested_device_names(config);
    erase_tp_keys(config);

    // The TP container is what we are reading right now. Forwarding it would
    // make the GPU plugin try to import the outer blob instead of its own.
    config.erase(ov::hint::compiled_blob.name());

    // ---- Header ----
    char blob_magic[sizeof(tp_blob::magic)] = {};
    blob.read(blob_magic, sizeof(blob_magic));
    OPENVINO_ASSERT(blob.gcount() == static_cast<std::streamsize>(sizeof(blob_magic)) &&
                        std::memcmp(blob_magic, tp_blob::magic, sizeof(blob_magic)) == 0,
                    "[TP_GPU] the stream does not hold a TP_GPU compiled blob");

    const auto blob_version = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(blob_version == tp_blob::version,
                    "[TP_GPU] compiled blob has format version ", blob_version,
                    ", this runtime writes and reads version ", tp_blob::version);

    const auto world_size = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(world_size >= 2 && world_size <= kMaxWorldSize,
                    "[TP_GPU] compiled blob declares ", world_size,
                    " ranks, which is outside the supported range [2, ", kMaxWorldSize, "]");

    const auto num_collectives = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(num_collectives <= kMaxCollectives,
                    "[TP_GPU] compiled blob declares ", num_collectives,
                    " collectives, which is not plausible; the cache entry is corrupted");

    const auto num_sharded_states = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(num_sharded_states <= kMaxShardedStates,
                    "[TP_GPU] compiled blob declares ", num_sharded_states,
                    " sharded states, which is not plausible; the cache entry is corrupted");

    std::vector<std::string> device_names(world_size);
    for (auto& name : device_names) {
        name = tp_blob::read_string(blob);
    }

    std::vector<std::string> sharded_state_ids(num_sharded_states);
    for (auto& id : sharded_state_ids) {
        id = tp_blob::read_string(blob, tp_blob::max_variable_id_length);
    }

    // The blob is bound to the topology it was produced on: rank r's graph was
    // compiled for a specific device, and the collectives assume that exact
    // rank order. Silently retargeting would produce a model that runs and is
    // wrong.
    OPENVINO_ASSERT(requested_devices.empty() || requested_devices == device_names,
                    "[TP_GPU] compiled blob was produced for ", ov::Any(device_names).as<std::string>(),
                    " but ", ov::Any(requested_devices).as<std::string>(), " was requested");

    // ---- Rank blobs ----
    auto setup = create_shared_l0(device_names);

    TPDeviceCoordinatorPtr coordinator;
    if (num_collectives > 0) {
        coordinator = std::make_shared<TPDeviceCoordinator>(setup.shared,
                                                            static_cast<int>(world_size),
                                                            static_cast<int>(num_collectives),
                                                            collective_timeout);
    }

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(world_size);
    for (uint32_t rank = 0; rank < world_size; ++rank) {
        const auto rank_blob_size = tp_blob::read_trivial<uint64_t>(blob);
        OPENVINO_ASSERT(rank_blob_size > 0, "[TP_GPU] compiled blob has an empty payload for rank ", rank);

        const auto rank_blob_start = blob.tellg();
        OPENVINO_ASSERT(rank_blob_start != std::istream::pos_type(-1),
                        "[TP_GPU] import_model needs a seekable stream to walk per-rank blobs");

        rank_compiled[rank] = get_core()->import_model(blob, setup.rank_ctx[rank], config);
        OPENVINO_ASSERT(rank_compiled[rank],
                        "[TP_GPU] the GPU plugin refused the cached blob of rank ", rank,
                        " on ", device_names[rank], "; it was compiled with a different cache mode");

        // The GPU reader stops wherever its own format ends, which need not be
        // the byte we recorded. Realign so the next rank starts in the right
        // place instead of parsing the tail of its predecessor.
        blob.clear();
        blob.seekg(rank_blob_start + static_cast<std::istream::off_type>(rank_blob_size));
        OPENVINO_ASSERT(blob.good(), "[TP_GPU] truncated compiled blob: rank ", rank, " payload is incomplete");
    }

    if (coordinator) {
        attach_collective_registry(rank_compiled, coordinator);
    }

    // No IR here: the user model was consumed at compile time and is not part
    // of the blob. CompiledModel derives its ports from rank 0 instead.
    return std::make_shared<CompiledModel>(nullptr, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(device_names),
                                           std::move(setup.shared),
                                           std::move(coordinator),
                                           std::move(sharded_state_ids),
                                           /*loaded_from_cache=*/true);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    return import_blob(model, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_THROW("[TP_GPU] importing into a caller-provided remote context is not supported: the plugin "
                   "creates its own Level-Zero context spanning every rank device");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model, const ov::AnyMap& properties) const {
    MemoryInputBuffer buffer(static_cast<const char*>(model.data()), model.get_byte_size());
    std::istream stream(&buffer);
    return import_blob(stream, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_THROW("[TP_GPU] importing into a caller-provided remote context is not supported: the plugin "
                   "creates its own Level-Zero context spanning every rank device");
}

}  // namespace tp_gpu
}  // namespace ov
