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
#include "tp_ze_throw.hpp"
#include "tp_gpu/op/tp_all_reduce.hpp"
#include "tp_gpu/op/tp_gather.hpp"
#include "tp_gpu/tp_config.hpp"
#include "tp_gpu/tp_debug.hpp"
#include "tp_gpu/tp_device_coordinator.hpp"
#include "intel_gpu/runtime/collective_comm_registry.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/zero_api.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace tp_gpu {

namespace {
/// Diagnostic (TP_SHARD_ONLY=1): strip every collective from a shard so it can
/// run alone on one GPU -- the upper bound tensor parallelism is chasing, with
/// no collectives, no cross-rank dispatch and no coordinator.
///
/// AllReduce preserves its shape, so it is dropped by reconnecting consumers to
/// its input. TPGather does not: on rank 0 it widens the vocabulary axis, and
/// its consumer is the model's Result. It becomes a concatenation of the shard
/// with itself -- the shape returns, the weights stay sharded, and only a copy
/// of a few hundred kilobytes of logits is added.
///
/// The values such a model produces are wrong; only its timings may be read.
void strip_collectives(const std::shared_ptr<ov::Model>& shard) {
    size_t dropped = 0;
    size_t widened = 0;
    for (const auto& node : shard->get_ordered_ops()) {
        if (!ov::is_type_any_of<op::TPAllReduce, op::TPGather>(node)) {
            continue;
        }
        OPENVINO_ASSERT(node->get_input_size() == 1 && node->get_output_size() == 1,
                        "[TP_GPU] TP_SHARD_ONLY cannot handle ", node->get_type_info().name,
                        ": it does not have exactly one input and one output.");

        const auto& in_shape = node->get_input_partial_shape(0);
        const auto& out_shape = node->get_output_partial_shape(0);
        if (out_shape.compatible(in_shape)) {
            ov::replace_output_update_name(node->output(0), node->input_value(0));
            ++dropped;
            continue;
        }

        // Find the one axis that grew and repeat the input along it.
        OPENVINO_ASSERT(in_shape.rank().is_static() && out_shape.rank() == in_shape.rank(),
                        "[TP_GPU] TP_SHARD_ONLY cannot widen ", node->get_type_info().name,
                        ": ranks differ or are dynamic.");
        int64_t axis = -1;
        int64_t factor = 0;
        for (int64_t i = 0; i < in_shape.rank().get_length(); ++i) {
            if (in_shape[i] == out_shape[i]) {
                continue;
            }
            OPENVINO_ASSERT(axis < 0 && in_shape[i].is_static() && out_shape[i].is_static() &&
                                out_shape[i].get_length() % in_shape[i].get_length() == 0,
                            "[TP_GPU] TP_SHARD_ONLY cannot widen ", node->get_type_info().name,
                            ": expected exactly one axis to grow by a whole factor, got ",
                            in_shape, " -> ", out_shape);
            axis = i;
            factor = out_shape[i].get_length() / in_shape[i].get_length();
        }
        OPENVINO_ASSERT(axis >= 0, "[TP_GPU] TP_SHARD_ONLY: ", node->get_type_info().name,
                        " changes shape but no axis grew: ", in_shape, " -> ", out_shape);

        ov::OutputVector copies(static_cast<std::size_t>(factor), node->input_value(0));
        auto concat = std::make_shared<ov::op::v0::Concat>(copies, axis);
        concat->set_friendly_name(node->get_friendly_name());
        ov::replace_node(node, concat);
        ++widened;
    }
    shard->validate_nodes_and_infer_types();

    TP_WARN_ALWAYS << "[TP_GPU] " << ov::tp_gpu::shard_only.name() << ": " << dropped
                   << " collectives short-circuited, " << widened
                   << " replaced by a self-concat. Results of this model are meaningless.";
}

/// \brief Read-only, seekable stream over an already materialized blob.
///
/// The rank readers want an `std::istream` and the TP container has to seek
/// between rank boundaries, so a buffer view is enough -- no copy.
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

/// Hands the collective state to the rank networks.
///
/// State of a compiled model, not a compilation parameter -- hence through the
/// compiled model, which also covers the imported one.
///
/// One registry per stream worker, shared by every rank of that worker. Only a
/// single worker exists until ov::num_streams is supported.
void attach_collective_registry(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled,
                                const TPDeviceCoordinatorPtr& coordinator) {
    auto registry = std::make_shared<ov::intel_gpu::CollectiveCommRegistry>();
    registry->set_coordinator(coordinator);
    std::vector<ov::intel_gpu::CollectiveCommRegistryPtr> registry_set{registry};

    for (const auto& compiled : rank_compiled) {
        compiled->set_property({{ov::intel_gpu::collective_comm_registry_set.name(), registry_set}});
    }
}

TPDeviceCoordinatorPtr make_coordinator(const TPL0SharedContextPtr& shared,
                                        uint32_t world_size,
                                        int num_collectives,
                                        std::chrono::milliseconds timeout,
                                        const TPConfig& config) {
    if (num_collectives <= 0) {
        return nullptr;
    }
    return std::make_shared<TPDeviceCoordinator>(shared, static_cast<int>(world_size), num_collectives, timeout, config);
}

void log_sharding_plan(const ShardingPlan& plan, int num_collectives) {
    const auto biased = std::count_if(plan.linears.begin(), plan.linears.end(), [](const auto& l) {
        return l.has_bias;
    });
    TP_LOG_INFO << "[TP] Sharding plan: layers=" << plan.num_layers
                << " linears=" << plan.linears.size()
                << " (column=" << (plan.linears.size() - static_cast<size_t>(num_collectives))
                << " row=" << num_collectives << " biased=" << biased << ")"
                << " heads=" << plan.num_heads
                << " kv_heads=" << plan.num_kv_heads
                << " head_dim=" << plan.head_dim
                << " hidden=" << plan.hidden_size
                << " intermediate=" << plan.intermediate_size << std::endl;
}

/// Attention is split by KV head. An uneven split gives the first ranks one
/// extra each, and since every layer ends in a collective the slowest rank
/// paces the whole model -- worth saying out loud, because the model still runs
/// and the cost is invisible otherwise.
void warn_on_kv_imbalance(const ShardingPlan& plan, uint32_t shard_degree) {
    const int remainder = plan.num_kv_heads % static_cast<int>(shard_degree);
    if (remainder == 0) {
        return;
    }
    const int base = plan.num_kv_heads / static_cast<int>(shard_degree);
    const int imbalance_pct = 100 / (base + 1);
    if (imbalance_pct <= 10) {
        return;
    }
    TP_WARN_ALWAYS << "[TP_GPU] Warning: " << plan.num_kv_heads << " KV heads do not divide evenly across "
                   << shard_degree << " ranks (" << remainder << " rank(s) get " << (base + 1) << ", the rest "
                   << base << "), about " << imbalance_pct << "% load imbalance.";
}

constexpr const char* kNoRemoteContextImport =
    "[TP_GPU] importing into a caller-provided remote context is not supported: the plugin "
    "creates its own Level-Zero context spanning every rank device";

/// Everything a TP blob declares before the first rank payload.
struct BlobHeader {
    uint32_t world_size = 0;
    uint32_t num_collectives = 0;
    std::vector<std::string> device_names;
    std::vector<std::string> sharded_state_ids;
};

/// Leaves the stream positioned at the first rank payload.
BlobHeader read_blob_header(std::istream& blob) {
    char magic[sizeof(tp_blob::magic)] = {};
    blob.read(magic, sizeof(magic));
    OPENVINO_ASSERT(blob.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
                        std::memcmp(magic, tp_blob::magic, sizeof(magic)) == 0,
                    "[TP_GPU] the stream does not hold a TP_GPU compiled blob");

    const auto version = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(version == tp_blob::version,
                    "[TP_GPU] compiled blob has format version ", version,
                    ", this runtime writes and reads version ", tp_blob::version);

    BlobHeader header;
    header.world_size = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(header.world_size >= 2 && header.world_size <= tp_blob::max_world_size,
                    "[TP_GPU] compiled blob declares ", header.world_size,
                    " ranks, which is outside the supported range [2, ", tp_blob::max_world_size, "]");

    header.num_collectives = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(header.num_collectives <= tp_blob::max_collectives,
                    "[TP_GPU] compiled blob declares ", header.num_collectives,
                    " collectives, which is not plausible; the cache entry is corrupted");

    const auto num_sharded_states = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(num_sharded_states <= tp_blob::max_sharded_states,
                    "[TP_GPU] compiled blob declares ", num_sharded_states,
                    " sharded states, which is not plausible; the cache entry is corrupted");

    header.device_names.resize(header.world_size);
    for (auto& name : header.device_names) {
        name = tp_blob::read_string(blob);
    }

    header.sharded_state_ids.resize(num_sharded_states);
    for (auto& id : header.sharded_state_ids) {
        id = tp_blob::read_string(blob, tp_blob::max_variable_id_length);
    }
    return header;
}
}  // namespace

Plugin::Plugin() {
    set_device_name("TP_GPU");
}

Plugin::SharedL0Setup Plugin::create_shared_l0(const std::vector<std::string>& device_names) const {
    // Cross-device L0 USM copies require every participating device to belong
    // to one ze_context_handle_t. Each rank's (driver, device) handles come
    // from the intel_gpu plugin's default RemoteContext; the context built over
    // them is handed back to each rank as a non-owning ContextType::ZE one.
    const auto rank_count = static_cast<uint32_t>(device_names.size());
    ZE_THROW(ov::zeInit(0));

    std::vector<ze_driver_handle_t> drivers(rank_count);
    std::vector<ze_device_handle_t> rank_devs(rank_count);

    for (uint32_t rank = 0; rank < rank_count; ++rank) {
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

    for (uint32_t rank = 1; rank < rank_count; ++rank) {
        OPENVINO_ASSERT(drivers[rank] == drivers[0],
                        "[TP_GPU] all ranks must share a single L0 driver, but ", device_names[rank],
                        " belongs to a different one than ", device_names[0]);
    }

    OPENVINO_ASSERT(ov::zeContextCreateEx != nullptr,
                    "[TP_GPU] L0 loader does not export zeContextCreateEx, which is required to "
                    "share one context across devices. Update the Level-Zero loader.");

    ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t shared_ctx_h = nullptr;
    ZE_THROW(ov::zeContextCreateEx(drivers[0], &ctx_desc,
                                   static_cast<uint32_t>(rank_devs.size()),
                                   rank_devs.data(), &shared_ctx_h));

    SharedL0Setup setup;
    setup.shared = std::make_shared<TPL0SharedContext>();
    setup.shared->driver = drivers[0];
    setup.shared->devices = rank_devs;
    setup.shared->context = shared_ctx_h;

    setup.rank_ctx.resize(rank_count);
    for (uint32_t rank = 0; rank < rank_count; ++rank) {
        ov::AnyMap rank_params{
            {ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::ZE},
            {ov::intel_gpu::ocl_context.name(), static_cast<ov::intel_gpu::gpu_handle_param>(shared_ctx_h)},
            {ov::intel_gpu::ze_device_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(rank_devs[rank])},
            {ov::intel_gpu::ze_driver_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(drivers[0])},
        };
        setup.rank_ctx[rank] = get_core()->create_context(device_names[rank], rank_params);
    }

    TP_LOG_INFO << "[TP] Created shared L0 context across " << rank_count << " devices (ctx="
                << shared_ctx_h << ")" << std::endl;

    return setup;
}

void Plugin::build_call_config(const ov::AnyMap& properties, TPConfig& config, ov::AnyMap& forwarded) const {
    // Whatever the plugin was configured with first, so the call can override
    // it rather than lose to it -- the old insert-into-a-map did the opposite
    // and silently kept the older value.
    config.set_user_property(m_config.user_properties(), ov::OptionVisibility::RELEASE);

    forwarded = m_gpu_config;
    ov::AnyMap own;
    for (const auto& entry : properties) {
        if (config.owns(entry.first)) {
            own.insert(entry);
        } else {
            forwarded[entry.first] = entry.second;
        }
    }
    config.set_user_property(own, ov::OptionVisibility::RELEASE);

    // Reads the environment and the config file, then locks the values down.
    config.finalize(nullptr, nullptr);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    TPConfig cfg;
    ov::AnyMap config;
    build_call_config(properties, cfg, config);

    auto device_names = cfg.resolve_device_names();
    // Weights are always divided by however many devices were asked for.
    // Under shard_only only the first rank is built and run, so the number of
    // ranks drops to one while the sharding degree does not.
    const auto shard_degree = static_cast<uint32_t>(device_names.size());
    const bool shard_only = cfg.shard_only();
    if (shard_only) {
        device_names.resize(1);
    }
    const auto rank_count = static_cast<uint32_t>(device_names.size());

    auto sharding_plan = GraphRewriter::analyze(model);
    const int num_collectives =
        shard_only ? 0
                   : GraphRewriter::count_collectives(sharding_plan, static_cast<int>(shard_degree), cfg);

    log_sharding_plan(sharding_plan, num_collectives);
    warn_on_kv_imbalance(sharding_plan, shard_degree);

    auto setup = create_shared_l0(device_names);
    auto coordinator = make_coordinator(setup.shared, rank_count, num_collectives,
                                        std::chrono::milliseconds{cfg.get_communication_timeout_ms()}, cfg);

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(rank_count);
    for (uint32_t rank = 0; rank < rank_count; ++rank) {
        auto rank_model = GraphRewriter::rewrite(model, sharding_plan, rank, shard_degree, cfg);

        TP_LOG_INFO << "[TP] Rank " << rank << ": " << rank_model->get_ordered_ops().size() << " ops, "
                    << num_collectives << " AllReduce points" << std::endl;

        if (shard_only) {
            strip_collectives(rank_model);
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
                                           GraphRewriter::sharded_state_ids(model, sharding_plan),
                                           /*loaded_from_cache=*/false,
                                           cfg);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    ov::AnyMap own;
    for (const auto& entry : properties) {
        if (m_config.owns(entry.first)) {
            own.insert(entry);
        } else {
            m_gpu_config[entry.first] = entry.second;
        }
    }
    // Validated here rather than at the next compile, and overwriting rather
    // than losing to whatever was set first.
    m_config.set_user_property(own, ov::OptionVisibility::RELEASE);
}

std::vector<std::string> Plugin::query_devices(const ov::AnyMap& arguments) const {
    TPConfig cfg;
    ov::AnyMap forwarded;
    build_call_config(arguments, cfg, forwarded);

    auto names = cfg.requested_device_names();
    if (!names.empty()) {
        return names;
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
        std::vector<ov::PropertyName> properties{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
        };

        const auto own = m_config.supported_properties();
        properties.insert(properties.end(), own.begin(), own.end());
        return properties;
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
        properties.emplace_back(ov::tp_gpu::tp_size.name(), ov::PropertyMutability::RW);
        properties.emplace_back(ov::tp_gpu::device_ids.name(), ov::PropertyMutability::RW);
        return properties;
    } else if (m_config.owns(name)) {
        // Unset options answer with their default rather than refusing: the
        // cache hash queries every caching property before any topology has
        // been chosen, and a throw there would disable caching outright.
        return m_config.get_property(name, ov::OptionVisibility::RELEASE);
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

    TPConfig cfg;
    ov::AnyMap config;
    build_call_config(properties, cfg, config);

    const auto device_names = cfg.resolve_device_names();

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
    TPConfig cfg;
    ov::AnyMap config;
    build_call_config(properties, cfg, config);

    // The TP container is what we are reading right now. Forwarding it would
    // make the GPU plugin try to import the outer blob instead of its own.
    config.erase(ov::hint::compiled_blob.name());

    auto header = read_blob_header(blob);

    // The blob is bound to the topology it was produced on: rank r's graph was
    // compiled for a specific device, and the collectives assume that exact
    // rank order. Silently retargeting would produce a model that runs and is
    // wrong.
    const auto requested_devices = cfg.requested_device_names();
    OPENVINO_ASSERT(requested_devices.empty() || requested_devices == header.device_names,
                    "[TP_GPU] compiled blob was produced for ",
                    ov::Any(header.device_names).as<std::string>(), " but ",
                    ov::Any(requested_devices).as<std::string>(), " was requested");

    auto setup = create_shared_l0(header.device_names);
    auto coordinator = make_coordinator(setup.shared, header.world_size,
                                        static_cast<int>(header.num_collectives),
                                        std::chrono::milliseconds{cfg.get_communication_timeout_ms()}, cfg);

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(header.world_size);
    for (uint32_t rank = 0; rank < header.world_size; ++rank) {
        const auto rank_blob_size = tp_blob::read_trivial<uint64_t>(blob);
        OPENVINO_ASSERT(rank_blob_size > 0, "[TP_GPU] compiled blob has an empty payload for rank ", rank);

        const auto rank_blob_start = blob.tellg();
        OPENVINO_ASSERT(rank_blob_start != std::istream::pos_type(-1),
                        "[TP_GPU] import_model needs a seekable stream to walk per-rank blobs");

        rank_compiled[rank] = get_core()->import_model(blob, setup.rank_ctx[rank], config);
        OPENVINO_ASSERT(rank_compiled[rank],
                        "[TP_GPU] the GPU plugin refused the cached blob of rank ", rank,
                        " on ", header.device_names[rank], "; it was compiled with a different cache mode");

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
                                           std::move(header.device_names),
                                           std::move(setup.shared),
                                           std::move(coordinator),
                                           std::move(header.sharded_state_ids),
                                           /*loaded_from_cache=*/true,
                                           cfg);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    return import_blob(model, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_THROW(kNoRemoteContextImport);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model, const ov::AnyMap& properties) const {
    MemoryInputBuffer buffer(static_cast<const char*>(model.data()), model.get_byte_size());
    std::istream stream(&buffer);
    return import_blob(stream, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_THROW(kNoRemoteContextImport);
}

}  // namespace tp_gpu
}  // namespace ov
