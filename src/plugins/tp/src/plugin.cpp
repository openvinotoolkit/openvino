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
#include "ov_ops/tp_collective.hpp"
#include "tp/tp_config.hpp"
#include "tp/tp_debug.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/paged_selective_ssm.hpp"

namespace ov {
namespace tp {

namespace {
void expect_one_stream(const ov::AnyMap& config) {
    const auto it = config.find(ov::num_streams.name());
    if (it == config.end()) {
        return;
    }
    const auto value = it->second.as<std::string>();
    OPENVINO_ASSERT(ov::Any(value).as<ov::streams::Num>() == 1,
                    "TP supports one stream per device (ov::num_streams = 1), got ",
                    value,
                    ".");
}

/// A collective call is matched across the ranks by the channel of the rank
/// model's stream, and the stream that runs a request is picked per request:
/// with more than one, the ranks of one TP request could meet on different
/// channels.
void force_one_stream(ov::AnyMap& forwarded) {
    expect_one_stream(forwarded);
    if (const auto it = forwarded.find(ov::device::properties.name()); it != forwarded.end()) {
        for (const auto& device : it->second.as<ov::AnyMap>()) {
            expect_one_stream(device.second.as<ov::AnyMap>());
        }
    }
    forwarded[ov::num_streams.name()] = ov::streams::Num{1};
}

void reject_linear_attention(const ov::Model& model) {
    constexpr const char* unsupported = "TP does not support models with linear-attention state yet.";
    for (const auto& op : model.get_ops()) {
        const bool linear_attention = ov::is_type_any_of<ov::op::internal::PagedGatedDeltaNet,
                                                         ov::op::internal::PagedCausalConv1D,
                                                         ov::op::internal::PagedSelectiveSSM>(op);
        OPENVINO_ASSERT(!linear_attention,
                        unsupported,
                        " Found ",
                        op->get_type_name(),
                        " '",
                        op->get_friendly_name(),
                        "'.");
    }
    for (const auto& input : model.inputs()) {
        for (const auto& name : input.get_names()) {
            OPENVINO_ASSERT(name.find("_state_table.") == std::string::npos,
                            unsupported,
                            " Found the state input '",
                            name,
                            "'.");
        }
    }
}

/// Diagnostic (TP_SHARD_ONLY=1): strip every collective from a shard so it can
/// run alone on one device -- the upper bound tensor parallelism is chasing, with
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
        if (!ov::is_type<ov::op::internal::TPCollectiveBase>(node)) {
            continue;
        }
        OPENVINO_ASSERT(node->get_input_size() == 1 && node->get_output_size() == 1,
                        "[TP] TP_SHARD_ONLY cannot handle ", node->get_type_info().name,
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
                        "[TP] TP_SHARD_ONLY cannot widen ", node->get_type_info().name,
                        ": ranks differ or are dynamic.");
        int64_t axis = -1;
        int64_t factor = 0;
        for (int64_t i = 0; i < in_shape.rank().get_length(); ++i) {
            if (in_shape[i] == out_shape[i]) {
                continue;
            }
            OPENVINO_ASSERT(axis < 0 && in_shape[i].is_static() && out_shape[i].is_static() &&
                                out_shape[i].get_length() % in_shape[i].get_length() == 0,
                            "[TP] TP_SHARD_ONLY cannot widen ", node->get_type_info().name,
                            ": expected exactly one axis to grow by a whole factor, got ",
                            in_shape, " -> ", out_shape);
            axis = i;
            factor = out_shape[i].get_length() / in_shape[i].get_length();
        }
        OPENVINO_ASSERT(axis >= 0, "[TP] TP_SHARD_ONLY: ", node->get_type_info().name,
                        " changes shape but no axis grew: ", in_shape, " -> ", out_shape);

        ov::OutputVector copies(static_cast<std::size_t>(factor), node->input_value(0));
        auto concat = std::make_shared<ov::op::v0::Concat>(copies, axis);
        concat->set_friendly_name(node->get_friendly_name());
        ov::replace_node(node, concat);
        ++widened;
    }
    shard->validate_nodes_and_infer_types();

    TP_WARN_ALWAYS << "[TP] " << ov::tp::shard_only.name() << ": " << dropped
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
    TP_WARN_ALWAYS << "[TP] Warning: " << plan.num_kv_heads << " KV heads do not divide evenly across "
                   << shard_degree << " ranks (" << remainder << " rank(s) get " << (base + 1) << ", the rest "
                   << base << "), about " << imbalance_pct << "% load imbalance.";
}

constexpr const char* kNoRemoteContextImport =
    "[TP] importing into a caller-provided remote context is not supported: the plugin "
    "creates the rank contexts of its device group itself";

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
                    "[TP] the stream does not hold a TP compiled blob");

    const auto version = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(version == tp_blob::version,
                    "[TP] compiled blob has format version ", version,
                    ", this runtime writes and reads version ", tp_blob::version);

    BlobHeader header;
    header.world_size = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(header.world_size >= 2 && header.world_size <= tp_blob::max_world_size,
                    "[TP] compiled blob declares ", header.world_size,
                    " ranks, which is outside the supported range [2, ", tp_blob::max_world_size, "]");

    header.num_collectives = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(header.num_collectives <= tp_blob::max_collectives,
                    "[TP] compiled blob declares ", header.num_collectives,
                    " collectives, which is not plausible; the cache entry is corrupted");

    const auto num_sharded_states = tp_blob::read_trivial<uint32_t>(blob);
    OPENVINO_ASSERT(num_sharded_states <= tp_blob::max_sharded_states,
                    "[TP] compiled blob declares ", num_sharded_states,
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
    set_device_name("TP");
}

std::vector<ov::SoPtr<ov::IRemoteContext>> Plugin::create_rank_contexts(const std::vector<std::string>& device_names,
                                                                        const TPConfig& config) const {
    // The contexts are created one after another: creating one only publishes
    // the rank's setup payload, and the device plugin collects them all when
    // the first rank model is bound, by which time every rank has published.
    const auto rank_count = static_cast<int>(device_names.size());
    const auto timeout_ms = config.get_communication_timeout_ms();
    auto endpoints = InProcessExchange::create_group(rank_count, std::chrono::milliseconds{timeout_ms});

    std::vector<ov::SoPtr<ov::IRemoteContext>> contexts(device_names.size());
    for (int rank = 0; rank < rank_count; ++rank) {
        ov::AnyMap params;
        params[ov::tp::internal::device_group.name()] = device_names;
        params[ov::tp::internal::groups.name()] =
            std::map<int, CollectiveGroup>{{0, CollectiveGroup{endpoints[static_cast<size_t>(rank)], config.get_collective_backend()}}};
        params[ov::tp::internal::communication_timeout_ms.name()] = timeout_ms;
        if (const auto comm_buffer_size = config.get_comm_buffer_size(); comm_buffer_size != 0) {
            params[ov::tp::internal::comm_buffer_size.name()] = comm_buffer_size;
        }
        contexts[static_cast<size_t>(rank)] = get_core()->create_context(device_names[static_cast<size_t>(rank)], params);
    }

    TP_LOG_INFO << "[TP] Created " << rank_count << " rank contexts" << std::endl;
    return contexts;
}

void Plugin::build_call_config(const ov::AnyMap& properties, TPConfig& config, ov::AnyMap& forwarded) const {
    // Whatever the plugin was configured with first, so the call can override
    // it rather than lose to it -- the old insert-into-a-map did the opposite
    // and silently kept the older value.
    config.set_user_property(m_config.user_properties(), ov::OptionVisibility::RELEASE);

    forwarded = m_device_config;
    ov::AnyMap own;
    for (const auto& entry : properties) {
        if (config.owns(entry.first)) {
            own.insert(entry);
        } else {
            forwarded[entry.first] = entry.second;
        }
    }
    config.set_user_property(own, ov::OptionVisibility::RELEASE);

    // The TP blob contains the rank blobs, and the core caches it as a whole;
    // a rank compilation must not add a cache entry of its own.
    forwarded[ov::cache_dir.name()] = std::string{};
    force_one_stream(forwarded);

    // Reads the environment and the config file, then locks the values down.
    config.finalize(nullptr, nullptr);
}

void Plugin::check_collective_support(const std::vector<std::string>& device_names, const TPConfig& config) const {
    const auto type = TPConfig::device_type(device_names);
    const auto internal = get_core()->get_property(type, ov::internal::supported_properties);
    OPENVINO_ASSERT(ov::util::contains(internal, ov::tp::internal::collective_support.name()),
                    get_device_name(),
                    ": the ",
                    type,
                    " plugin does not support tensor-parallel collectives");

    // The device plugin decides what it can serve, distinct devices included.
    const auto reason = get_core()
                            ->get_property(type,
                                           ov::tp::internal::collective_support.name(),
                                           {{ov::tp::internal::device_group.name(), device_names},
                                            {ov::tp::internal::collective_backend.name(), config.get_collective_backend()}})
                            .as<std::string>();
    OPENVINO_ASSERT(reason.empty(), get_device_name(), ": ", reason);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    TPConfig cfg;
    ov::AnyMap config;
    build_call_config(properties, cfg, config);

    auto device_names = cfg.resolve_device_names();
    reject_linear_attention(*model);
    const bool shard_only = cfg.shard_only();
    if (!shard_only) {
        check_collective_support(device_names, cfg);
    }
    // Weights are always divided by however many devices were asked for.
    // Under shard_only only the first rank is built and run, so the number of
    // ranks drops to one while the sharding degree does not.
    const auto shard_degree = static_cast<uint32_t>(device_names.size());
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

    // A shard stripped of its collectives runs alone, on an ordinary context.
    const auto rank_contexts =
        shard_only ? std::vector<ov::SoPtr<ov::IRemoteContext>>{} : create_rank_contexts(device_names, cfg);

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(rank_count);
    for (uint32_t rank = 0; rank < rank_count; ++rank) {
        auto rank_model = GraphRewriter::rewrite(model, sharding_plan, rank, shard_degree, cfg);

        TP_LOG_INFO << "[TP] Rank " << rank << ": " << rank_model->get_ordered_ops().size() << " ops, "
                    << num_collectives << " AllReduce points" << std::endl;

        if (shard_only) {
            strip_collectives(rank_model);
            rank_compiled[rank] = get_core()->compile_model(rank_model, device_names[rank], config);
        } else {
            rank_compiled[rank] = get_core()->compile_model(rank_model, rank_contexts[rank], config);
        }
    }

    return std::make_shared<CompiledModel>(model, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(device_names),
                                           static_cast<uint32_t>(num_collectives),
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
            m_device_config[entry.first] = entry.second;
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

    return cfg.requested_device_names();
}

std::vector<ov::PropertyName> Plugin::rank_caching_properties(const ov::AnyMap& arguments) const {
    // Without rank devices there is no device plugin to ask.
    const auto devices = query_devices(arguments);
    if (devices.empty()) {
        return {};
    }
    return get_core()->get_property(devices.front(), ov::internal::caching_properties);
}

ov::Any Plugin::aggregate_rank_property(const std::string& name, const ov::AnyMap& arguments) const {
    // A TP blob is a container of one device-plugin blob per rank, so anything
    // that makes a single rank's blob unique makes the whole thing unique.
    // Values are joined across ranks so a heterogeneous set of devices cannot
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
        return std::string("TP");
    } else if (name == ov::device::capabilities.name()) {
        return std::vector<std::string>{ov::device::capability::EXPORT_IMPORT};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
        };
    } else if (name == ov::internal::caching_properties.name()) {
        auto properties = rank_caching_properties(arguments);
        properties.emplace_back(ov::device::priorities.name(), ov::PropertyMutability::RW);
        properties.emplace_back(ov::tp::tp_strategy.name(), ov::PropertyMutability::RW);
        return properties;
    } else if (m_config.owns(name)) {
        // Unset options answer with their default rather than refusing: the
        // cache hash queries every caching property before any topology has
        // been chosen, and a throw there would disable caching outright.
        return m_config.get_property(name, ov::OptionVisibility::RELEASE);
    }

    // Everything the device plugin folds into its cache hash has to be answerable
    // here too, otherwise Core cannot build a compile config for TP.
    if (ov::util::contains(rank_caching_properties(arguments), name)) {
        return aggregate_rank_property(name, arguments);
    }
    OPENVINO_THROW("[TP] Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model != nullptr, "[TP] query_model: model is null");

    TPConfig cfg;
    ov::AnyMap config;
    build_call_config(properties, cfg, config);

    const auto device_names = cfg.resolve_device_names();

    // Every rank compiles the same op set, so whatever the device plugin supports
    // on one rank is what TP supports as a whole. The collectives inserted
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
    // make the device plugin try to import the outer blob instead of its own.
    config.erase(ov::hint::compiled_blob.name());

    auto header = read_blob_header(blob);

    // The blob is bound to the topology it was produced on: rank r's graph was
    // compiled for a specific device, and the collectives assume that exact
    // rank order. Silently retargeting would produce a model that runs and is
    // wrong.
    const auto requested_devices = cfg.requested_device_names();
    OPENVINO_ASSERT(requested_devices.empty() || requested_devices == header.device_names,
                    "[TP] compiled blob was produced for ",
                    ov::Any(header.device_names).as<std::string>(), " but ",
                    ov::Any(requested_devices).as<std::string>(), " was requested");
    check_collective_support(header.device_names, cfg);

    const auto rank_contexts = create_rank_contexts(header.device_names, cfg);

    std::vector<ov::SoPtr<ov::ICompiledModel>> rank_compiled(header.world_size);
    for (uint32_t rank = 0; rank < header.world_size; ++rank) {
        const auto rank_blob_size = tp_blob::read_trivial<uint64_t>(blob);
        OPENVINO_ASSERT(rank_blob_size > 0, "[TP] compiled blob has an empty payload for rank ", rank);

        const auto rank_blob_start = blob.tellg();
        OPENVINO_ASSERT(rank_blob_start != std::istream::pos_type(-1),
                        "[TP] import_model needs a seekable stream to walk per-rank blobs");

        rank_compiled[rank] = get_core()->import_model(blob, rank_contexts[rank], config);
        OPENVINO_ASSERT(rank_compiled[rank],
                        "[TP] the device plugin refused the cached blob of rank ", rank,
                        " on ", header.device_names[rank], "; it was compiled with a different cache mode");

        // The rank reader stops wherever its own format ends, which need not be
        // the byte we recorded. Realign so the next rank starts in the right
        // place instead of parsing the tail of its predecessor.
        blob.clear();
        blob.seekg(rank_blob_start + static_cast<std::istream::off_type>(rank_blob_size));
        OPENVINO_ASSERT(blob.good(), "[TP] truncated compiled blob: rank ", rank, " payload is incomplete");
    }

    // No IR here: the user model was consumed at compile time and is not part
    // of the blob. CompiledModel derives its ports from rank 0 instead.
    return std::make_shared<CompiledModel>(nullptr, shared_from_this(),
                                           std::move(rank_compiled),
                                           std::move(header.device_names),
                                           header.num_collectives,
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

}  // namespace tp
}  // namespace ov
