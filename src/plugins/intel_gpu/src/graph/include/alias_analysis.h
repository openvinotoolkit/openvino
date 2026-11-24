// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Phase 0.5a: the single engine that partitions output tensors into alias classes (views, in-place concat slots, oneDNN in-place
// writers) and reports, per class, the root allocator, the member kinds and why the class cannot be handled by a class-level
// allocation policy (fallback reasons) or must live outside the activation pool (domain reasons).
// Design: /home/ceciliapeng/repos/reports/memman/layer1/11_p1_design_r2.zh-CN.md section 2 (r3).
//
// This engine is a pure function of the graph. It never mutates nodes and has no influence on allocation or restrictions;
// every consumer (class-level restriction generation, shadow validation, intervals, war_safe, arenas) must go through it and
// must not re-implement owner / view / slot / writer detection.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cldnn {

struct program;
struct program_node;

enum class tensor_space : uint8_t { OUTPUT, INTERNAL };

// Input ports are not tensors. The uid is projected to uint32 (restriction sets are keyed by uint32).
struct tensor_id {
    uint32_t uid = 0;
    tensor_space space = tensor_space::OUTPUT;
    uint16_t index = 0;
    bool operator==(const tensor_id& o) const {
        return uid == o.uid && space == o.space && index == o.index;
    }
    bool operator!=(const tensor_id& o) const {
        return !(*this == o);
    }
};

enum class alias_kind : uint8_t { NONE, MUST_VIEW, MUST_INPLACE, MUST_SUBREGION, MAY_VIEW, MAY_SUBREGION, MAY_INPLACE };

// Offset of a member inside the buffer of its provider. Byte values are added later (not needed before Phase 0).
enum class offset_tag : uint8_t { ZERO, LAYOUT_PADDING, STATIC_BYTES, RUNTIME };

enum class alias_role : uint8_t { allocator, zero_alloc_member, may_allocate_member };

// Bit mask. A class with any fallback reason must keep the LEGACY (per-node) allocation behaviour.
enum fallback_reason : uint32_t {
    fb_oooq = 1u << 0,
    fb_cpu_impl = 1u << 1,
    fb_maybe_cpu = 1u << 2,
    fb_shape_of = 1u << 3,
    fb_body_ctrl_flow = 1u << 4,
    fb_internal_net = 1u << 5,
    fb_role_conflict = 1u << 6,       // multi allocator / opaque (lora) / member bytes above root bytes / pred that is itself a view
    fb_nonzero_port_root = 1u << 7,   // temporary until P1-M2
    fb_war_unsafe = 1u << 8,          // reserved for Phase 1a
    fb_war_unknown = 1u << 9,         // reserved for Phase 1a
    fb_unknown_size = 1u << 10,       // reserved for the dynamic route
};

// Bit mask. Domain reasons do not make the class unusable, they only say where its allocation lives.
enum domain_reason : uint32_t {
    dm_external_root = 1u << 0,
    dm_persistent = 1u << 1,   // kv / variable / read_value / assign
    dm_user_output = 1u << 2,
};

enum class placement : uint8_t { LEGACY, EXTERNAL, STATE, OUTPUT_DOMAIN, POOL /* Phase 4s: ARENA */ };

struct alias_member {
    tensor_id id;
    alias_kind kind = alias_kind::NONE;
    offset_tag off = offset_tag::ZERO;
    alias_role role = alias_role::allocator;
    const program_node* node = nullptr;
    // Phase 0.5b (additive): provider edge table. first_edge / edge_count index alias_analysis::edges() entries whose member is this
    // tensor (edge_count > 1 for tensors with several providers, e.g. a cbo kv_cache that is also a concat slot).
    int32_t first_edge = -1;
    uint8_t edge_count = 0;
    const program_node* provider_node = nullptr;   // provider of the first edge
    uint16_t provider_port = 0;
    bool opaque = false;                           // alias target not statically known
};

// Phase 0.5b: why a class is a role conflict (statistics only; fb_role_conflict is written at the same points as before).
enum role_conflict_cause : uint32_t {
    rc_allocators_not_one = 1,
    rc_opaque = 2,
    rc_member_exceeds_root = 4,
    rc_view_slot_pred = 8,
};

struct alias_class {
    tensor_id root;
    const program_node* root_node = nullptr;
    std::vector<alias_member> members;                 // includes the root; sorted by (debug PO rank, port)
    std::vector<const program_node*> readers;          // users of members that are not members themselves; shape_of users are skipped
    uint32_t fallback = 0;
    uint32_t domain = 0;
    uint32_t role_conflict_causes = 0;                 // role_conflict_cause bits, statistics only

    bool has_fallback() const {
        return fallback != 0;
    }
    placement get_placement() const {
        if (has_fallback()) {
            return placement::LEGACY;
        }
        if (domain & dm_external_root) {
            return placement::EXTERNAL;
        }
        if (domain & dm_persistent) {
            return placement::STATE;
        }
        if (domain & dm_user_output) {
            return placement::OUTPUT_DOMAIN;
        }
        return placement::POOL;
    }
};

enum class edge_activity { ALWAYS, NEVER, MAYBE };

// Which alias edges are in force. compile_time_envelope: the MAY envelope of the compiled graph (static edges ALWAYS,
// dynamic / runtime-skippable ones MAYBE). The per-round decisions of the dynamic route implement the same interface later.
// `m.kind` holds the MUST_* kind the structural analysis found; `m.node` is the member that owns the edge.
struct decision_source {
    virtual ~decision_source() = default;
    virtual edge_activity active(const alias_member& m) const = 0;
};

struct compile_time_envelope : decision_source {
    edge_activity active(const alias_member& m) const override;
};

// Phase 0.5b ---------------------------------------------------------------------------------------------------------------------
enum class edge_origin : uint8_t { concat_slot, view, onednn_sum, onednn_mul, kv_cache, stateless_kv, lora, hypothetical };

// One alias edge: the member tensor lives in (a region of) the buffer of the provider tensor.
struct alias_edge {
    const program_node* member_node = nullptr;
    uint16_t member_port = 0;
    const program_node* provider_node = nullptr;
    uint16_t provider_port = 0;
    alias_kind kind = alias_kind::NONE;                // after the envelope: MUST_* / MAY_*
    offset_tag off = offset_tag::ZERO;
    edge_activity act = edge_activity::ALWAYS;         // NEVER edges are not recorded
    int32_t dep_index = -1;                            // dependency index of the member through which it reads the provider; -1 for concat slots
    edge_origin origin = edge_origin::view;
};

enum class persistent_writer_kind : uint8_t { kv_cache_inplace, kv_cache_state_port, stateless_kv_past, assign_copy };

struct storage_key {
    enum class kind : uint8_t { variable, external_node } k = kind::variable;
    std::string variable_id;
    const program_node* node = nullptr;
    bool operator==(const storage_key& o) const {
        return k == o.k && variable_id == o.variable_id && node == o.node;
    }
};

struct persistent_writer {
    const program_node* writer = nullptr;
    uint16_t port = 0;
    persistent_writer_kind kind = persistent_writer_kind::kv_cache_inplace;
};

struct persistent_binding {
    storage_key key;
    std::vector<std::pair<const program_node*, uint16_t>> aliases;   // tensors that alias the storage (read_value output 0 of the variable)
    std::vector<persistent_writer> writers;
};

// A what-if in-place writer for tests and for Phase 1a (relaxing F10). Precondition (N13): the output layout of `writer` (padding
// included) equals the layout of the tensor it would overwrite; entries that do not satisfy it are not added (see rejected_hypotheticals()).
struct hypothetical_inplace {
    const program_node* writer = nullptr;
    int32_t dep_index = 0;
};

// Shared with the validation code (single implementation, N8). cpu_state: same rule the allocator uses.
inline constexpr int8_t cpu_none = 0, cpu_sure = 1, cpu_maybe = 2;
int8_t cpu_state(const program_node& n);
bool is_persistent_type(const program_node& n);

// kv_cache / stateless_kv: every kv_cache (cbo or not) and every stateless_kv has an alias edge to its input 0 (MAY_INPLACE, or the settled
// view edge of a cbo kv_cache). Source: user decision of 2026-11-20, errata of design 08 B.2 step 3; runtime reference:
// primitive_inst::do_runtime_in_place_kv_cache runs for every kv_cache regardless of can_be_optimized.
class alias_analysis {
public:
    // `extra` (Phase 0.5b) adds hypothetical MUST_INPLACE edges; with nullptr the result is bit-identical to the plain build.
    static alias_analysis build(const program& p, const decision_source& src, const std::vector<hypothetical_inplace>* extra = nullptr);
    static alias_analysis build(const program& p) {
        return build(p, compile_time_envelope{});
    }

    // ---- Phase 0.5b: provider edge table, persistent registry, tensor-level user enumeration ----
    const std::vector<alias_edge>& edges() const {
        return m_edges;
    }
    // Indices into edges(): edges whose provider / member is the tensor (n, port). Empty for unknown tensors.
    const std::vector<size_t>& edges_into(const program_node& provider, uint16_t port) const;
    const std::vector<size_t>& edges_from(const program_node& member, uint16_t port) const;
    bool tensor_is_opaque(const program_node& n, uint16_t port) const;
    const std::vector<persistent_binding>& persistent_bindings() const {
        return m_bindings;
    }
    // First binding that lists `writer` among its writers (nullptr if none).
    const persistent_binding* binding_written_by(const program_node& writer) const;
    const std::vector<const program_node*>& rejected_hypotheticals() const {
        return m_rejected_hypotheticals;
    }
    // Tensor-level user enumeration, shared by war_safe and P2 (the only implementation): fn(user, dep_index) is called once for every
    // dependency index of a user of the graph that reads (n, port). shape_of users are NOT skipped: the caller decides.
    void for_each_tensor_user(const program_node& n, uint16_t port, const std::function<void(const program_node&, size_t)>& fn) const;

    // port must be >= 0. Returns nullptr for nodes that are not part of the analysed graph. Looked up by node identity, so it is exact
    // even when unique ids are not unique (see duplicate_uid_count()).
    const alias_class* class_of(const program_node& n, int32_t port = 0) const;
    // Looked up by uid projection. A uid shared by several tensors is ambiguous: such keys return nullptr (and root_of returns the id
    // unchanged). Consumers (P1 generator, P2) must query by node identity whenever duplicate_uid_count() > 0.
    const alias_class* class_of(const tensor_id& id) const;
    tensor_id root_of(const tensor_id& id) const;
    const std::vector<alias_class>& classes() const {
        return m_classes;
    }
    // Number of tensors whose uid projection collided with an earlier tensor. Nodes created after compile_graph (weights reorders added by
    // post_optimize_weights) keep unique id 0 and collide with node 0; every uid-keyed consumer (restriction sets, memory_user) has to know.
    size_t duplicate_uid_count() const {
        return m_duplicate_uids;
    }
    // P1-B-S: uid level view of the same collision data (the private m_colliding_keys is keyed by (uid, space, index)). True if the uid
    // projection of any OUTPUT tensor of a node with this uid collided with an earlier tensor, i.e. the uid is shared by several nodes
    // (or several tensors of different identity). Class level generators must not promote a class that contains such a uid.
    bool uid_is_ambiguous(uint32_t uid) const;
    size_t ambiguous_uid_count() const;

private:
    // Orders members and classes deterministically. NOT a time axis: the lifetime axis is network::_exec_order (F9).
    size_t debug_po_rank(const program_node& n) const;

    static uint64_t key_of(const tensor_id& id) {
        return (static_cast<uint64_t>(id.uid) << 32) | (static_cast<uint64_t>(id.space) << 16) | id.index;
    }

    std::vector<alias_class> m_classes;
    std::unordered_map<uint64_t, size_t> m_class_of_tensor;                          // by uid projection (first wins)
    std::unordered_map<const program_node*, std::vector<size_t>> m_class_of_node;    // by node identity, indexed by port
    size_t m_duplicate_uids = 0;
    std::unordered_set<uint64_t> m_colliding_keys;
    std::unordered_map<const program_node*, size_t> m_po_rank;

    // Phase 0.5b
    size_t tensor_index(const program_node& n, uint16_t port) const;   // SIZE_MAX if unknown
    std::vector<alias_edge> m_edges;
    std::vector<std::vector<size_t>> m_edges_from;     // per tensor
    std::vector<std::vector<size_t>> m_edges_into;     // per tensor
    std::vector<char> m_opaque;                        // per tensor
    std::unordered_map<const program_node*, std::pair<size_t, size_t>> m_tensor_base;   // node -> (first tensor index, port count)
    std::vector<persistent_binding> m_bindings;
    std::vector<const program_node*> m_rejected_hypotheticals;
};

}  // namespace cldnn
