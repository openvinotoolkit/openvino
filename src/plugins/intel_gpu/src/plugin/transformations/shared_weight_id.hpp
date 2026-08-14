// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/shared_weight_id_attribute.hpp"

namespace ov::intel_gpu {

// Cross-CompiledModel device-weight sharing: working-form rt_info key.
//
// A model builder (today only DiffusionGemma's MoE / FC loaders) stamps weight
// Constants that are reused across separately-built graphs (encoder/decoder) with
// a stable, host-pointer-independent id, so the GPU plugin can keep a single
// device buffer per logical weight. The plugin reads this plain-string rt_info
// key in create_data (ops/constant.cpp). It is a strict no-op for every model
// that never sets it. The companion "shared_weight_stable" flag is written by the
// model builder and read straight from create_data; the compression passes here
// mostly only need to move the id, but a pass that rebuilds a weight from several
// stamped constants (FullyConnectedHorizontalFusion) must also be able to clear
// the stable flag off the rebuilt weight, so both keys are exposed.
//
// SHARED_WEIGHT_ID_KEY / SHARED_WEIGHT_STABLE_KEY must match the literals read in
// ops/constant.cpp.
inline constexpr const char* SHARED_WEIGHT_ID_KEY = "shared_weight_id";
inline constexpr const char* SHARED_WEIGHT_STABLE_KEY = "shared_weight_stable";

// Carry the cross-graph sharing id from one weight node onto a node rebuilt from
// it. Several FC compression passes recreate the weight as a brand-new Constant
// whose rt_info starts empty (the copy-ctor does not copy rt_info), so the id
// must be moved explicitly — same pattern as ov::copy_weightless_cache_attr.
// No-op when `from` carries no id (i.e. every model except DiffusionGemma, and
// scale/zp constants which are never stamped).
//
// The SHARED_WEIGHT_STABLE flag is deliberately NOT carried over: the sole caller
// runs at a repack boundary (the `to` constant was just produced by a reshape),
// which invalidates the "no repack before create_data" precondition that lets the
// consumer skip the content hash for stable weights. A repacked constant may have
// graph-divergent bytes, so it must stay non-stable and re-hash — dropping the
// flag here keeps the content-addressed safety net engaged by construction.
inline void copy_shared_weight_id(const std::shared_ptr<ov::Node>& from, const std::shared_ptr<ov::Node>& to) {
    if (!from || !to)
        return;
    const auto& from_rt = from->get_rt_info();
    auto id_it = from_rt.find(SHARED_WEIGHT_ID_KEY);
    if (id_it == from_rt.end())
        return;
    to->get_rt_info()[SHARED_WEIGHT_ID_KEY] = id_it->second;
}

// Walk back from a weight input to the underlying weight Constant, skipping the
// shape/layout-only wrappers a compression pass may have inserted in front of it
// (Convert for dtype, Reshape / Transpose for layout). Returns nullptr if the
// chain does not bottom out in a Constant — the caller then treats the weight as
// un-stamped and falls back to the non-shared path.
inline std::shared_ptr<ov::op::v0::Constant> find_weight_constant(const std::shared_ptr<ov::Node>& node) {
    auto cur = node;
    while (cur) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(cur))
            return constant;
        if (!ov::is_type<ov::op::v0::Convert>(cur) && !ov::is_type<ov::op::v1::Reshape>(cur) && !ov::is_type<ov::op::v1::Transpose>(cur))
            break;
        if (cur->get_input_size() == 0)
            break;
        cur = cur->get_input_node_shared_ptr(0);
    }
    return nullptr;
}

// Restore the GPU working-form plain-string rt_info keys (SHARED_WEIGHT_ID_KEY /
// SHARED_WEIGHT_STABLE_KEY) from the serializable ov::SharedWeightId attribute.
//
// ov::serialize drops plain-string rt_info but preserves is_deterministic
// RuntimeAttributes, so a model round-tripped through IR (e.g. an on-disk model
// cache) comes back carrying ONLY the attribute mirror, not the working keys that
// create_data (ops/constant.cpp) and the FC compression passes actually read.
// Re-materialize the plain-string form here so those consumers behave identically
// on a cached model and on a freshly built one — without this, a cached model
// silently loses its cross-graph sharing ids and each graph uploads a private copy
// of every shared weight.
//
// Run this once, before the compression passes, at the start of the GPU
// transformation pipeline. It is a strict no-op for a freshly built model (which
// still carries the plain-string keys — emplace does not overwrite them) and for
// every model that never sets the id. Idempotent.
inline void restore_shared_weight_ids(const std::shared_ptr<ov::Model>& model) {
    if (!model)
        return;
    for (const auto& node : model->get_ops()) {
        auto& rt = node->get_rt_info();
        auto it = rt.find(ov::SharedWeightId::get_type_info_static());
        // Guard the deserialization boundary: the value under this key is normally
        // an ov::SharedWeightId, but a stale/hand-edited IR cache could carry
        // something else. Skip rather than let ov::Any::as<> throw and abort compile.
        if (it == rt.end() || !it->second.is<ov::SharedWeightId>())
            continue;
        const auto& swid = it->second.as<ov::SharedWeightId>();
        if (swid.id.empty())
            continue;
        rt.emplace(SHARED_WEIGHT_ID_KEY, swid.id);
        if (swid.stable)
            rt.emplace(SHARED_WEIGHT_STABLE_KEY, std::string("1"));
    }
}

}  // namespace ov::intel_gpu
