// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "partitioning/patterns/pre_compute.hpp"

namespace ov {
namespace npuw {
namespace longrope {

using PortsMap = std::unordered_map<std::string, ov::Output<const ov::Node>>;

// The rotation that carries a key already rotated under one LongRoPE regime to the one
// it would have had under the other. Laid out as num_tokens rows of rotary_ndims/2
// planes, row r belonging to the key cached at row r.
struct RegimeDelta {
    std::vector<float> cos;
    std::vector<float> sin;
    size_t half = 0;  // planes per row; 0 means "nothing to do"
};

// Builds the delta for cached rows [0, num_tokens), whose absolute positions start at
// first_position_id.
//
// The rotation is the complex quotient of the two regimes' coefficients:
//
//   k' = k * (cos_new + i*sin_new) / (cos_old + i*sin_old)
//
// The quotient - rather than the plain difference of the two angles - is what inverts
// the f16 table values the graph actually used, whose magnitude is only approximately
// one.
//
// Returns an empty delta (half == 0) when the two regimes share their coefficients.
RegimeDelta make_regime_delta(patterns::pre_compute::LongRopeCosSin& tables,
                              int64_t first_position_id,
                              uint32_t num_tokens,
                              bool to_long);

// One past-key tensor resolved down to what the rewrite actually walks: a host pointer
// plus the plane/row geometry derived from its shape. Produced by check_key_tensor(),
// which is the only place that may reject a tensor - once a layout exists the rewrite
// is pure arithmetic and cannot fail.
struct KeyTensorLayout {
    void* data = nullptr;
    ov::element::Type type;
    size_t outer = 0;           // planes before the sequence axis
    size_t seq_len = 0;         // entries along the sequence axis
    size_t seq_stride = 0;      // elements per sequence entry
    size_t rows_per_token = 0;  // head rows inside one sequence entry
    size_t head_dim = 0;
};

// Resolves one past-key tensor and throws unless the delta can be applied to it in
// place. Checks the element type, host accessibility, that the tensor is long enough,
// and that it is fully densely packed - rerotate_keys() addresses rows by plain
// pointer arithmetic, which a strided or padded tensor would silently send to the
// wrong offsets.
KeyTensorLayout check_key_tensor(const ov::SoPtr<ov::ITensor>& tensor,
                                 uint32_t seq_dim,
                                 uint32_t num_tokens,
                                 const RegimeDelta& delta);

// Applies the delta in place to the leading num_tokens sequence entries of one densely
// packed past-key tensor, turning the rotate_half pair (j, j + half) of every head.
// Channels beyond 2 * delta.half are pass-through in a partial-rotary model and were
// never rotated, so they are left alone.
void rerotate_keys(const KeyTensorLayout& layout, uint32_t num_tokens, const RegimeDelta& delta);

// check_key_tensor() + rerotate_keys() for a single tensor.
void rerotate_keys(const ov::SoPtr<ov::ITensor>& tensor,
                   uint32_t seq_dim,
                   uint32_t num_tokens,
                   const RegimeDelta& delta);

// Rewrites every past-key input of the request into the requested regime.
//
// A LongRoPE model rotates with the short- or the long-factor coefficients depending on
// the current maximum position, so a cache filled under one regime stops matching the
// queries the moment that choice flips mid-conversation. Instead of dropping the cache,
// its keys are turned by the difference between the two regimes - see make_regime_delta.
// Values carry no rotation and are left alone.
//
// All or nothing: every past-key port is resolved and checked before the first byte is
// written, and anything that cannot be rewritten throws. A cache turned only in part
// would leave the model comparing queries and keys in two different rotation frames -
// exactly the corruption this path exists to prevent - so there is no useful weaker
// outcome to report.
void rerotate_cached_keys(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                          const PortsMap& in_ports,
                          const std::vector<std::string>& past_kv_names,
                          patterns::pre_compute::LongRopeCosSin& tables,
                          uint32_t seq_dim,
                          uint32_t num_tokens,
                          int64_t first_position_id,
                          bool to_long);

}  // namespace longrope
}  // namespace npuw
}  // namespace ov
