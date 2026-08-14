// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines the serializable cross-graph shared-weight id attribute
 * @file shared_weight_id_attribute.hpp
 */

#pragma once

#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

/**
 * @ingroup ov_runtime_attr_api
 * @brief Serializable carrier for the GPU cross-CompiledModel device-weight
 * sharing id.
 *
 * A model builder marks weight Constants that are reused across separately-built
 * graphs (e.g. DiffusionGemma's encoder/decoder MoE experts and dense FC weights)
 * so the GPU plugin keeps a single device buffer per logical weight. The working
 * representation the GPU plugin reads is the plain-string rt_info
 * "shared_weight_id" (+ "shared_weight_stable"), but plain-string rt_info is
 * dropped by ov::serialize. This RuntimeAttribute mirrors that information in a
 * form the IR (de)serializer preserves, so a model round-tripped through an IR
 * cache keeps its sharing ids; the consumer restores the plain-string form from
 * this attribute after read_model.
 *
 * The attribute is a strict no-op for models that never set it (i.e. everything
 * except DiffusionGemma today).
 */
class TRANSFORMATIONS_API SharedWeightId : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("SharedWeightId", "0", ov::RuntimeAttribute);

    SharedWeightId() = default;
    SharedWeightId(std::string id, bool stable) : id(std::move(id)), stable(stable) {}

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("id", id);
        visitor.on_attribute("stable", stable);
        return true;
    }

    std::string id;
    bool stable = false;
};

}  // namespace ov
