// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "openvino/core/model.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::frontend::onnx::weightless {

/// \brief Version of the synthetic ID scheme (design: name-based ID, FNV-1a 64 + fmix64, 60-bit hash).
/// Stored in bits 62-60 of every synthetic ID and in the model rt_info. Any change to the ID definition needs a new
/// scheme number.
constexpr uint64_t kIdScheme = 1;

/// \brief Model rt_info key holding the ov::WeightlessConstants value the frontend applied.
constexpr const char* kRtInfoWeightlessConstants = "weightless_constants";
/// \brief Model rt_info key holding kIdScheme, written when synthetic IDs are enabled.
constexpr const char* kRtInfoIdScheme = "weightless_id_scheme";

/// \brief Marks a Constant created from an ONNX initializer or Constant op. Not copyable: Constants derived from it in
/// translators or transformations must not inherit it. Removed from every node before conversion returns.
class OnnxInitializerId : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("OnnxInitializerId", "0", ov::RuntimeAttribute);

    OnnxInitializerId(std::string value_name, SourceModelLocation value_location, bool value_backed)
        : name(std::move(value_name)),
          location(value_location),
          backed(value_backed) {}

    bool is_copyable() const override {
        return false;
    }

    std::string name;              ///< Initializer name, or the output name of a Constant op
    SourceModelLocation location;  ///< Where the source model holds the data, as reported by the producer
    bool backed;                   ///< True if it may get a synthetic ID (the source model provides its data)
};

/// \brief Returns true if bin_offset is a synthetic ID (bit 63 set) rather than a real file offset.
constexpr bool is_synthetic_id(uint64_t bin_offset) {
    return (bin_offset >> 63) != 0;
}

/// \brief Returns the scheme version stored in a synthetic ID.
constexpr uint64_t get_id_scheme(uint64_t synthetic_id) {
    return (synthetic_id >> 60) & 0x7;
}

/// \brief Computes the scheme-1 synthetic ID of an initializer from its name, byte size and element type.
uint64_t make_synthetic_id(std::string_view name, uint64_t byte_size, const ov::element::Type& element_type);

/// \brief Applies the provenance rules to a Constant created from an initializer of a GraphIterator model, and attaches
/// an OnnxInitializerId to it.
/// \param constant           The Constant created from the initializer, possibly with a file-backed
///                           WeightlessCacheAttribute.
/// \param name               The initializer name.
/// \param location           The producer's answer, SourceModelLocation::UNKNOWN without a TensorOriginProvider.
/// \param declared           The declared external data location, for SourceModelLocation::EXTERNAL.
/// \param in_memory_marker   True if the data is referenced through an ORT in-memory address marker.
void mark_initializer(const std::shared_ptr<ov::op::v0::Constant>& constant,
                      const std::string& name,
                      SourceModelLocation location,
                      const std::optional<SourceModelExternalData>& declared,
                      bool in_memory_marker);

/// \brief Gives backed initializers synthetic IDs when mode is WeightlessConstants::ALL, records the applied mode in
/// the model rt_info, and logs a summary. Called before FrontEnd::normalize().
/// \param has_origin_provider True if the producer reported provenance, so the index-based fallback of
///                            enumerate_constants() must not be used.
void stamp_weightless_constants(const std::shared_ptr<ov::Model>& model,
                                ov::WeightlessConstants mode,
                                bool has_origin_provider);

/// \brief Returns true if the model rt_info records the weightless constants mode, which disables the index-based
/// fallback of enumerate_constants().
bool has_weightless_constants_mode(const std::shared_ptr<ov::Model>& model);

/// \brief Removes OnnxInitializerId from every node of the model and its subgraphs.
void remove_initializer_ids(const std::shared_ptr<ov::Model>& model);

}  // namespace ov::frontend::onnx::weightless
