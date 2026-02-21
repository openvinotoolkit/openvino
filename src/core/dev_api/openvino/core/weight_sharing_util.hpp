// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>
#include <variant>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
class MappedMemory;
class AlignedBuffer;
class Model;

namespace op::v0 {
class Constant;
}
}  // namespace ov

namespace ov::weight_sharing {
using DataID = uint64_t;

/** @brief Defined INVALID ID for weight source */
inline constexpr auto INVALID_SOURCE_ID = std::numeric_limits<DataID>::max();
/** @brief Defined INVALID ID for constant */
inline constexpr auto INVALID_CONSTANT_ID = INVALID_SOURCE_ID;

/** @brief Metadata for a constant */
struct ConstantMetaData {
    size_t m_offset;
    size_t m_size;
    ov::element::Type m_type;
};

/** @brief Metadata for the origin of a constant */
struct ConstantOriginMetaData {
    DataID m_id;
    size_t m_offset;
    size_t m_size;
    ov::element::Type m_type;
};

/** @brief Variant type for weight buffer shared pointer*/
using WeightBuffer = std::variant<std::shared_ptr<ov::AlignedBuffer>, std::shared_ptr<ov::MappedMemory>>;
/** @brief Variant type for weight buffer observer*/
using WeakWeightBuffer = std::variant<std::weak_ptr<ov::AlignedBuffer>, std::weak_ptr<ov::MappedMemory>>;

/** @brief Map [key: Constant ID, value: ConstantMetaData] of constant meta data for single container. */
using ConstantMap = std::unordered_map<DataID, ConstantMetaData>;
/** @brief Map [key: Source ID, value: ConstantMap] of constant metadata for weight sources. */
using SourceConstantMap = std::unordered_map<DataID, ConstantMap>;
/** @brief Map [key: Source ID, value: WeakWeightBuffer] of pointers to weight sources. */
using WeightSourceMap = std::unordered_map<DataID, WeakWeightBuffer>;
/** @brief Map [key: Blob ID, value: blob name] of blobs to model name/tag */
using BlobMap = std::unordered_map<DataID, std::string>;

/** @brief Shared context for weight and constant management */
struct Context {
    // keep as standard layout as will be used between libraries
    SourceConstantMap m_constants_meta_data;
    WeightSourceMap m_weight_sources;
};

/** @brief Extension iface for classes which manage shared context */
struct OPENVINO_API Extension {
    /** @brief Get the constant source id for constant node.
     *
     * @param constant Constant node to get source id for.
     * @return Return  Id or INVALID_CONSTANT_ID if not found.
     */
    static DataID get_constant_source_id(const ov::op::v0::Constant& constant);

    /** @brief Get the constant id for constant node.
     *
     * @param constant Constant node to get id for.
     * @return Return Id or INVALID_CONSTANT_ID if not found.
     */
    static DataID get_constant_id(const ov::op::v0::Constant& constant);

    /** @brief Get the constant origin metadata for constant node.
     *
     * @param constant Constant node to get origin metadata for.
     * @return Return optional with ConstantOriginMetaData if found, std::nullopt otherwise.
     */
    static std::optional<ConstantOriginMetaData> get_constant_origin(const ov::op::v0::Constant& constant);

    /**
     * @brief Set constant metadata in constant map for given constant node.
     *
     * @param constant_map Constant source map to set metadata in.
     * @param constant Constant node to set metadata for.
     * @return Return true if metadata was set successfully, false otherwise.
     */
    static bool set_constant_in_constant_map(SourceConstantMap& constant_map, const ov::op::v0::Constant& constant);

    /** @brief Create the weight sources for a given model.
     *
     * @param model Model to get weight sources for.
     * @return Return map of weight sources.
     */
    static WeightSourceMap make_weight_sources(const Model& model);

    /** @brief Creates the nested map where Source ID and Data ID identifies to constant meta data.
     *
     * @param model Model to get constant map for.
     * @return Return map of constants.
     */
    static SourceConstantMap make_constant_map(const Model& model);
};

/** @brief Get the source buffer for a given source id.
 *
 * @param shared_context Shared context to get source buffer from.
 * @param source_id Source id to get buffer for.
 * @return Return shared pointer to AlignedBuffer if found, nullptr otherwise.
 */
OPENVINO_API std::shared_ptr<ov::AlignedBuffer> get_source_buffer(const Context& shared_context,
                                                                  const DataID source_id);

/** @brief Get the buffer for a given source id and constant id.
 *
 * The returned buffer is ready to use for Constant node creation.
 *
 * @param shared_context Shared context to get buffer from.
 * @param source_id Source id to get buffer for.
 * @param constant_id Constant id to get buffer for.
 * @return Return shared pointer to AlignedBuffer if found, nullptr otherwise.
 */
OPENVINO_API std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                                           const DataID source_id,
                                                           const DataID constant_id);

/** @brief Get the buffer for a given weight buffer (provide source id as hint) and constant id.
 *
 * The returned buffer is ready to use for Constant node creation.
 *
 * @param shared_context Shared context to get buffer from.
 * @param weight_buffer Weight buffer to get buffer from.
 * @param constant_id Constant id to get buffer for.
 * @return Return shared pointer to AlignedBuffer if found, nullptr otherwise.
 */
OPENVINO_API std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                                           const WeightBuffer& weight_buffer,
                                                           const DataID constant_id);

/** @brief Set the constant's buffer in context for sharing.
 *
 * @param shared_context Shared context to set constant buffer in.
 * @param constant Constant node to set buffer for.
 * @return Return true if buffer was set successfully, false otherwise.
 */
OPENVINO_API bool set_constant(Context& shared_context, const ov::op::v0::Constant& constant);

/** @brief Set the weight source in context for sharing.
 *
 * @param shared_context Shared context to set weight source in.
 * @param constant Weight buffer to set as source.
 * @return Return true if source was set successfully, false otherwise.
 */
OPENVINO_API bool set_weight_source(Context& shared_context, const WeightBuffer& constant);
}  // namespace ov::weight_sharing

namespace ov {
namespace wsh = ov::weight_sharing;
}
