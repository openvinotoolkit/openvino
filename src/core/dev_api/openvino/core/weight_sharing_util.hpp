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

/** @brief Defined invalid ID for weight source */
inline constexpr DataID invalid_source_id = 0;
/** @brief Defined invalid ID for constant */
inline constexpr DataID invalid_constant_id = invalid_source_id;

/** @brief Metadata for a weight */
struct WeightMetaData {
    size_t m_offset;
    size_t m_size;
    ov::element::Type m_type;
};

/** @brief Metadata for the origin of a weight */
struct WeightOriginMetaData {
    DataID m_id;
    size_t m_offset;
    size_t m_size;
    ov::element::Type m_type;
};

/** @brief Variant type for weight buffer shared pointer*/
using WeightBuffer = std::shared_ptr<ov::AlignedBuffer>;
/** @brief Variant type for weight buffer observer*/
using WeakWeightBuffer = std::weak_ptr<ov::AlignedBuffer>;

/** @brief Map [key: Constant ID, value: WeightMetaData] of constant meta data for single container. */
using WeightMetaMap = std::unordered_map<DataID, WeightMetaData>;
/** @brief Map [key: Source ID, value: WeightMetaMap] of constant metadata for constant sources. */
using WeightRegistry = std::unordered_map<DataID, WeightMetaMap>;
/** @brief Map [key: Source ID, value: WeakWeightBuffer] of pointers to constant sources. */
using WeightSourceRegistry = std::unordered_map<DataID, WeakWeightBuffer>;
/** @brief Map [key: Blob ID, value: blob name] of blobs to model name/tag */
using BlobMap = std::unordered_map<DataID, std::string>;

/** @brief Shared context for weight and constant management */
struct Context {
    WeightRegistry m_weight_registry;        //!< Weight metadata stored in cache for weight sources.
    WeightSourceRegistry m_cache_sources;    //!< Weight sources stored in cache.
    WeightSourceRegistry m_runtime_sources;  //!< Weight sources available in runtime, not stored in cache.
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
    static std::optional<WeightOriginMetaData> get_constant_origin(const ov::op::v0::Constant& constant);

    /** @brief Get the constant source buffer for constant node.
     *
     * @param constant Constant node to get source buffer for.
     * @return Return shared pointer to AlignedBuffer if found, nullptr otherwise.
     */
    static std::shared_ptr<ov::AlignedBuffer> get_constant_source_buffer(const ov::op::v0::Constant& constant);

    /**
     * @brief Set constant metadata in weight registry for given constant node.
     *
     * @param weight_registry Weight registry to set metadata in.
     * @param constant Constant node to set metadata for.
     * @return Return true if metadata was set successfully, false otherwise.
     */
    static bool set_constant_in_weight_registry(WeightRegistry& weight_registry, const ov::op::v0::Constant& constant);

    /** @brief Gets the weight sources for a given model.
     *
     * @param model Model to get weight sources for.
     * @return Return map of weight sources.
     */
    static WeightSourceRegistry get_weight_sources(const Model& model);

    /** @brief Gets the map where Source ID and Data ID are keys to identify weight meta data.
     *
     * @param model Model to get weight registry for.
     * @return Return map of weight metadata.
     */
    static WeightRegistry get_weight_registry(const Model& model);
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

/** @brief Get the buffer for a given source buffer (provide source id as hint) and constant id.
 *
 * The returned buffer is ready to use for Constant node creation.
 *
 * @param shared_context Shared context to get buffer from.
 * @param source_buffer Source buffer to restore constant buffer to get buffer from.
 * @param constant_id Constant id to get buffer for.
 * @return Return shared pointer to AlignedBuffer if found, nullptr otherwise.
 * @{
 */
OPENVINO_API std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                                           const std::shared_ptr<ov::AlignedBuffer>& source_buffer,
                                                           const DataID constant_id);

OPENVINO_API std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                                           const std::shared_ptr<ov::MappedMemory>& source_buffer,
                                                           const DataID constant_id);
/** @} */

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
 * @param source_buffer Weight buffer to set as source.
 * @return Return true if source was set successfully, false otherwise.
 */
OPENVINO_API bool set_weight_source(Context& shared_context, const WeightBuffer& source_buffer);

/**
 * @brief Set the weight source object for a given constant node.
 *
 * @param shared_context Shared context to set weight source in.
 * @param constant Constant node to extract source buffer and set as weight source.
 * @return Return true if source was set successfully, false otherwise.
 */
OPENVINO_API bool set_weight_source(Context& shared_context, const ov::op::v0::Constant& constant);

/** @brief Set the weight source in runtime source map of context for sharing.
 *
 * @param shared_context Shared context to set weight source in.
 * @param source_buffer Weight buffer to set as source.
 * @return Return true if source was set successfully, false otherwise.
 */
OPENVINO_API bool set_runtime_weight_source(Context& shared_context, const WeightBuffer& source_buffer);
}  // namespace ov::weight_sharing

namespace ov {
namespace wsh = ov::weight_sharing;
}
