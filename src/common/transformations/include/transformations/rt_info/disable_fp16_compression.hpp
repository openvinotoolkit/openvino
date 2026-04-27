// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void disable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool fp16_compression_is_disabled(const std::shared_ptr<const Node>& node);

TRANSFORMATIONS_API void postpone_fp16_compression(RTMap& rt_info);

TRANSFORMATIONS_API bool is_fp16_compression_postponed(const RTMap& rt_info);

TRANSFORMATIONS_API void do_not_postpone_fp16_compression(RTMap& rt_info);

/**
 * @ingroup ov_runtime_attr_api
 * @brief DisableFP16Compression class represents runtime info attribute that marks operation
 * as prohibited to convert to lower precision (e.g. to FP16) and they should be inferred precisely in the original
 * precision.
 */
class TRANSFORMATIONS_API DisableFP16Compression : public RuntimeAttribute {
public:
    OPENVINO_RTTI("precise", "0", RuntimeAttribute);

    DisableFP16Compression() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }

    bool is_copyable() const override {
        return false;
    }
};

/**
 * @brief Disable precision compression from any type (dynamic) to the specified type on a @ref Node.
 *
 * @param node  Node to apply the attribute to.
 * @param to    Target element type to disable compression to.
 */
TRANSFORMATIONS_API void disable_compression_to(const std::shared_ptr<Node>& node, const element::Type& to);

/**
 * @brief Disable precision compression from one type to another on a @ref Node.
 *
 * @param node  Node to apply the attribute to.
 * @param from  Source element type.
 * @param to    Target element type.
 */
TRANSFORMATIONS_API void disable_compression_from_to(const std::shared_ptr<Node>& node,
                                                     const element::Type& from,
                                                     const element::Type& to);

/**
 * @brief Disable precision compression for all combinations of the specified source and target types on a @ref Node.
 *
 * @param node        Node to apply the attribute to.
 * @param from_types  Source element types.
 * @param to_types    Target element types.
 */
TRANSFORMATIONS_API void disable_compression_from_to(const std::shared_ptr<Node>& node,
                                                     const std::vector<element::Type>& from_types,
                                                     const std::vector<element::Type>& to_types);

/**
 * @brief Enable precision compression from any type (dynamic) to the specified type on a @ref Node.
 *
 * @param node  Node to remove the attribute from.
 * @param to    Target element type to enable compression to.
 */
TRANSFORMATIONS_API void enable_compression_to(const std::shared_ptr<Node>& node, const element::Type& to);

/**
 * @brief Enable precision compression from one type to another on a @ref Node.
 *
 * @param node  Node to remove the attribute from.
 * @param from  Source element type.
 * @param to    Target element type.
 */
TRANSFORMATIONS_API void enable_compression_from_to(const std::shared_ptr<Node>& node,
                                                    const element::Type& from,
                                                    const element::Type& to);

/**
 * @brief Enable precision compression for all combinations of the specified source and target types on a @ref Node.
 *
 * @param node        Node to remove the attribute from.
 * @param from_types  Source element types.
 * @param to_types    Target element types.
 */
TRANSFORMATIONS_API void enable_compression_from_to(const std::shared_ptr<Node>& node,
                                                    const std::vector<element::Type>& from_types,
                                                    const std::vector<element::Type>& to_types);

/**
 * @brief Check if precision compression from any type (dynamic) to the specified type is disabled on a @ref Node.
 *
 * @param node  Node to check.
 * @param to    Target element type to check.
 * @return true if compression to the given type is disabled, false otherwise.
 */
TRANSFORMATIONS_API bool is_compression_disabled_to(const std::shared_ptr<const Node>& node, const element::Type& to);

/**
 * @brief Check if precision compression from one type to another is disabled on a @ref Node.
 *
 * @param node  Node to check.
 * @param from  Source element type.
 * @param to    Target element type.
 * @return true if compression from the given source to the given target type is disabled, false otherwise.
 */
TRANSFORMATIONS_API bool is_compression_disabled_from_to(const std::shared_ptr<const Node>& node,
                                                         const element::Type& from,
                                                         const element::Type& to);
struct ElementTypeHash {
    size_t operator()(const ov::element::Type& t) const {
        return t.hash();
    }
};
using DisabledPrecisionMap = std::unordered_map<ov::element::Type, std::set<ov::element::Type>, ElementTypeHash>;

class TRANSFORMATIONS_API DisablePrecisionConversion : public RuntimeAttribute {
public:
    OPENVINO_RTTI("DisablePrecisionConversion", "0", RuntimeAttribute);

    DisablePrecisionConversion() = default;

    explicit DisablePrecisionConversion(const element::Type& from, const element::Type& to) {
        m_disabled_precisions[from].insert(to);
    }

    bool is_copyable() const override {
        return false;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    DisabledPrecisionMap m_disabled_precisions = {};
};

template <>
class TRANSFORMATIONS_API AttributeAdapter<DisabledPrecisionMap> : public ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<DisabledPrecisionMap>");

    explicit AttributeAdapter(DisabledPrecisionMap& value) : m_ref(value), m_serialized() {}

    const std::string& get() override;
    void set(const std::string& value) override;

private:
    DisabledPrecisionMap& m_ref;
    std::string m_serialized;
};

}  // namespace ov