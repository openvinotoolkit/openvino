// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_config.hpp
 */

#pragma once

#include <string>
#include "openvino/runtime/properties.hpp"

// ! [public_header:properties]
namespace ov {
namespace template_plugin {

enum class Value {
    UNDEFINED = -1, //
    SOME = 0,
    OTHER = 1
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Value& value) {
    switch (value) {
    case Value::UNDEFINED:
        return os << "";
    case Value::SOME:
        return os << "SOME";
    case Value::OTHER:
        return os << "OTHER";
    default:
        throw ov::Exception{"Unsupported template property value"};
    }
}

inline std::istream& operator>>(std::istream& is, Value& value) {
    std::string str;
    is >> str;
    if (str == "") {
        value = Value::UNDEFINED;
    } else if (str == "SOME") {
        value = Value::SOME;
    } else if (str == "OTHER") {
        value = Value::OTHER;
    } else {
        throw ov::Exception{"Unsupported template property value: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief Read write property example
 */
static constexpr Property<Value> rw_property{"TEMPLATE_READ_WRITE_PROPERTY"};
/**
 * @brief Read-only property example
 */
static constexpr Property<uint32_t, PropertyMutability::RO> ro_property{"TEMPLATE_READ_ONLY_PROPERTY"};
}
}
// ! [public_header:properties]
