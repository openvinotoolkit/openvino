// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"

namespace pugi {
class xml_node;
}

namespace ov {

/**
 * @brief Provide an abstract interface for lazy reading of meta information
 */
class OPENVINO_API Meta {
public:
    /**
     * @brief Parses and returns meta information by request
     *
     * @return ov::AnyMap with meta information
     */
    virtual operator ov::AnyMap&() = 0;
    /**
     * @brief Parses and returns meta information by request
     *
     * @return const ov::AnyMap with meta information
     */
    virtual operator const ov::AnyMap&() const = 0;

    /**
     * @brief Destructor
     */
    virtual ~Meta() = default;
};

class MetaDataWithPugixml : public Meta {
public:
    /**
     * @brief Returns meta unchanged meta information. Throws ov::Exception if the meta was potentially changed
     *
     * @return const pugi::xml_node& with meta information
     */
    virtual const pugi::xml_node& get_pugi_node() const = 0;
};

}  // namespace ov
