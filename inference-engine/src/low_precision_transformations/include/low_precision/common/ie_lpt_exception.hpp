// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <string>
#include <ngraph/node.hpp>
#include <low_precision/lpt_visibility.hpp>

/**
* @def THROW_TRANSFORMATION_EXCEPTION_LPT
* @brief A macro used to throw the exception with a notable description for low precision transformations
*/
#define THROW_IE_LPT_EXCEPTION(node) throw ::ngraph::pass::low_precision::InferenceEngineLptException(__FILE__, __LINE__, node)

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API Exception : std::exception {
    std::shared_ptr<std::ostringstream> buffer;
    mutable std::string buffer_str;
public:
    Exception() {
        buffer = std::make_shared<std::ostringstream>();
    }

    template <typename T>
    Exception& operator<< (const T& x) {
        *buffer << x;
        return *this;
    }

    const char* what() const noexcept override {
        buffer_str = buffer->str();
        return buffer_str.c_str();
    }
};

#define THROW_TRANSFORMATION_EXCEPTION throw ::ngraph::pass::low_precision::Exception() << __FILE__ << ":" << __LINE__ << " "


class LP_TRANSFORMATIONS_API InferenceEngineLptException : public Exception {
public:
    InferenceEngineLptException(const std::string& filename, const size_t line, const Node& node) {
        *this
            << filename << ":" << line << " Exception during low precision transformation for "
            << node << " node with type '" << node.get_type_name() << "', name '" << node.get_friendly_name() << "'. ";
    }
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
