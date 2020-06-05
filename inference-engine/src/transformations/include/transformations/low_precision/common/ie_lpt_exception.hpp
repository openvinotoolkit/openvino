// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <ngraph/node.hpp>

/**
* @def THROW_TRANSFORMATION_EXCEPTION_LPT
* @brief A macro used to throw the exception with a notable description for low precision transformations
*/
#define THROW_IE_LPT_EXCEPTION(layer) throw ::ngraph::pass::low_precision::InferenceEngineLptException(__FILE__, __LINE__, layer)

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API InferenceEngineException : std::exception {
    std::shared_ptr<std::ostringstream> buffer;
    mutable std::string buffer_str;
public:
    template <typename T>
    InferenceEngineException& operator<< (const T& x) {
        *buffer << x;
        return *this;
    }

    const char* what() const noexcept override {
        buffer_str = buffer->str();
        return buffer_str.c_str();
    }
};

#define THROW_TRANSFORMATION_EXCEPTION throw ::ngraph::pass::low_precision::InferenceEngineException() << __FILE__ << ":" << __LINE__ << " "


class TRANSFORMATIONS_API InferenceEngineLptException : public InferenceEngineException {
public:
    InferenceEngineLptException(const std::string& filename, const int line, std::shared_ptr<const Node> layer) {
        *this
            << filename << ":" << line << " Exception during low precision transformation for "
            << layer << " node with name '" << layer->get_friendly_name() << "'. ";
    }
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph