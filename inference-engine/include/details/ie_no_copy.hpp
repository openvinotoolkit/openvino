// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief header file for no_copy class
 * @file ie_no_copy.hpp
 */
#pragma once

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is used for objects returned from the shared library factory to prevent copying
 */
class no_copy {
protected:
    /**
     * @brief A default constructor
     */
    no_copy() = default;

    /**
     * @brief A default destructor
     */
    virtual ~no_copy() = default;

    /**
     * @brief A removed copy constructor
     */
    no_copy(no_copy const &) = delete;

    /**
     * @brief A removed assign operator
     */
    no_copy &operator=(no_copy const &) = delete;

    /**
     * @brief A removed move constructor
     */
    no_copy(no_copy &&) = delete;

    /**
     * @brief A removed move operator
     */
    no_copy &operator=(no_copy &&) = delete;
};
}  // namespace details
}  // namespace InferenceEngine
