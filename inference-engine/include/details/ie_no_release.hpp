// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Utility header file. Provides no release base class
 *
 * @file ie_no_release.hpp
 */
#pragma once

namespace InferenceEngine {
namespace details {

// TODO: eshoguli: remove
IE_SUPPRESS_DEPRECATED_START

/**
 * @brief prevent Release method from being called on specific objects
 */
template <class T>
class NoReleaseOn : public T {
private:
    void Release() noexcept = 0;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
