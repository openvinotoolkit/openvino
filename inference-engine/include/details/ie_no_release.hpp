// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Utility header file. Provides no release base class
 * @file ie_no_release.hpp
 */
#pragma  once

namespace  InferenceEngine {
namespace details {

/**
 * @brief prevent Release method from being called on specific objects
 */
template<class T>
class NoReleaseOn : public T {
 private :
    void Release() noexcept = 0;
};

}  // namespace details
}  // namespace InferenceEngine