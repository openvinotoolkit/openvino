// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Inference Engine plugins destruction mechanism
 * @file ie_irelease.hpp
 */
#pragma once

#include "ie_no_copy.hpp"
#include <memory>

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is used for objects allocated by a shared module (in *.so)
 */
class IRelease : public no_copy {
public:
    /**
     * @brief Releases current allocated object and all related resources.
     * Once this method is called, the pointer to this interface is no longer valid
     */
    virtual void Release() noexcept = 0;

 protected:
    /**
     * @brief Default destructor
     */
    ~IRelease() override = default;
};



template <class T> inline std::shared_ptr<T> shared_from_irelease(T * ptr) {
    std::shared_ptr<T> pointer(ptr, [](IRelease *p) {
        p->Release();
    });
    return pointer;
}

}  // namespace details
}  // namespace InferenceEngine
