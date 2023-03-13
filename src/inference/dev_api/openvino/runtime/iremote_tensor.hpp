// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime IRemoteTensor interface
 * @file openvino/runtime/iremote_tensor.hpp
 */

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class OPENVINO_RUNTIME_API IRemoteTensor : public ITensor {
private:
    template <typename T>
    struct fail : std::false_type {};

public:
    void* data(const element::Type& type = {}) const final {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ~IRemoteTensor() override;

    /**
     * @brief Return additional information associated with tensor
     * @return Map of property names to properties
     */
    virtual AnyMap get_properties() const = 0;
};
}  // namespace ov
