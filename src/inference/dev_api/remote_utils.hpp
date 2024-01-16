// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_ngraph_utils.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace legacy_convert {

class INFERENCE_ENGINE_API_CLASS(TensorHolder) {
public:
    TensorHolder(ov::SoPtr<ov::ITensor> tensor) : _tensor(tensor) {}

    const ov::SoPtr<ov::ITensor>& get_tensor() const {
        return _tensor;
    }

private:
    ov::SoPtr<ov::ITensor> _tensor;
};

}  // namespace legacy_convert
}  // namespace ov
