// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_context.hpp"

#include "openvino/runtime/make_tensor.hpp"

ov::SoPtr<ov::ITensor> ov::IRemoteContext::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return ov::SoPtr<ov::ITensor>(ov::make_tensor(type, shape), nullptr);
}
