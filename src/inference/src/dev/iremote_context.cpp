// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_context.hpp"

#include "dev/make_tensor.hpp"

std::shared_ptr<ov::ITensor> ov::IRemoteContext::create_host_tensor(const ov::element::Type type,
                                                                    const ov::Shape& shape) {
    return ov::make_tensor(type, shape);
}
