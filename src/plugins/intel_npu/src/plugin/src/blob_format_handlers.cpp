// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_handlers.hpp"

namespace intel_npu {

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream npu_formatted_blob) {
    return nullptr;
}

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob) {
    return nullptr;
}

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
