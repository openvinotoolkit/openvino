// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"



namespace ov {
std::shared_ptr<ITagBuffer> as_itag_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer){
    return std::dynamic_pointer_cast<ITagBuffer>(buffer);
}

const ITagBuffer* as_itag_buffer(const ov::AlignedBuffer& buffer){
    return dynamic_cast<const ITagBuffer*>(&buffer);
}



}  // namespace ov
