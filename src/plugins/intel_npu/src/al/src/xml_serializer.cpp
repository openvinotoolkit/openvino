// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/xml_serializer.hpp"

#include "openvino/op/util/op_types.hpp"

namespace intel_npu {

ov::util::ConstantWriter& XmlSerializer::get_constant_write_handler() {
    return m_weightless_writer;
}

}  // namespace intel_npu
