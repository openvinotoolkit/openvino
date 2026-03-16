// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

namespace paddle {
namespace framework {
namespace proto {
class ProgramDesc;
}
}  // namespace framework
}  // namespace paddle

namespace ov {
namespace frontend {
namespace paddle {

/// Convert a PIR JSON model (inference.json) to an in-memory ProgramDesc protobuf.
/// This allows PP-OCRv5 and newer Paddle models that use the PIR JSON format
/// to be loaded through the existing protobuf-based pipeline without changes.
std::shared_ptr<::paddle::framework::proto::ProgramDesc> json_to_program_desc(std::istream& json_stream);

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
