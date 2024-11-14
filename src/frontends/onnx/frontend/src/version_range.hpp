// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace frontend {
namespace onnx {

constexpr int LATEST_SUPPORTED_ONNX_OPSET_VERSION = ONNX_OPSET_VERSION;
struct VersionRange {
    constexpr VersionRange(int since_version, int until_version) : m_since(since_version), m_until(until_version) {}
    static constexpr VersionRange since(int since_version) {
        return VersionRange{since_version, LATEST_SUPPORTED_ONNX_OPSET_VERSION};
    }
    static constexpr VersionRange until(int until_version) {
        return VersionRange{1, until_version};
    }
    static constexpr VersionRange in(int version) {
        return VersionRange{version, version};
    }
    // -1 means that that a left/right boundary of the range was not specified
    const int m_since = -1, m_until = -1;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
