// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace frontend {
namespace onnx {

constexpr int LATEST_SUPPORTED_ONNX_OPSET_VERSION = ONNX_OPSET_VERSION;
struct VersionRange {
    constexpr VersionRange(int from, int to) : m_from(from), m_to(to) {}
    static constexpr VersionRange from_version(int from) {
        return VersionRange{from, LATEST_SUPPORTED_ONNX_OPSET_VERSION};
    }
    static constexpr VersionRange to_version(int to) {
        return VersionRange{1, to};
    }
    static constexpr VersionRange in_version(int version) {
        return VersionRange{version, version};
    }
    static constexpr VersionRange single_version_for_all_opsets() {
        return VersionRange{1, LATEST_SUPPORTED_ONNX_OPSET_VERSION};
    }
    // -1 means that that a left/right boundary of the range was not specified
    const int m_from = -1, m_to = -1;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
