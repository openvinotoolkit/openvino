// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph {
namespace onnx_import {
static constexpr const int LATEST_SUPPORTED_ONNX_OPSET_VERSION = ONNX_OPSET_VERSION;
struct VersionRange {
    VersionRange(int from, int to) : m_from(from), m_to(to) {}
    static VersionRange from_version(int from) {
        return VersionRange{from, LATEST_SUPPORTED_ONNX_OPSET_VERSION};
    }
    static VersionRange to_version(int to) {
        return VersionRange{1, to};
    }
    static VersionRange in_version(int version) {
        return VersionRange{version, version};
    }
    static VersionRange single_version_for_all_opsets() {
        return VersionRange{1, LATEST_SUPPORTED_ONNX_OPSET_VERSION};
    }
    const int m_from = -1, m_to = -1;
};
}  // namespace onnx_import
}  // namespace ngraph
