// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "openvino/runtime/properties.hpp"

namespace cldnn {
class BinaryInputBuffer;
class engine;
}  // namespace cldnn

namespace ov::intel_gpu::hsm {

// Owns HSM section streams through graph reconstruction; leaves legacy decoding to the plugin.
class ReaderHandler {
public:
    ReaderHandler(std::istream& source, cldnn::engine& engine, const size_t* tensor_base, size_t tensor_size, Crypt decrypt);
    ~ReaderHandler();

    bool is_hsm() const;
    std::istream& legacy_stream();
    cldnn::BinaryInputBuffer& buffer();
    ov::CacheMode cache_mode() const;
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::intel_gpu::hsm
