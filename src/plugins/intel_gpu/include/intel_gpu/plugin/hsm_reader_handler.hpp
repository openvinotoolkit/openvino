// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>
#include <string>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace cldnn {
class BinaryInputBuffer;
class engine;
}  // namespace cldnn

namespace ov::intel_gpu::hsm {

// Routes legacy/HSM and owns all temporary section streams through graph reconstruction.
class ReaderHandler {
public:
    ReaderHandler(std::istream& source, cldnn::engine& engine, std::shared_ptr<ov::Tensor> backing, Crypt decrypt);
    ~ReaderHandler();

    cldnn::BinaryInputBuffer& buffer();
    ov::CacheMode cache_mode() const;
    const std::string& requirements() const;
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::intel_gpu::hsm
