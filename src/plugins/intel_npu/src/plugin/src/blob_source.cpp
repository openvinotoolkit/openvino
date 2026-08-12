// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_source.hpp"

namespace {

using namespace intel_npu;

}  // namespace

namespace intel_npu {

BlobSource::BlobSource(std::istream& source, const ov::log::Level log_level = Logger::global().level()) {}

BlobSource::BlobSource(const ov::Tensor& source, const ov::log::Level log_level = Logger::global().level()) {}

void BlobSource::copy_from_source(void* destination, const size_t size) {}

void* BlobSource::interpret_from_source(const size_t size) {}

ov::Tensor BlobSource::get_roi_tensor_from_source(const size_t size) {}

void BlobSource::move_cursor(const std::streamoff offset, const std::ios_base::seekdir reference = std::ios::beg) {}

std::streampos BlobSource::get_cursor() const {}

size_t BlobSource::get_size() const {}

}  // namespace intel_npu
