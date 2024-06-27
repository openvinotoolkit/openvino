// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

bool is_set(const std::size_t sub_idx, const std::string& opt);

// Every great project has its own string class...
// NB: Newer C++ standards would allow to use string views or smt
ov::Tensor tensor_from_const(const std::shared_ptr<ov::Node>& node);

bool starts_with(const std::string& str, const std::string& prefix);

std::string fmt(std::size_t number, std::size_t total);

struct UnpackOptions {
    bool bUseOvParallelFor;
    size_t nPartitions;  // if 0 we use 64 elements step in parallel for, otherwise  target workload is dynamically
                         // calculated
    bool bStrictPartitioning;  // cannot reduce partitions in favor of speed
    explicit UnpackOptions(bool useParallelFor, size_t nPartitions, bool bStrictPartitioning)
        : bUseOvParallelFor(useParallelFor),
          nPartitions(nPartitions),
          bStrictPartitioning(bStrictPartitioning) {}
};

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void to_f32(const ov::Tensor& in, ov::Tensor& out);

}  // namespace util
}  // namespace npuw
}  // namespace ov
