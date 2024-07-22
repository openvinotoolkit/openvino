// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

// Note: the patterns below are only utilized by the online partitioner
namespace patterns {

class RMSNormAvoid : public ov::pass::MatcherPass {
public:
    RMSNormAvoid(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

class DequantMatMulGQ : public ov::pass::MatcherPass {
public:
    DequantMatMulGQ(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class DequantMatMulCW : public ov::pass::MatcherPass {
public:
    DequantMatMulCW(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class SwishMultXMM : public ov::pass::MatcherPass {
public:
    SwishMultXMM(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class RMSNorm : public ov::pass::MatcherPass {
public:
    RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class AdditionalCompute : public ov::pass::MatcherPass {
public:
    AdditionalCompute(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
