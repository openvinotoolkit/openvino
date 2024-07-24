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
namespace compute {

class DQMatMulGQ : public ov::pass::MatcherPass {
public:
    DQMatMulGQ(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class DQMatMulCW : public ov::pass::MatcherPass {
public:
    DQMatMulCW(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class SwishMul : public ov::pass::MatcherPass {
public:
    SwishMul(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class RMSNorm : public ov::pass::MatcherPass {
public:
    RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace compute
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
