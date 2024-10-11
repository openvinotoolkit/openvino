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

class DQMatMulGQu4 : public ov::pass::MatcherPass {
public:
    DQMatMulGQu4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class DQMatMulCWu4 : public ov::pass::MatcherPass {
public:
    DQMatMulCWu4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class DQMatMulGQi4 : public ov::pass::MatcherPass {
public:
    DQMatMulGQi4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class DQMatMulCWi4 : public ov::pass::MatcherPass {
public:
    DQMatMulCWi4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class VocabMatMul : public ov::pass::MatcherPass {
public:
    VocabMatMul(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class RMSNorm : public ov::pass::MatcherPass {
public:
    RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace compute
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
