// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "snippets/op/brgemm.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface MatMulToBrgemm
 * @brief Replaces ov::MatMul with snippets::op::Brgemm operation (only non-trasposing MatMuls are currently supported)
 * @ingroup snippets
 */
class MatMulToBrgemm: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulToBrgemm", "0");
    MatMulToBrgemm();

private:
    void init_ports(const std::shared_ptr<op::Brgemm>& brgemm) const;
};


}  // namespace pass
}  // namespace snippets
}  // namespace ov
