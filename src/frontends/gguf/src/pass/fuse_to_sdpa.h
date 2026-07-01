// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

class FuseToSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::gguf::pass::FuseToSDPA")
    FuseToSDPA();
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
