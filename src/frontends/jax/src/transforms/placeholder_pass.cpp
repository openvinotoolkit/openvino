// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "placeholder_pass.hpp"

#include "openvino/op/if.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "jax_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace pass {

using namespace ov::op;

PrimPlaceholderReplacer::PrimPlaceholderReplacer() {
    auto tuple_unpack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tuple_unpack,
                                                          "ov::frontend::jax::pass::PrimPlaceholderReplacer");
    this->register_matcher(m, callback);
};

bool PlaceholderInBodyReplacer::run_on_model(const std::shared_ptr<Model>& model) {
    return true;
};

}  // namespace pass
}  // namespace jax
}  // namespace frontend
}  // namespace ov
