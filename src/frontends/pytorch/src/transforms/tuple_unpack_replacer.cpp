// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tuple_unpack_replacer.hpp"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

PrimTupleUnpackReplacer::PrimTupleUnpackReplacer() {
    auto tuple_unpack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto tuple_unpack = cast_fw_node(m.get_match_root(), "prim::TupleUnpack");
        if (!tuple_unpack)
            return false;
        OutputVector outputs;
        auto input_node = tuple_unpack->get_input_node_shared_ptr(0);
        auto tuple_construct = cast_fw_node(input_node, "prim::TupleConstruct");
        if (!tuple_construct) {
            return false;
        }
        for (const auto& input : input_node->inputs()) {
            const auto& out = input.get_source_output();
            outputs.push_back(out);
        }
        replace_node(tuple_unpack, outputs);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tuple_unpack,
                                                          "ov::frontend::pytorch::pass::PrimTupleUnpackReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
