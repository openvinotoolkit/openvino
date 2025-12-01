// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "complex_type_mark_remover.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

ComplexTypeMarkRemover::ComplexTypeMarkRemover() {
    auto complex_type_mark = ov::pass::pattern::wrap_type<ov::frontend::ComplexTypeMark>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto complex_node = ov::as_type_ptr<ov::frontend::ComplexTypeMark>(m.get_match_root());
        if (!complex_node)
            return false;

        // Replace ComplexTypeMark with its underlying data representation
        // The data is a floating-point tensor with shape [..., 2] where
        // [..., 0] is the real part and [..., 1] is the imaginary part
        auto data = complex_node->get_data();
        complex_node->output(0).replace(data);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(complex_type_mark,
                                                          "ov::frontend::pytorch::pass::ComplexTypeMarkRemover");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
