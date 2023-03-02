// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/matmul_to_brgemm.hpp"

#include "snippets/op/brgemm.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

MatMulToBrgemm::MatMulToBrgemm() {
    MATCHER_SCOPE(MatMulToBrgemm);
    auto matmul_pattern = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ngraph::pattern::any_input(),
                                                                               ngraph::pattern::any_input()});

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MatMulToBrgemm")
        auto& pm = m.get_pattern_value_map();
        const auto matmul = as_type_ptr<ngraph::opset1::MatMul>(pm.at(matmul_pattern).get_node_shared_ptr());
        // Brgemm doesn't support transposed inputs currently, so we don't convert such matmuls
        if (matmul->get_transpose_a() || matmul->get_transpose_b())
            return false;

        auto brgemm = std::make_shared<op::Brgemm>(matmul->get_input_source_output(0), matmul->get_input_source_output(1));
        brgemm->set_friendly_name(matmul->get_friendly_name());
        ngraph::copy_runtime_info(matmul, brgemm);
        ngraph::replace_node(matmul, brgemm);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
