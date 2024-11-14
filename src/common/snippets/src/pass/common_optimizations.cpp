// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/transform_convert.hpp"
#include "snippets/pass/validate.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/pass/extract_constants.hpp"
#include "snippets/pass/extract_unsupported_transposes.hpp"
#include "snippets/pass/subgraph_manager.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

#define REGISTER_SNIPPETS_PASS(manager, pass, enabled, ...) \
    if (enabled) \
        manager.register_pass<pass>(__VA_ARGS__);

CommonOptimizations::CommonOptimizations(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(CommonOptimizations);
    ov::graph_rewrite_callback callback = [&](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        const auto& body = subgraph->body_ptr();
        const auto is_quantized = subgraph->is_quantized();
        const auto is_domain_sensitive = subgraph->has_domain_sensitive_ops();

        // Firstly, we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable specific transformation for quantized subgraphs.
        ov::pass::Manager manager(get_pass_config(), "Snippets:CommonOptimizations");
        REGISTER_SNIPPETS_PASS(manager, ov::snippets::pass::TransformConvertToConvertTruncation, true);
        REGISTER_SNIPPETS_PASS(manager, ov::snippets::pass::ExplicitTransposeMatMulInputs, is_domain_sensitive);
        REGISTER_SNIPPETS_PASS(manager, ov::snippets::pass::CommonFakeQuantizeDecomposition, is_quantized);
        REGISTER_SNIPPETS_PASS(manager, ov::snippets::pass::SoftmaxReshapeElimination, is_domain_sensitive);
        manager.run_passes(body);

        ov::snippets::pass::CommonOptimizations::SubgraphManager subgraph_manager;
        // At the moment only non-scalar Constants of FakeQuantize can be inside Subgraph
        // so we can enable ExtractConstants pass for quantized models
        REGISTER_SNIPPETS_PASS(subgraph_manager, ov::snippets::pass::ExtractConstants, is_quantized);
        REGISTER_SNIPPETS_PASS(subgraph_manager, ov::snippets::pass::ExtractUnsupportedTransposes, is_domain_sensitive);
        REGISTER_SNIPPETS_PASS(subgraph_manager, ov::snippets::pass::SplitDimensionM, is_domain_sensitive && config.get_split_m_dimension(),
                               config.get_concurrency());
        subgraph_manager.run_passes(subgraph);

        // Validate the body after all common optimizations
        ov::snippets::pass::Validate(get_pass_config()).run_on_model(body);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>(), matcher_name);
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ov
