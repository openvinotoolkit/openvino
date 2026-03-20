// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_kvcache_to_precision.hpp"

#include "../logging.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/kv_cache_concat.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/move_fake_convert_up_through_kv_cache_concat.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/op_conversions/fake_convert_decomposition.hpp"

namespace opp = ov::pass::pattern;

namespace {

class FakeConvertDestinationTypeExtractor : public ov::pass::MatcherPass {
public:
    FakeConvertDestinationTypeExtractor(std::set<ov::element::Type>& fcDestinationTypes) {
        auto fake_convert_m = opp::wrap_type<ov::op::v13::FakeConvert>();

        ov::matcher_pass_callback callback = [=, &fcDestinationTypes](opp::Matcher& m) {
            const auto& pattern_to_output = m.get_pattern_value_map();
            const auto fake_convert =
                ov::as_type_ptr<ov::op::v13::FakeConvert>(pattern_to_output.at(fake_convert_m).get_node_shared_ptr());

            if (fake_convert == nullptr) {
                return false;
            }
            fcDestinationTypes.insert(fake_convert->get_destination_element_type());
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(fake_convert_m, "FakeConvertDestinationTypeExtractor"),
                         callback);
    }
};

class ConvertTypeRelaxedToRegular : public ov::pass::MatcherPass {
public:
    ConvertTypeRelaxedToRegular() {
        auto pattern = opp::wrap_type<ov::op::TypeRelaxed<ov::op::v1::Multiply>>();

        ov::matcher_pass_callback callback = [](opp::Matcher& m) {
            auto tr_mul = std::dynamic_pointer_cast<ov::op::TypeRelaxed<ov::opset1::Multiply>>(m.get_match_root());
            if (tr_mul) {
                auto new_mul = std::make_shared<ov::opset1::Multiply>(tr_mul->input_value(0), tr_mul->input_value(1));
                new_mul->set_friendly_name(tr_mul->get_friendly_name());
                ov::copy_runtime_info(tr_mul, new_mul);
                ov::replace_node(tr_mul, new_mul);
            }
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(pattern, "ConvertTypeRelaxedToRegular"), callback);
    }
};

std::shared_ptr<ov::Model> cvt_kvcache_to_low_precision(const std::shared_ptr<ov::Model>& model,
                                                        const ov::element::Type lptype) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (const auto& tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(lptype);
        }
    }

    for (const auto& tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(lptype);
        }
    }

    return ppp.build();
}

}  // namespace

namespace ov {
namespace npuw {

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Running FP8 static quantization on model: " << model->get_name());

    auto kv_kache_storage_type = ov::element::f16;

    ov::pass::low_precision::LayerTransformation::Params params;
    std::set<ov::element::Type> fcTypesInput, fcTypesRemained;

    ov::pass::Manager manager("optimize_fp8");
    params.defaultPrecisions = ov::pass::low_precision::precision_set::get_fp8_support();
    manager.register_pass<FakeConvertDestinationTypeExtractor>(fcTypesInput);
    manager.register_pass<ov::pass::low_precision::MoveFakeConvertUpThroughKVCacheConcat>();
    auto graph_rewrite = manager.register_pass<ov::pass::GraphRewrite>();
    graph_rewrite->add_matcher<ov::pass::FakeConvertDecomposition>();
    graph_rewrite->add_matcher<ov::pass::low_precision::ConcatTransformation>(params);
    graph_rewrite->add_matcher<ov::pass::low_precision::KVCacheConcat>(model);
    manager.register_pass<ConvertTypeRelaxedToRegular>();
    manager.register_pass<FakeConvertDestinationTypeExtractor>(fcTypesRemained);
    manager.run_passes(model);
    if (fcTypesInput.empty()) {
        LOG_WARN("FakeConvert layers not detected - leaving kv-cache in " << kv_kache_storage_type << " precision");
    } else if (!fcTypesRemained.empty()) {
        LOG_WARN(fcTypesRemained.size() << " FakeConvert layers not decomposed - leaving kv-cache in "
                                        << kv_kache_storage_type << " precision");
    } else if (fcTypesInput.size() > 1) {
        auto it2 = std::next(fcTypesInput.begin(), 1);
        LOG_WARN("FakeConvert layers had several precisions (" << fcTypesInput.size() << ")-" << *fcTypesInput.begin()
                                                               << ", " << *it2 << ", ... "
                                                               << "supported only single precision");
        LOG_WARN("Leaving KV-cache in " << kv_kache_storage_type << " precision");
    } else {
        kv_kache_storage_type = *fcTypesInput.begin();
        LOG_DEBUG("KV cache storage precision changed to " << kv_kache_storage_type);
    }
    return kv_kache_storage_type;
}

}  // namespace npuw
}  // namespace ov

ConvertKVCacheToPrecision::ConvertKVCacheToPrecision(const ov::element::Type lptype) : m_lp_type(lptype) {}

bool ConvertKVCacheToPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto ppp_result = cvt_kvcache_to_low_precision(model, m_lp_type);
    // PrePostProcessor currently always modifies the model in-place and returns the same model pointer, but let's
    // be defensive here and check it just in case
    OPENVINO_ASSERT(ppp_result == model,
                    "PrePostProcessor should not create a new model, but returned a different one.");

    return true;
}
