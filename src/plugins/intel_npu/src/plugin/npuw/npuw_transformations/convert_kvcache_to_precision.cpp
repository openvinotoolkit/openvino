// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_kvcache_to_precision.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "kv_cache_compressed.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/kv_cache_concat.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/move_fake_convert_up_through_kv_cache_concat.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
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
                                                        const ov::element::Type lptype,
                                                        const bool v_tensors_transposed,
                                                        const std::string& kv_cache_compression_config = "") {
    // Resolve storage types first and apply them through PPP for both inputs and outputs.
    // Default path keeps KV cache in f16; the integer hint stores each cache in the precision resolved from
    // NPUW_LLM_KV_CACHE_COMPRESSION (key: the hint, i8 or i4; value: i8 or i4).
    auto key_storage_type = lptype;
    auto value_storage_type = lptype;

    const bool use_integer_kv_storage = (lptype == ov::element::i8 || lptype == ov::element::u8);
    ov::npuw::KVCacheCompressionParams dq_params;
    if (use_integer_kv_storage) {
        dq_params = ov::npuw::resolve_kv_cache_compression(
            lptype,
            ov::npuw::parse_kv_cache_compression_config(kv_cache_compression_config));
        key_storage_type = dq_params.key.quantization_dt;
        value_storage_type = dq_params.value.quantization_dt;
    }
    ov::preprocess::PrePostProcessor ppp(model);

    for (const auto& tensor : model->inputs()) {
        // Defensive: get_any_name() throws on a nameless port.
        if (tensor.get_names().empty()) {
            continue;
        }
        const auto& name = tensor.get_any_name();
        if (ov::npuw::util::isPastKeyValuesKey(name).has_value()) {
            ppp.input(name).tensor().set_element_type(key_storage_type);
        } else if (ov::npuw::util::isPastKeyValuesValue(name).has_value()) {
            ppp.input(name).tensor().set_element_type(value_storage_type);
        }
    }

    for (const auto& tensor : model->outputs()) {
        // See above.
        if (tensor.get_names().empty()) {
            continue;
        }
        const auto& name = tensor.get_any_name();
        if (ov::npuw::util::isPresentKeyValuesKey(name).has_value()) {
            ppp.output(name).tensor().set_element_type(key_storage_type);
        } else if (ov::npuw::util::isPresentKeyValuesValue(name).has_value()) {
            ppp.output(name).tensor().set_element_type(value_storage_type);
        }
    }
    auto new_model = ppp.build();

    if (use_integer_kv_storage) {
        LOG_DEBUG("Running KV-cache compression passes: key=" << key_storage_type << ", value=" << value_storage_type
                                                              << " on model[" << model->get_friendly_name() << "]");
        ov::npuw::run_kv_cache_dynamic_quantization_passes(new_model, dq_params, v_tensors_transposed);
    }
    return new_model;
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

namespace ov::npuw {

KVCacheCompressionSetting parse_kv_cache_compression_config(const std::string& config_str) {
    using QuantizationType = KVCacheCompressionConfig::QuantizationType;
    KVCacheCompressionSetting result;
    if (config_str.empty()) {
        LOG_DEBUG("Using default KV-cache compression: KEY:SYM:BY_TOKEN,VALUE:SYM:BY_TOKEN");
        return result;
    }

    auto split = [](const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        while (true) {
            const auto pos = str.find(delimiter, start);
            tokens.push_back(str.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
            if (pos == std::string::npos) {
                return tokens;
            }
            start = pos + 1;
        }
    };
    auto trim = [](const std::string& str) {
        const auto first = str.find_first_not_of(" \t");
        if (first == std::string::npos) {
            return std::string{};
        }
        return str.substr(first, str.find_last_not_of(" \t") - first + 1);
    };
    auto to_upper = [](std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return str;
    };

    bool key_seen = false;
    bool value_seen = false;
    for (const auto& raw_part : split(config_str, ',')) {
        const auto part = trim(raw_part);
        OPENVINO_ASSERT(!part.empty(), "Empty part in KV-cache compression config '", config_str, "'");

        // Every attribute is a plain string matched by name, so their order does not matter
        std::optional<bool> is_key;
        std::optional<QuantizationType> type;
        std::optional<std::string> granularity;
        std::optional<ov::element::Type> precision;
        auto set_once = [&part](auto& slot, auto value, const char* what) {
            OPENVINO_ASSERT(!slot.has_value(),
                            "More than one ",
                            what,
                            " in KV-cache compression config part '",
                            part,
                            "'");
            slot = value;
        };
        for (const auto& raw_attr : split(part, ':')) {
            const auto attr = to_upper(trim(raw_attr));
            if (attr == "KEY" || attr == "VALUE") {
                set_once(is_key, attr == "KEY", "KEY/VALUE label");
            } else if (attr == "SYM" || attr == "ASYM") {
                set_once(type,
                         attr == "SYM" ? QuantizationType::Symmetric : QuantizationType::Asymmetric,
                         "quantization type (SYM/ASYM)");
            } else if (attr == "BY_TOKEN") {
                set_once(granularity, attr, "granularity");
            } else if (attr == "BY_CHANNEL") {
                OPENVINO_THROW("Granularity BY_CHANNEL is not supported yet in KV-cache compression config part '",
                               part,
                               "' (use BY_TOKEN)");
            } else if (attr == "INT4" || attr == "I4") {
                set_once(precision, ov::element::i4, "precision");
            } else if (attr == "INT8" || attr == "I8") {
                set_once(precision, ov::element::i8, "precision");
            } else {
                OPENVINO_THROW("Unknown attribute '",
                               raw_attr,
                               "' in KV-cache compression config part '",
                               part,
                               "' (expected KEY|VALUE, SYM|ASYM, BY_TOKEN, INT4|INT8)");
            }
        }
        OPENVINO_ASSERT(is_key.has_value(),
                        "KV-cache compression config part '",
                        part,
                        "' does not say which cache it configures (add KEY or VALUE)");
        auto& seen = *is_key ? key_seen : value_seen;
        OPENVINO_ASSERT(!seen,
                        "KV-cache compression config '",
                        config_str,
                        "' configures the ",
                        *is_key ? "KEY" : "VALUE",
                        " cache twice");
        seen = true;

        auto& cache = *is_key ? result.key : result.value;
        cache.type = type.value_or(QuantizationType::Symmetric);
        cache.granularity = granularity.value_or("BY_TOKEN");
        cache.precision = precision;
    }

    auto describe = [](const KVCacheCompressionSetting::Cache& cache) {
        std::string str = cache.type == QuantizationType::Symmetric ? "SYM" : "ASYM";
        str += ":" + cache.granularity;
        if (cache.precision) {
            str += ":" + cache.precision->to_string();
        }
        return str;
    };
    LOG_DEBUG("Parsed KV-cache compression config: KEY:" << describe(result.key)
                                                         << ", VALUE:" << describe(result.value));
    return result;
}

KVCacheCompressionParams resolve_kv_cache_compression(const ov::element::Type& kv_cache_precision,
                                                      const KVCacheCompressionSetting& setting) {
    using QuantizationType = KVCacheCompressionConfig::QuantizationType;
    OPENVINO_ASSERT(kv_cache_precision == ov::element::i8 || kv_cache_precision == ov::element::u8,
                    "KV-cache compression needs an integer KV-cache precision, got ",
                    kv_cache_precision);

    KVCacheCompressionParams params;
    params.key.quantization_dt = setting.key.precision.value_or(kv_cache_precision);
    params.key.quantization_type = setting.key.type;
    params.value.quantization_dt = setting.value.precision.value_or(ov::element::i8);
    params.value.quantization_type = setting.value.type;

    // A u8 key is decomposed only as asymmetric (DynamicQuantizeLinear style, DecomposeDynamicQuantize2);
    // there is no symmetric u8 decomposition, so the KEY quantization type applies to an i8 / i4 key only.
    if (params.key.quantization_dt == ov::element::u8 &&
        params.key.quantization_type != QuantizationType::Asymmetric) {
        LOG_INFO("u8 KV-cache key supports only asymmetric quantization, using KEY:ASYM");
        params.key.quantization_type = QuantizationType::Asymmetric;
    }
    return params;
}

ConvertKVCacheToPrecision::ConvertKVCacheToPrecision(const ov::element::Type lptype,
                                                     bool v_tensors_transposed,
                                                     const std::string& kv_cache_compression_config)
    : m_lp_type(lptype),
      m_v_tensors_transposed(v_tensors_transposed),
      m_kv_cache_compression_config(kv_cache_compression_config) {}

bool ConvertKVCacheToPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto ppp_result =
        cvt_kvcache_to_low_precision(model, m_lp_type, m_v_tensors_transposed, m_kv_cache_compression_config);
    // PrePostProcessor currently always modifies the model in-place and returns the same model pointer, but let's
    // be defensive here and check it just in case
    OPENVINO_ASSERT(ppp_result == model,
                    "PrePostProcessor should not create a new model, but returned a different one.");

    return true;
}

}  // namespace ov::npuw
