// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_paged_qsa_indexer_inputs.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "openvino/op/convert.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov::intel_gpu {

ConvertPagedQSAIndexerInputs::ConvertPagedQSAIndexerInputs(size_t page_size) {
    using namespace ov::pass::pattern;
    using Indexer = ov::op::internal::PagedQSAIndexer;
    using In = Indexer::InputIdx;

    const auto p_indexer = wrap_type<Indexer>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto indexer = ov::as_type_ptr<Indexer>(m.get_match_root());
        if (!indexer || transformation_callback(indexer)) {
            return false;
        }
        const auto& raw_keys = indexer->get_input_partial_shape(In::RAW_INDEX_KEYS);
        const auto& config = indexer->get_config();
        if (raw_keys.rank().is_dynamic() || raw_keys.size() != 2 || raw_keys[1].is_dynamic() || config.indexer_head_dim == 0 || config.key_block_size == 0) {
            return false;
        }

        // The page size an index_key_cache declares, else the paged-attention block size.
        size_t page = page_size;
        const auto& key_cache_shape = indexer->get_input_partial_shape(In::INDEX_KEY_CACHE);
        if (key_cache_shape.rank().is_static() && key_cache_shape.size() == 4 && key_cache_shape[2].is_static()) {
            page = static_cast<size_t>(key_cache_shape[2].get_length());
        }
        if (page == 0 || page % config.key_block_size != 0) {
            return false;
        }

        const auto heads = raw_keys[1].get_length() / static_cast<int64_t>(config.indexer_head_dim);
        const auto head_dim = static_cast<int64_t>(config.indexer_head_dim);
        const auto summaries = static_cast<int64_t>(page / config.key_block_size);
        const std::pair<size_t, ov::PartialShape> layouts[] = {
            {In::INDEX_KEY_CACHE, {-1, heads, static_cast<int64_t>(page), head_dim}},
            {In::INDEX_BLOCK_ROPE_CACHE, {-1, summaries, static_cast<int64_t>(2 * config.indexer_rotary_dim)}},
            {In::BLOCK_INDEX_KEYS_CACHE, {-1, heads, summaries, head_dim}},
        };
        bool changed = false;
        for (const auto& [idx, shape] : layouts) {
            // ConvertPrecision keeps the precision of a keep_const_precision cache by converting it in front of the
            // indexer: the kernels would then update the converted copy, so the cache itself is made f16 instead.
            auto source = indexer->get_input_node_shared_ptr(idx);
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source);
            if (convert && convert->get_output_target_inputs(0).size() == 1) {
                source = convert->get_input_node_shared_ptr(0);
            }
            const auto cache = ov::as_type_ptr<ov::op::v0::Parameter>(source);
            if (!cache || cache->get_output_target_inputs(0).size() != 1) {
                continue;
            }
            if (convert && convert->get_input_node_shared_ptr(0) == cache) {
                indexer->input(idx).replace_source_output(cache->output(0));
                changed = true;
            }
            // Already laid out: f16 with static dims (but the block count) matching the layout.
            const auto& current = cache->get_partial_shape();
            const bool laid_out = cache->get_element_type() == ov::element::f16 && current.rank() == shape.rank() && current.compatible(shape) &&
                                  std::all_of(current.begin() + 1, current.end(), [](const ov::Dimension& d) {
                                      return d.is_static();
                                  });
            if (laid_out) {
                continue;
            }
            cache->set_element_type(ov::element::f16);
            cache->set_partial_shape(shape);
            ov::enable_keep_const_precision(cache);
            cache->validate_and_infer_types();
            changed = true;
        }
        if (changed) {
            indexer->validate_and_infer_types();
        }
        return changed;
    };

    register_matcher(std::make_shared<Matcher>(p_indexer, "ConvertPagedQSAIndexerInputs"), callback);
}

}  // namespace ov::intel_gpu
