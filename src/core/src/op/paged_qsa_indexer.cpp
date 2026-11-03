// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include "itt.hpp"
#include "openvino/op/qsa_indexer.hpp"

namespace ov::op::internal {
namespace {

Dimension dim_at(const PartialShape& shape, size_t axis) {
    return shape.rank().is_static() && shape.size() > axis ? shape[axis] : Dimension::dynamic();
}

QSAIndexer::Config dense_config(const PagedQSAIndexer::Config& config) {
    return {config.key_block_size,
            config.block_topk,
            config.num_selection_heads,
            config.indexer_rotary_dim,
            config.indexer_head_dim,
            config.epsilon,
            config.scale};
}

}  // namespace

PagedQSAIndexer::PagedQSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool PagedQSAIndexer::validate_config(const Config& config, std::string& reason) {
    return QSAIndexer::validate_config(dense_config(config), reason);
}

float PagedQSAIndexer::effective_scale(const Config& config) {
    return QSAIndexer::effective_scale(dense_config(config));
}

size_t PagedQSAIndexer::get_index_query_heads() const {
    const auto& shape = get_input_partial_shape(InputIdx::INDEX_QUERY);
    OPENVINO_ASSERT(m_config.indexer_head_dim > 0 && shape.rank() == 2 && shape[1].is_static(),
                    "PagedQSAIndexer requires a static query width to determine its head count");
    return static_cast<size_t>(shape[1].get_length()) / m_config.indexer_head_dim;
}

size_t PagedQSAIndexer::get_index_key_heads() const {
    const auto& shape = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    OPENVINO_ASSERT(m_config.indexer_head_dim > 0 && shape.rank() == 2 && shape[1].is_static(),
                    "PagedQSAIndexer requires a static key width to determine its head count");
    return static_cast<size_t>(shape[1].get_length()) / m_config.indexer_head_dim;
}

bool PagedQSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedQSAIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("epsilon", m_config.epsilon);
    visitor.on_attribute("scale", m_config.scale);
    return true;
}

void PagedQSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(PagedQSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "PagedQSAIndexer expects 13 inputs");
    std::string reason;
    NODE_VALIDATION_CHECK(this,
                          validate_config(m_config, reason),
                          "PagedQSAIndexer configuration is invalid: ",
                          reason);
    m_config.scale = effective_scale(m_config);

    auto data_type = element::dynamic;
    for (const auto idx : {InputIdx::INDEX_QUERY,
                           InputIdx::RAW_INDEX_KEYS,
                           InputIdx::ROPE_COS,
                           InputIdx::ROPE_SIN,
                           InputIdx::Q_NORM_WEIGHT,
                           InputIdx::K_NORM_WEIGHT,
                           InputIdx::INDEX_KEY_CACHE,
                           InputIdx::INDEX_BLOCK_ROPE_CACHE,
                           InputIdx::BLOCK_INDEX_KEYS_CACHE}) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_type, data_type, get_input_element_type(idx)),
                              "PagedQSAIndexer floating-point inputs must share one element type; input ",
                              idx);
    }
    NODE_VALIDATION_CHECK(
        this,
        data_type.is_dynamic() || data_type == element::f32 || data_type == element::f16 || data_type == element::bf16,
        "PagedQSAIndexer expects f32, f16 or bf16 data");
    for (const auto idx :
         {InputIdx::PAST_LENS, InputIdx::SUBSEQUENCE_BEGINS, InputIdx::BLOCK_INDICES, InputIdx::BLOCK_INDICES_BEGINS}) {
        const auto& type = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              type.is_dynamic() || type == element::i32,
                              "PagedQSAIndexer expects i32 paging metadata; input ",
                              idx);
        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(idx).rank().compatible(1),
                              "PagedQSAIndexer expects rank-1 paging metadata; input ",
                              idx);
    }

    const auto check_shape = [this](size_t idx, const PartialShape& expected) {
        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(idx).compatible(expected),
                              "PagedQSAIndexer input ",
                              idx,
                              " must be compatible with ",
                              expected,
                              ", got ",
                              get_input_partial_shape(idx));
    };
    const auto& query_shape = get_input_partial_shape(InputIdx::INDEX_QUERY);
    const auto& keys_shape = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    auto tokens = Dimension::dynamic();
    for (const auto idx : {InputIdx::INDEX_QUERY, InputIdx::RAW_INDEX_KEYS, InputIdx::ROPE_COS, InputIdx::ROPE_SIN}) {
        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(tokens, tokens, dim_at(get_input_partial_shape(idx), 0)),
                              "PagedQSAIndexer query, key and RoPE token counts must match");
    }
    check_shape(InputIdx::INDEX_QUERY, PartialShape{tokens, -1});
    check_shape(InputIdx::RAW_INDEX_KEYS, PartialShape{tokens, -1});
    const auto heads = [this](const PartialShape& shape) {
        const auto width = dim_at(shape, 1);
        if (width.is_dynamic()) {
            return Dimension::dynamic();
        }
        NODE_VALIDATION_CHECK(this,
                              width.get_length() > 0 && width.get_length() % m_config.indexer_head_dim == 0,
                              "PagedQSAIndexer projection width must be a positive multiple of indexer_head_dim");
        return Dimension(width.get_length() / m_config.indexer_head_dim);
    };
    const auto query_heads = heads(query_shape);
    auto key_heads = heads(keys_shape);
    for (const auto idx : {InputIdx::INDEX_KEY_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE}) {
        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(key_heads, key_heads, dim_at(get_input_partial_shape(idx), 1)),
                              "PagedQSAIndexer keys and caches must share the key head count");
    }
    NODE_VALIDATION_CHECK(this,
                          key_heads.is_dynamic() || key_heads.get_length() > 0,
                          "PagedQSAIndexer key head count must be positive");
    const Dimension head_dim(m_config.indexer_head_dim);
    const Dimension rotary_dim(m_config.indexer_rotary_dim);
    check_shape(InputIdx::ROPE_COS, PartialShape{tokens, rotary_dim});
    check_shape(InputIdx::ROPE_SIN, PartialShape{tokens, rotary_dim});
    check_shape(InputIdx::Q_NORM_WEIGHT, PartialShape{head_dim});
    check_shape(InputIdx::K_NORM_WEIGHT, PartialShape{head_dim});

    auto pages = Dimension::dynamic();
    for (const auto idx :
         {InputIdx::INDEX_KEY_CACHE, InputIdx::INDEX_BLOCK_ROPE_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE}) {
        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(pages, pages, dim_at(get_input_partial_shape(idx), 0)),
                              "PagedQSAIndexer caches must share the physical page pool");
    }
    auto page_size = dim_at(get_input_partial_shape(InputIdx::INDEX_KEY_CACHE), 2);
    auto blocks_per_page = Dimension::dynamic();
    if (page_size.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              page_size.get_length() > 0 && page_size.get_length() % m_config.key_block_size == 0,
                              "PagedQSAIndexer page_size must be a positive multiple of key_block_size");
        blocks_per_page = Dimension(page_size.get_length() / m_config.key_block_size);
    }
    for (const auto& [idx, axis] : {std::pair{InputIdx::INDEX_BLOCK_ROPE_CACHE, size_t{1}},
                                    std::pair{InputIdx::BLOCK_INDEX_KEYS_CACHE, size_t{2}}}) {
        NODE_VALIDATION_CHECK(
            this,
            Dimension::merge(blocks_per_page, blocks_per_page, dim_at(get_input_partial_shape(idx), axis)),
            "PagedQSAIndexer summary caches must share the number of blocks per page");
    }
    NODE_VALIDATION_CHECK(this,
                          blocks_per_page.is_dynamic() || blocks_per_page.get_length() > 0,
                          "PagedQSAIndexer must have a positive number of blocks per page");
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(page_size, page_size, blocks_per_page * Dimension(m_config.key_block_size)),
                          "PagedQSAIndexer cache page sizes must agree");
    check_shape(InputIdx::INDEX_KEY_CACHE, PartialShape{pages, key_heads, page_size, head_dim});
    check_shape(InputIdx::INDEX_BLOCK_ROPE_CACHE,
                PartialShape{pages, blocks_per_page, Dimension(2 * m_config.indexer_rotary_dim)});
    check_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE, PartialShape{pages, key_heads, blocks_per_page, head_dim});
    const auto sequences = dim_at(get_input_partial_shape(InputIdx::PAST_LENS), 0);
    auto boundaries = sequences + 1;
    for (const auto idx : {InputIdx::SUBSEQUENCE_BEGINS, InputIdx::BLOCK_INDICES_BEGINS}) {
        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(boundaries, boundaries, dim_at(get_input_partial_shape(idx), 0)),
                              "PagedQSAIndexer paging boundary lengths must match");
    }
    check_shape(InputIdx::SUBSEQUENCE_BEGINS, PartialShape{boundaries});
    check_shape(InputIdx::BLOCK_INDICES_BEGINS, PartialShape{boundaries});
    if (query_heads.is_static() && key_heads.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              query_heads.get_length() % key_heads.get_length() == 0,
                              "PagedQSAIndexer query heads must be a multiple of key heads");
    }
    if (key_heads.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              m_config.num_selection_heads == 1 ||
                                  m_config.num_selection_heads == static_cast<size_t>(key_heads.get_length()),
                              "PagedQSAIndexer num_selection_heads must be 1 or the key head count");
    }
    const Dimension selection_heads(m_config.num_selection_heads);
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES,
                    element::i32,
                    PartialShape{tokens, selection_heads, Dimension(m_config.block_topk)});
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, PartialShape{tokens, selection_heads});
}

std::shared_ptr<Node> PagedQSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PagedQSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
