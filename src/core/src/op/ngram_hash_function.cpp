// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ngram_hash_function.hpp"

#include <limits>
#include <numeric>

#include "itt.hpp"
#include "ngram_hash_function_shape_inference.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {

NGramHashFunction::NGramHashFunction(const Output<Node>& ngrams,
                                     int64_t ngram_size,
                                     std::vector<int64_t> heads_per_order,
                                     std::vector<int64_t> multipliers,
                                     std::vector<int64_t> head_vocab_sizes,
                                     std::vector<int64_t> head_offsets)
    : Op({ngrams}),
      m_ngram_size(ngram_size),
      m_heads_per_order(std::move(heads_per_order)),
      m_multipliers(std::move(multipliers)),
      m_head_vocab_sizes(std::move(head_vocab_sizes)),
      m_head_offsets(std::move(head_offsets)) {
    constructor_validate_and_infer_types();
}

bool NGramHashFunction::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_NGramHashFunction_visit_attributes);
    visitor.on_attribute("ngram_size", m_ngram_size);
    visitor.on_attribute("heads_per_order", m_heads_per_order);
    visitor.on_attribute("multipliers", m_multipliers);
    visitor.on_attribute("head_vocab_sizes", m_head_vocab_sizes);
    visitor.on_attribute("head_offsets", m_head_offsets);
    return true;
}

size_t NGramHashFunction::get_num_heads() const {
    int64_t heads = 0;
    for (const auto h : m_heads_per_order) {
        heads += std::max<int64_t>(h, 0);
    }
    return static_cast<size_t>(heads);
}

void NGramHashFunction::validate_and_infer_types() {
    OV_OP_SCOPE(internal_NGramHashFunction_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_ngram_size >= 2, "ngram_size must be >= 2. Got: ", m_ngram_size);
    NODE_VALIDATION_CHECK(this,
                          m_multipliers.size() == static_cast<size_t>(m_ngram_size),
                          "multipliers must have ngram_size (",
                          m_ngram_size,
                          ") values. Got: ",
                          m_multipliers.size());
    NODE_VALIDATION_CHECK(this,
                          m_heads_per_order.size() == static_cast<size_t>(m_ngram_size - 1),
                          "heads_per_order must have ngram_size - 1 (",
                          m_ngram_size - 1,
                          ") values. Got: ",
                          m_heads_per_order.size());
    for (const auto h : m_heads_per_order) {
        NODE_VALIDATION_CHECK(this, h >= 0, "heads_per_order values must be non-negative. Got: ", h);
    }
    const auto num_heads = get_num_heads();
    NODE_VALIDATION_CHECK(this, num_heads > 0, "NGramHashFunction must have at least one head.");
    NODE_VALIDATION_CHECK(this,
                          m_head_vocab_sizes.size() == num_heads && m_head_offsets.size() == num_heads,
                          "head_vocab_sizes and head_offsets must have H = sum(heads_per_order) (",
                          num_heads,
                          ") values. Got: ",
                          m_head_vocab_sizes.size(),
                          " and ",
                          m_head_offsets.size());
    constexpr auto int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for (size_t h = 0; h < num_heads; ++h) {
        const auto vocab = m_head_vocab_sizes[h];
        const auto offset = m_head_offsets[h];
        NODE_VALIDATION_CHECK(this, vocab > 0, "head_vocab_sizes values must be positive. Got: ", vocab);
        NODE_VALIDATION_CHECK(this, offset >= 0, "head_offsets values must be non-negative. Got: ", offset);
        NODE_VALIDATION_CHECK(this,
                              vocab <= int32_max && offset <= int32_max - (vocab - 1),
                              "The rows of head ",
                              h,
                              " (head_offsets + head_vocab_sizes - 1) must fit int32.");
    }

    const auto& ngrams_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          ngrams_type.is_dynamic() || ngrams_type == element::i32 || ngrams_type == element::i64,
                          "The element type of the ngrams input must be i32 or i64. Got: ",
                          ngrams_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, element::i32, output_shapes[0]);
}

std::shared_ptr<Node> NGramHashFunction::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_NGramHashFunction_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<NGramHashFunction>(new_args.at(0),
                                               m_ngram_size,
                                               m_heads_per_order,
                                               m_multipliers,
                                               m_head_vocab_sizes,
                                               m_head_offsets);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
