// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/make_stateful.hpp"

namespace tests {

inline std::shared_ptr<ov::Model> make_llm_kv_cache_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                            ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                            ov::Dimension n_features = ov::Dimension::dynamic(),
                                                            ov::element::Type_t element_type = ov::element::f32,
                                                            bool stateful = false) {
    ov::PartialShape kv_cache_size = {batch, n_heads, -1, n_features};
    ov::PartialShape new_token_size = {batch, -1, n_heads, n_features};
    ov::PartialShape matmul_in_size = {batch, n_heads, -1, -1};

    auto in_kv_prev = std::make_shared<ov::op::v0::Parameter>(element_type, kv_cache_size);
    in_kv_prev->set_friendly_name("past_key_values");
    auto in_new_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);
    in_new_token->set_friendly_name("new_token_input");
    auto in_matmul = std::make_shared<ov::op::v0::Parameter>(element_type, matmul_in_size);
    in_matmul->set_friendly_name("in_matmul");

    auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {new_token_size.size()}, {0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(in_new_token, transpose_const);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{in_kv_prev, transpose}, 2);
    auto convert = std::make_shared<ov::op::v0::Convert>(concat, element_type);
    auto kv_present = std::make_shared<ov::op::v0::Result>(convert);
    kv_present->set_friendly_name("present_key_values");
    auto matmul = std::make_shared<ov::op::v0::MatMul>(in_matmul, concat, false, false);
    auto matmul_out = std::make_shared<ov::op::v0::Result>(matmul);
    matmul_out->set_friendly_name("matmul_out");

    ov::ParameterVector params{in_kv_prev, in_new_token, in_matmul};
    ov::ResultVector results{kv_present, matmul_out};
    auto model = std::make_shared<ov::Model>(results, params, "LLM-KV-Cache");
    if (stateful) {
        ov::pass::MakeStateful({{in_kv_prev, kv_present}}).run_on_model(model);
    }

    return model;
}

} // namespace tests
