// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "config.h"
#include "openvino/core/model.hpp"
#include "openvino/op/pa_kv_reorder.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "transformation_pipeline.h"

namespace ov::intel_cpu {

TEST(PaKVReorderPrecisionTest, KeepsF16CacheInputs) {
    auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, Shape{4, 8, 32, 64});
    auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, Shape{4, 8, 32, 64});
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    auto pa_kv_reorder = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                     value_cache,
                                                                     block_indices,
                                                                     block_indices_begins,
                                                                     block_update_indices,
                                                                     block_update_indices_begins);
    auto model = std::make_shared<Model>(OutputVector{pa_kv_reorder},
                                         ParameterVector{key_cache,
                                                         value_cache,
                                                         block_indices,
                                                         block_indices_begins,
                                                         block_update_indices,
                                                         block_update_indices_begins});

    Config config;
    Transformations transformations(model, config);
    transformations.UpToLpt();

    EXPECT_EQ(pa_kv_reorder->get_input_element_type(0), element::f16);
    EXPECT_EQ(pa_kv_reorder->get_input_element_type(1), element::f16);
    EXPECT_EQ(pa_kv_reorder->input_value(0).get_node_shared_ptr(), key_cache);
    EXPECT_EQ(pa_kv_reorder->input_value(1).get_node_shared_ptr(), value_cache);
}

}  // namespace ov::intel_cpu