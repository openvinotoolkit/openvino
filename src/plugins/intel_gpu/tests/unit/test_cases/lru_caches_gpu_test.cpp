// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"

#include "intel_gpu/runtime/lru_cache.hpp"
#include "shape_of_inst.h"

#include <vector>

using namespace cldnn;
using namespace ::tests;

TEST(lru_cache, basic_data_type)
{
    const size_t cap = 4;
    LruCache<int, int> ca(cap * sizeof(int));

    std::vector<int> inputs = {1, 2, 3, 4, 2, 1, 5};
    std::vector<std::pair<int, int>> input_values;
    for (auto i :  inputs) {
        input_values.push_back(std::make_pair(i, i + 10));
    }

    ASSERT_EQ(ca.get_lru_element().second, int());

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, false};
    for (size_t i = 0; i < input_values.size(); i++) {
        auto& in = input_values[i];
        int data = 0;
        bool hitted = ca.has(in.first);
        if (hitted) {
            data = ca.get(in.first);
        } else {
            ca.add(in.first, in.second);
            data = ca.get(in.first);
        }
        ASSERT_EQ(data, in.second);
        ASSERT_EQ(hitted, (bool)expected_hitted[i]);
    }

    std::vector<std::pair<int, int>> expected_value;
    for (size_t i = ca.size(); i > 0; i--) {  // 5, 1, 2, 4
        auto idx = input_values.size() - i;
        expected_value.push_back(input_values[idx]);
    }

    auto idx = expected_value.size() - 1;
    for (auto key : ca.get_all_keys()) {
        ASSERT_EQ(key, expected_value[idx--].first);
    }
}

class lru_cache_test_data {
public:
    lru_cache_test_data(int a, int b, int c) : x(a), y(b), z(c) {
        key = "key_" + std::to_string(a) + "_" + std::to_string(b) + "_" + std::to_string(c);
    }

    bool operator==(const lru_cache_test_data&rhs) {
        return (this->x == rhs.x && this->y == rhs.y && this->z == rhs.z);
    }

    bool operator!=(const lru_cache_test_data&rhs) {
        return (this->x != rhs.x || this->y != rhs.y || this->z != rhs.z);
    }

    operator std::string() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
    }

    std::string key;
    int x;
    int y;
    int z;

};

using test_impl_cache = LruCache<std::string, std::shared_ptr<lru_cache_test_data>>;

TEST(lru_cache, custom_data_type) {
    const size_t cap = 4;
    test_impl_cache ca(cap);

    std::vector<std::shared_ptr<lru_cache_test_data>> inputs;
    inputs.push_back(std::make_shared<lru_cache_test_data>(1, 21, 11));
    inputs.push_back(std::make_shared<lru_cache_test_data>(2, 22, 12));
    inputs.push_back(std::make_shared<lru_cache_test_data>(3, 23, 13));
    inputs.push_back(std::make_shared<lru_cache_test_data>(4, 24, 14));
    inputs.push_back(std::make_shared<lru_cache_test_data>(2, 22, 12));
    inputs.push_back(std::make_shared<lru_cache_test_data>(1, 21, 11));
    inputs.push_back(std::make_shared<lru_cache_test_data>(3, 23, 13));
    inputs.push_back(std::make_shared<lru_cache_test_data>(5, 25, 15));

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, true, false};

    ASSERT_EQ(ca.get_lru_element().second, std::shared_ptr<lru_cache_test_data>());
    for (size_t i = 0; i < inputs.size(); i++) {
        auto& in = inputs[i];
        std::shared_ptr<lru_cache_test_data> p_data;
        bool hitted = ca.has(in->key);
        if (hitted) {
            p_data = ca.get(in->key);
        } else {
            ca.add(in->key, in);
            p_data = ca.get(in->key);
        }
        ASSERT_EQ(p_data->key, in->key);
        ASSERT_EQ(hitted, (bool)expected_hitted[i]);
    }

    ASSERT_EQ(cap, ca.size());

    std::vector<std::string> expected_keys;
    for (size_t i = cap; i > 0; i--) {
        expected_keys.push_back(inputs[inputs.size() - i]->key);
    }

    auto idx = expected_keys.size() - 1;
    for (auto key : ca.get_all_keys()) {
        ASSERT_EQ(key, expected_keys[idx--]);
    }
}

namespace {
struct ImplHasher {
    size_t operator()(const kernel_impl_params &k) const {
        return k.hash();
    }
};
}  // namespace

TEST(lru_cache, collisions) {
    auto l1 = layout{{1, 3, 27, 92}, data_types::f32, format::bfyx};
    auto l2 = layout{{1, 3, 28, 29}, data_types::f32, format::bfyx};

    auto input1_prim = std::make_shared<input_layout>("input1", l1);
    auto input2_prim = std::make_shared<input_layout>("input2", l2);
    auto shape_of1_prim = std::make_shared<shape_of>("shape_of1", input_info("input1"), data_types::i64);
    auto shape_of2_prim = std::make_shared<shape_of>("shape_of2", input_info("input2"), data_types::i64);

    using ImplementationsCache = cldnn::LruCacheThreadSafe<kernel_impl_params, std::shared_ptr<primitive_impl>, ImplHasher>;
    ImplementationsCache cache(0);

    program prog(get_test_engine());
    auto& input1_node = prog.get_or_create(input1_prim);
    auto& input2_node = prog.get_or_create(input2_prim);
    auto& shape_of1_node = prog.get_or_create(shape_of1_prim);
    auto& shape_of2_node = prog.get_or_create(shape_of2_prim);
    program_wrapper::add_connection(prog, input1_node, shape_of1_node);
    program_wrapper::add_connection(prog, input2_node, shape_of2_node);

    auto params1 = *shape_of1_node.get_kernel_impl_params();
    auto params2 = *shape_of1_node.get_kernel_impl_params();

    auto out_layouts1 = shape_of_inst::calc_output_layouts<ov::PartialShape>(shape_of1_node, params1);
    auto out_layouts2 = shape_of_inst::calc_output_layouts<ov::PartialShape>(shape_of2_node, params2);

    shape_of1_node.set_output_layouts(out_layouts1);
    shape_of2_node.set_output_layouts(out_layouts2);

    shape_of1_node.set_preferred_impl_type(impl_types::ocl);
    shape_of2_node.set_preferred_impl_type(impl_types::ocl);

    auto impl1 = shape_of1_node.type()->create_impl(shape_of1_node);
    auto impl2 = shape_of2_node.type()->create_impl(shape_of2_node);

    // Ensure that hashes for primitive, input layouts and full impl params are same due to collision
    ASSERT_EQ(shape_of1_prim->hash(), shape_of2_prim->hash());
    ASSERT_EQ(l1.hash(), l2.hash());
    ASSERT_EQ(shape_of1_node.get_kernel_impl_params()->hash(), shape_of2_node.get_kernel_impl_params()->hash());
    ASSERT_FALSE(shape_of1_node.get_kernel_impl_params() == shape_of2_node.get_kernel_impl_params());

    cache.add(*shape_of1_node.get_kernel_impl_params(), impl1->clone());
    cache.add(*shape_of2_node.get_kernel_impl_params(), impl2->clone());

    // But cache still contains both entries, as input layouts are differenet - thus kernels are different
    ASSERT_EQ(cache.size(), 2);
}
