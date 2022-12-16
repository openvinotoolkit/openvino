// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/lru_cache.hpp"
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

    ASSERT_EQ(ca.get_lru_element(), int());

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
        int idx = input_values.size() - i;
        expected_value.push_back(input_values[idx]);
    }

    int idx = expected_value.size() - 1;
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

    ASSERT_EQ(ca.get_lru_element(), std::shared_ptr<lru_cache_test_data>());
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

    int idx = expected_keys.size() - 1;
    for (auto key : ca.get_all_keys()) {
        ASSERT_EQ(key, expected_keys[idx--]);
    }
}
