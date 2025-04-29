// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/constants_reduce.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/log.hpp"

#define LARGE_TENSOR_BYTE_SIZE 64

namespace ov::pass {

using BlobCacheKey = std::shared_ptr<ov::Node>;

struct KeyHash {
    std::size_t operator()(const BlobCacheKey& key) const {
        std::size_t hash = 0;

        auto node = ov::as_type_ptr<op::v0::Constant>(key);

        auto type = node->get_output_element_type(0);
        auto shape = node->get_shape();
        std::size_t size = node->get_byte_size();
        const char* data = node->get_data_ptr<char>();

        for (auto dim : shape) {
            hash ^= std::hash<size_t>{}(dim);
        }

        for (std::size_t i = 0; i < size; i++) {
            hash ^= ((hash << 5) + hash) + data[i];
        }

        hash ^= type.hash();
        hash ^= size;

        return hash;
    }
};

struct KeyEqual {
    bool operator()(const BlobCacheKey& lhs, const BlobCacheKey& rhs) const {
        auto lhs_node = ov::as_type_ptr<op::v0::Constant>(lhs);
        auto rhs_node = ov::as_type_ptr<op::v0::Constant>(rhs);

        auto lhs_type = lhs_node->get_output_element_type(0);
        auto rhs_type = rhs_node->get_output_element_type(0);

        if (lhs_type != rhs_type)
            return false;

        auto lhs_shape = lhs_node->get_shape();
        auto rhs_shape = rhs_node->get_shape();

        if (lhs_shape != rhs_shape)
            return false;

        std::size_t lhs_size = lhs_node->get_byte_size();
        std::size_t rhs_size = rhs_node->get_byte_size();

        if (lhs_size != rhs_size)
            return false;

        // Retrieve buffer pointers
        const char* lhs_data = lhs_node->get_data_ptr<char>();
        const char* rhs_data = rhs_node->get_data_ptr<char>();

        if (lhs_data == rhs_data)
            return true;

        return std::memcmp(lhs_data, rhs_data, lhs_size) == 0;
    }
};

bool ConstantsReduce::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(ConstantsReduce);

    std::unordered_map<BlobCacheKey, std::shared_ptr<ov::Node>, KeyHash, KeyEqual> blobMemCache;

    const auto& ops = m->get_ops();

    unsigned int copies = 0;

    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Constant>(op))
            continue;

        auto const_node = ov::as_type_ptr<op::v0::Constant>(op);

        // Limit size of node reading to avoid reading large tensors
        if (const_node->get_byte_size() > LARGE_TENSOR_BYTE_SIZE)
            continue;

        const auto cache_key = op;
        auto bufIter = blobMemCache.find(cache_key);

        if (bufIter == blobMemCache.end()) {
            blobMemCache[cache_key] = op;
        } else {
            copies++;
            auto users = const_node->get_users();
            for (auto user : users) {
                for (size_t i = 0; i < user->get_input_size(); i++) {
                    if (user->input_value(i) == op->output(0)) {
                        user->input(i).replace_source_output(blobMemCache[cache_key]);
                    }
                }
            }
        }
    }
    OPENVINO_DEBUG("Reduced ", copies, " constant node duplications from model");

    // Return true if we have made any replacements
    return copies > 0;
}

}  // namespace ov::pass
