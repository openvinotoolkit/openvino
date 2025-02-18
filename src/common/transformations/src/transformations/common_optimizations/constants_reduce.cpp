// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/constants_reduce.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/util/log.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {

ConstantsReduce::ConstantsReduce() {}

bool ConstantsReduce::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(ConstantsReduce);

    using BlobCacheKey = std::tuple<const char*, ov::Shape, ov::element::Type>;
    std::map<BlobCacheKey, std::shared_ptr<ov::Node>> blobMemCache;

    int copies = 0;

    const std::vector<std::shared_ptr<ov::Node>> ops = m->get_ops();
    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Constant>(op)) continue;
        auto const_node = ov::as_type_ptr<op::v0::Constant>(op);

        auto data = const_node->get_data_ptr<char>();
        auto const_shape = const_node->get_shape();

        const auto cache_key = std::make_tuple(data, const_shape, const_node->get_output_element_type(0));
        auto bufIter = blobMemCache.find(cache_key);

        if (bufIter == blobMemCache.end()) {
            blobMemCache[cache_key] = op;
        } else {
            auto users = const_node->get_users();
            copies++;
            for(auto user : users) {
                for(size_t i = 0; i < user->get_input_size(); i++) {
                    if(user->input_value(i) == op->output(0)) {
                        user->input(i).replace_source_output(blobMemCache[cache_key]);
                    }
                }
            }
        }
    }

    OPENVINO_DEBUG("Reduced ", copies, " constant node duplications from model");

    return true;
}

}  // namespace pass
}  // namespace ov
