// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/utils/ranges.hpp"

#include <map>
#include <queue>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/util/op_types.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"

namespace ov {
namespace test {
namespace utils {

const std::shared_ptr<ov::test::utils::InputGenerateData> ModelRange::get_range_for_param(
    const std::shared_ptr<ov::Node>& node) {
    return node_ranges.at(get_range_id(node));
}

std::string ModelRange::get_range_id(const std::shared_ptr<ov::Node>& node) {
    return node->get_name() + '_' + node->get_element_type().to_string();
}

ov::Tensor ModelRange::generate_input(std::shared_ptr<ov::Node> node, size_t port, const ov::Shape& targetShape) {
    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(node->get_type_info());
    if (it == inputMap.end()) {
        throw std::runtime_error("Couln't find Operation in inputMap: " + std::string(node->get_type_name()));
    }

    std::string range_id = get_range_id(node->get_input_node_shared_ptr(port));
    return it->second(node, port, node->get_input_element_type(port), targetShape, node_ranges[range_id]);
}

void ModelRange::find_mode_ranges(const std::shared_ptr<ov::Model>& model) {
    for (auto param : model->get_parameters()) {
        std::shared_ptr<ov::test::utils::InputGenerateData> data =
            std::make_shared<ov::test::utils::InputGenerateData>(ov::test::utils::rangeByType.get_range(param->get_element_type()));

        bool range_corrected = true;
        std::queue<std::shared_ptr<ov::Node>> queue;
        queue.push(param);
        try {
            while (!queue.empty()) {
                auto node = queue.front();
                queue.pop();

                for (auto& output : node->outputs()) {
                    for (auto& out_target_input : output.get_target_inputs()) {
                        queue.push(out_target_input.get_node()->shared_from_this());
                        auto it = ov::test::utils::inputRanges.find(out_target_input.get_node()->get_type_info());
                        ov::test::utils::InputGenerateData range;
                        if (it != ov::test::utils::inputRanges.end()) {
                            auto ranges = it->second;
                            range = ranges.get_data(out_target_input.get_index(), out_target_input.get_element_type());
                        } else {
                            range = ov::test::utils::rangeByType.get_range(out_target_input.get_element_type());
                        }
                        range_corrected = data->correct_range(range);
                        if (!range_corrected) {
                            throw std::runtime_error("WARNING: range correction is failed for " +
                                                     node->get_friendly_name() +
                                                     ", it looks like we can not find intersection for ranges any "
                                                     "more, so last founded intersection will be used");
                        } else if (range.input_attribute) {
                            throw std::runtime_error(
                                "WARNING: parameter " + node->get_friendly_name() +
                                " is input attribute, propagation is finished and it's range will be used");
                        }
                    }
                }
            }
        } catch (const std::exception& ex) {
            (void)ex;
#ifndef NDEBUG
            std::cout << ex.what() << std::endl;
#endif
        }
#ifndef NDEBUG
        std::cout << "RANGE FOR PARAMETER: " << param->get_friendly_name()
                  << "  start from: " << std::to_string(data->start_from) << "  range: " << std::to_string(data->range)
                  << "  resolution: " << std::to_string(data->resolution) << "  seed: " << std::to_string(data->seed)
                  << std::endl;
#endif

        std::string range_id = get_range_id(param);
        node_ranges[range_id] = data;
    }
}

}  // namespace utils
}  // namespace test
}  // namespace ov
