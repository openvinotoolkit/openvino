// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/type_ranges.hpp"
#include "openvino/core/node.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace ov {
namespace test {
namespace utils {

// NOTE: Default ranges are collected by data type and have resolution 1(for real types too)
// to set up correct ranges and resolutions, please, configure range for Op in input_ranges structure
struct Range {
    std::vector<ov::test::utils::InputGenerateData> int_port_ranges;
    std::vector<ov::test::utils::InputGenerateData> real_port_ranges;

    Range(const std::vector<ov::test::utils::InputGenerateData>& int_ranges = {},
          const std::vector<ov::test::utils::InputGenerateData>& real_ranges = {})
        : int_port_ranges(int_ranges),
          real_port_ranges(real_ranges) {
        size_t max_known_port = std::max(real_port_ranges.size(), int_port_ranges.size());
        max_known_port = std::max(static_cast<int>(max_known_port), 1);
        for (size_t port = 0; port < max_known_port; port++) {
            std::map<ov::element::Type, ov::test::utils::InputGenerateData> type_map;
            for (const auto& type : get_known_types()) {
                ov::test::utils::InputGenerateData new_range = rangeByType.get_range(type);
                if (type.is_real() && port < real_port_ranges.size()) {
                    new_range.correct_range(real_port_ranges.at(port));
                    new_range.input_attribute = real_port_ranges.at(port).input_attribute;
                } else if (type.is_integral() && port < int_port_ranges.size()) {
                    new_range.correct_range(int_port_ranges.at(port));
                    new_range.input_attribute = int_port_ranges.at(port).input_attribute;
                }
                type_map[type] = new_range;
            }
            data.push_back(type_map);
        }
    }

    std::vector<std::map<ov::element::Type, ov::test::utils::InputGenerateData>> data;

    ov::test::utils::InputGenerateData get_data(size_t port, ov::element::Type type) {
        if (port < data.size()) {
            return data.at(port).at(type);
        } else {
            return data.at(0).at(type);
        }
    }
};

const std::map<ov::NodeTypeInfo, Range>& get_input_ranges();

class ModelRange {
    // key for map calculated in get_range_id and contais [Parameter Name]_[parameter type]
    std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>> node_ranges;

public:
    void find_mode_ranges(const std::shared_ptr<ov::Model>& function);
    std::string get_range_id(const std::shared_ptr<ov::Node>& node);
    ov::Tensor generate_input(std::shared_ptr<ov::Node> node, size_t port, const ov::Shape& targetShape);

    const std::shared_ptr<ov::test::utils::InputGenerateData> get_range_for_param(
        const std::shared_ptr<ov::Node>& node);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
