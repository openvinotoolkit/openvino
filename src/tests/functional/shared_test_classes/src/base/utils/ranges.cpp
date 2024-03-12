// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <vector>

#include "gtest/gtest.h"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
namespace utils {

ov::test::utils::InputGenerateData get_range_by_type(ov::element::Type temp_type, uint64_t kMaxRange) {
    double min_start = 0 - (int32_t)round(kMaxRange / 2);
    uint32_t max_range = kMaxRange - 1;

    ov::test::utils::InputGenerateData inData;
    #define CASE(X)                                                                                                 \
    case X: {                                                                                                       \
        auto lowest = std::numeric_limits<element_type_traits<X>::value_type>::lowest();                            \
        auto max = std::numeric_limits<element_type_traits<X>::value_type>::max();                                  \
        double tmp_range = static_cast<double>(max) - static_cast<double>(lowest);                                  \
        if (tmp_range < kMaxRange) {                                                                                \
            inData.start_from = lowest;                                                                             \
            inData.range = (uint32_t)round(tmp_range);                                                              \
        } else {                                                                                                    \
            inData.range = kMaxRange - 1;                                                                           \
            inData.start_from = lowest > min_start ? static_cast<double>(lowest) : min_start;                       \
        }                                                                                                           \
        break;                                                                                                      \
    }                                                                                                               \

    switch (temp_type) {
        case(ov::element::Type_t::undefined): {
            inData.start_from = min_start;
            inData.range = max_range;
            break;
        }
        case(ov::element::Type_t::boolean): {
            inData.start_from = 0;
            inData.range = 1;
            break;
        }
        case(ov::element::Type_t::bf16): {
            ov::bfloat16 lowest_tmp = std::numeric_limits<ov::bfloat16>::lowest();
            ov::bfloat16 max_tmp = std::numeric_limits<ov::bfloat16>::max();

            double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
            double max = max_tmp.to_bits();

            double tmp_range = max - lowest;
            if (tmp_range < kMaxRange) {
                inData.start_from = lowest;
                inData.range = (uint32_t)round(tmp_range);
            } else {
                inData.start_from = lowest > min_start ? lowest : min_start;
                inData.range = kMaxRange - 1;
            }

            break;
        }
        case ov::element::Type_t::f16: {
            ov::float16 lowest_tmp = std::numeric_limits<ov::float16>::lowest();
            ov::float16 max_tmp = std::numeric_limits<ov::float16>::max();

            double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
            double max = max_tmp.to_bits();

            double tmp_range = max - lowest;
            if (tmp_range < kMaxRange) {
                inData.start_from = lowest;
                inData.range = (uint32_t)round(tmp_range);
            } else {
                inData.start_from = lowest > min_start ? lowest : min_start;
                inData.range = kMaxRange - 1;
            }

            break;
        }
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
        CASE(ov::element::Type_t::i4)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u1)
        CASE(ov::element::Type_t::u4)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        break;
    }

    return inData;
}

std::string get_range_id(const std::shared_ptr<ov::Node>& node, size_t port, bool spectial) {
    std::string range_id = std::to_string(node->input(port).get_element_type().is_real());
    if (spectial) {
        range_id += '_' + std::string(node->get_type_name()) + '_' + std::to_string(port);
    }
    return range_id;
}

std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>> collect_ranges(const std::shared_ptr<ov::Model>& function, uint64_t kMaxRange) {
    bool success = true;
    std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>> inputDataMap;
    inputDataMap["0"] = std::make_shared<ov::test::utils::InputGenerateData>(ov::test::utils::get_range_by_type(ov::element::Type_t::undefined,
                                                                                                                kMaxRange));
    inputDataMap["1"] = std::make_shared<ov::test::utils::InputGenerateData>(ov::test::utils::get_range_by_type(ov::element::Type_t::undefined,
                                                                                                                kMaxRange));
    for (const auto& node : function->get_ordered_ops()) {
        if (ov::op::util::is_output(node) ||
            ov::op::util::is_constant(node) ||
            ov::op::util::is_parameter(node)) {
            continue;
        }

        auto it = ov::test::utils::inputRanges.find(node->get_type_info());
        auto ranges = std::vector<std::vector<ov::test::utils::InputGenerateData>>{};
        if (it != ov::test::utils::inputRanges.end()) {
            ranges = it->second;
            if (ranges.size() != 2) {
                throw std::runtime_error("Incorrect size of ranges. It should be 2 (real and int cases)");
            }
        }

        const size_t inNodeCnt = node->get_input_size();
        for (size_t port = 0; port < inNodeCnt; ++port) {
            if (ov::op::util::is_constant(node->get_input_node_ptr(port))) {
                continue;
            }

            std::string range_id = get_range_id(node, port);
            ov::test::utils::InputGenerateData temp_range;
            if (ranges.empty()) {
                temp_range = ov::test::utils::get_range_by_type(node->input(port).get_element_type(), kMaxRange);
                if (temp_range.start_from == inputDataMap[range_id]->start_from && temp_range.range == inputDataMap[range_id]->range) {
                    continue;
                }
            } else {
                auto op_range = ranges.at(node->input(port).get_element_type().is_real());
                temp_range = op_range.size() <= port ? op_range.front() : op_range.at(port);
                if (temp_range.spetial) {
                    range_id = get_range_id(node, port, true);
                    inputDataMap[range_id] = std::make_shared<ov::test::utils::InputGenerateData>(temp_range);
                    continue;
                }
            }

            success = inputDataMap[range_id]->correct_range(node, port, temp_range);
            if (!success)
                break;
        }

        if (!success)
            break;
    }

#ifndef NDEBUG
    if (success) {
        for (auto& range : inputDataMap) {
            std::cout << "RANGE " << range.first << " :\n" <<
                         "  start from: " << std::to_string(range.second->start_from) <<
                         "  range: " << std::to_string(range.second->range) <<
                         "  resolution: " << std::to_string(range.second->resolution) <<
                         "  seed: " << std::to_string(range.second->resolution) << std::endl;
        }
    } else {
        std::cout << " RANGE NOT FOUND \n";
    }
#endif
    return success ? inputDataMap : std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>>{};
}


} // namespace utils
} // namespace test
} // namespace ov
