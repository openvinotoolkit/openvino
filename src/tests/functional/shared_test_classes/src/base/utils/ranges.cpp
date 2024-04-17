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

#include "shared_test_classes/base/utils/generate_inputs.hpp"

namespace ov {
namespace test {
namespace utils {

ov::test::utils::InputGenerateData get_range_by_type(ov::element::Type temp_type, uint64_t kMaxRange) {
    double min_start = 0 - (int32_t)round(kMaxRange / 2);
    uint32_t max_range = kMaxRange - 1;

    // ov::test::utils::InputGenerateData inData;
    // if (temp_type.is_real()) {
    //     inData.start_from = -1000;
    //     inData.range = 2000;
    //     inData.resolution = 32;
    // } else if (temp_type.is_signed()) {
    //     inData.start_from = -1000;
    //     inData.range = 2000;
    // }

    ov::test::utils::InputGenerateData inData(min_start, max_range);
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
        case ov::element::Type_t::f8e4m3: {
            ov::float8_e4m3 lowest_tmp = std::numeric_limits<ov::float8_e4m3>::lowest();
            ov::float8_e4m3 max_tmp = std::numeric_limits<ov::float8_e4m3>::max();

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
        case ov::element::Type_t::f8e5m2: {
            ov::float8_e5m2 lowest_tmp = std::numeric_limits<ov::float8_e5m2>::lowest();
            ov::float8_e5m2 max_tmp = std::numeric_limits<ov::float8_e5m2>::max();

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

std::string ModelRange::get_range_id(const std::shared_ptr<ov::Node>& node, size_t port) {
    std::string range_id = std::string(node->get_type_name());
    range_id += "_" + TYPE_ALIAS[node->input(port).get_element_type().is_real()];
    range_id += "_" + std::to_string(port);
    return range_id;
}

void ModelRange::collect_ranges(const std::shared_ptr<ov::Model>& function, uint64_t kMaxRange) {
    general_real = std::make_shared<ov::test::utils::InputGenerateData>(ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, kMaxRange));
    general_integral = std::make_shared<ov::test::utils::InputGenerateData>(ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, kMaxRange));

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
            ov::test::utils::InputGenerateData port_range;
            if (ranges.empty()) {
                port_range = ov::test::utils::get_range_by_type(node->input(port).get_element_type(), kMaxRange);
            } else {
                auto op_range = ranges.at(node->input(port).get_element_type().is_real());
                port_range = op_range.size() <= port ? op_range.front() : op_range.at(port);
            }
            node_ranges[range_id] = std::make_shared<ov::test::utils::InputGenerateData>(port_range);
        }
    }
}

void ModelRange::find_general_ranges() {
    bool find_integral = true;
    bool find_real = true;

    for (auto& node_range : node_ranges) {
        if (node_range.second->spetial) {
            continue;
        }

        if (node_range.first.find(TYPE_ALIAS[0]) != std::string::npos) {
            find_integral = general_integral->correct_range(node_range.second);
        } else if (node_range.first.find(TYPE_ALIAS[1]) != std::string::npos) {
            find_real = general_real->correct_range(node_range.second);
        }
    }

    if (find_integral) {
// #ifndef NDEBUG
            std::cout << "INEGRAL RANGE FOUND \n" <<
                         "  start from: " << std::to_string(general_integral->start_from) <<
                         "  range: " << std::to_string(general_integral->range) <<
                         "  resolution: " << std::to_string(general_integral->resolution) <<
                         "  seed: " << std::to_string(general_integral->seed) << std::endl;
// #endif
    } else {
        general_integral = nullptr;
// #ifndef NDEBUG
        std::cout << " RANGE NOT FOUND \n";
// #endif
    }

    if (find_real) {
// #ifndef NDEBUG
            std::cout << "REAL RANGE FOUND \n" <<
                         "  start from: " << std::to_string(general_real->start_from) <<
                         "  range: " << std::to_string(general_real->range) <<
                         "  resolution: " << std::to_string(general_real->resolution) <<
                         "  seed: " << std::to_string(general_real->seed) << std::endl;
// #endif
    } else {
        general_real = nullptr;
// #ifndef NDEBUG
        std::cout << " RANGE NOT FOUND \n";
// #endif
    }
}

ov::Tensor ModelRange::generate_input(std::shared_ptr<ov::Node> node, size_t port, const ov::Shape& targetShape) {
    std::string range_id = get_range_id(node, port);
    if (node_ranges.find(range_id) == node_ranges.end()) {
        throw std::runtime_error("Node info not in rages: " + node->get_friendly_name());
    }

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(node->get_type_info());
    if (it == inputMap.end()) {
        throw std::runtime_error("Couln't find Operation in inputMap: " + std::string(node->get_type_name()));
    }

    std::shared_ptr<ov::test::utils::InputGenerateData> range = node_ranges[range_id];
    if (!node_ranges[range_id]->spetial) {
        if (node->get_input_element_type(port).is_real() && general_real) {
            range = general_real;
        } else {
            range = general_integral;
        }
    }

    return it->second(node, port, node->get_input_element_type(port), targetShape, range);
}

const std::shared_ptr<ov::test::utils::InputGenerateData> ModelRange::get_general_real_range() {
    return general_real;
}

const std::shared_ptr<ov::test::utils::InputGenerateData> ModelRange::get_general_integral_range() {
    return general_integral;
}

} // namespace utils
} // namespace test
} // namespace ov
