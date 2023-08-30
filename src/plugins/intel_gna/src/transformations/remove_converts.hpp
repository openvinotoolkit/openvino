// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief remove convert layers after inputs and changing it's precision
 * to support preprocessing conversions from user's precision to network presicion
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *         Convert
 *           |
 *        Any layer
 *
 * And transforms to
 *     Any input layer
 *           |
 *        Any layer
 */
const std::vector<ov::element::Type> kSupportedInputTypesFrom = {ov::element::u8, ov::element::i16, ov::element::f32};

const std::vector<ov::element::Type> kSupportedInputTypesTo = {ov::element::u8,
                                                               ov::element::i8,
                                                               ov::element::i16,
                                                               ov::element::i32,
                                                               ov::element::f32};

const std::vector<std::pair<ov::element::Type, ov::element::Type>> kSupportedInputConverts{
    //     FROM      ->       TO
    {ov::element::u8, ov::element::u8},
    {ov::element::u8, ov::element::i8},
    {ov::element::u8, ov::element::i16},
    {ov::element::i16, ov::element::i8},
    {ov::element::i16, ov::element::i16},
    {ov::element::f32, ov::element::i8},
    {ov::element::f32, ov::element::i16},
    {ov::element::f32, ov::element::i32},
    {ov::element::f32, ov::element::f32}};

class RemoveInputConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveInputConvert", "0");
    RemoveInputConvert();
};

/**
 * @brief remove convert layers after before outputs changing it's precision
 * to support postprocessing conversions from network to user's precision
 *
 * Searches for next pattern
 *        Any layer
 *           |
 *         Convert
 *           |
 *     Any output layer
 *
 * And transforms to
 *        Any layer
 *           |
 *    Any output layer
 */

const std::vector<ov::element::Type> kSupportedOutputTypesFrom = {ov::element::f32};

const std::vector<ov::element::Type> kSupportedOutputTypesTo = {ov::element::i32, ov::element::f32};

const std::vector<std::pair<ov::element::Type, ov::element::Type>> kSupportedOutputConverts{
    //     FROM      ->       TO
    {ov::element::f32, ov::element::f32},
    {ov::element::f32, ov::element::i32}};

class RemoveOutputConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveOutputConvert", "0");
    RemoveOutputConvert();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
