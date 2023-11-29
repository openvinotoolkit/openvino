// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> makeConvPoolRelu(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                            ov::element::Type_t ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeConvPool2Relu2(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                              ov::element::Type_t ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeConvPoolReluNonZero(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                   ov::element::Type_t ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                               ov::element::Type_t ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeKSOFunction(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                           ov::element::Type_t ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeNestedBranchConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                      ov::element::Type ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeNestedSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                     ov::element::Type ngPrc = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeConvBias(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                        ov::element::Type type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeReadConcatSplitAssign(std::vector<size_t> inputShape = {1, 1, 2, 4},
                                                     ov::element::Type type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeMatMulBias(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                          ov::element::Type type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeConvertTranspose(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                std::vector<size_t> inputOrder = {0, 1, 2, 3},
                                                ov::element::Type type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeMultipleInputOutputReLU(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                       ov::element::Type_t type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeMultipleInputOutputDoubleConcat(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                               ov::element::Type_t type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeSingleConcatWithConstant(std::vector<size_t> inputShape = {1, 1, 2, 4},
                                                        ov::element::Type type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeConcatWithParams(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                ov::element::Type_t type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeSingleSplit(std::vector<size_t> inputShape = {1, 4, 32, 32},
                                           ov::element::Type_t type = ov::element::Type_t::f32);

std::shared_ptr<ov::Model> makeSplitConcat(std::vector<size_t> inputShape = {1, 4, 24, 24},
                                           ov::element::Type_t type = ov::element::Type_t::f32);

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
