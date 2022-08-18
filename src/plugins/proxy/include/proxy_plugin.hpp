// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace ov {
namespace proxy {

void create_plugin(std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin);

/**
 * @brief Restore one fallback order from orders of different plugins
 *
 * @param generated_order the orders from different plugins in format "A->B,C->B,D->B->E"
 *
 * @return the common fallback order (for example: A->C->D->B->E)
 */
std::string restore_order(const std::string& generated_order);

}  // namespace proxy
}  // namespace ov
