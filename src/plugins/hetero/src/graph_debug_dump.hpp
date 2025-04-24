// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"

namespace ov {
namespace hetero {
namespace debug {

void dump_affinities(const std::shared_ptr<ov::Model>& model,
                     const std::map<std::string, std::string>& supported_ops_map,
                     const std::unordered_set<std::string>& devices);
void dump_subgraphs(const std::shared_ptr<ov::Model>& model,
                    const std::map<std::string, std::string>& supported_ops_map,
                    const std::map<std::string, int>& map_id);

}  // namespace debug
}  // namespace hetero
}  // namespace ov
