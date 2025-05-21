// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

class ModelGenerator {
public:
    ModelGenerator() = default;

    std::shared_ptr<ov::Model> get_model_with_one_op();
    std::shared_ptr<ov::Model> get_model_without_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks();

private:
    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

private:
    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    size_t m_name_idx;
};
