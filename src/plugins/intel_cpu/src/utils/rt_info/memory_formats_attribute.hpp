// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov::intel_cpu {

constexpr const char* InputMemoryFormatsAttr = "InputMemoryFormats";
constexpr const char* OutputMemoryFormatsAttr = "OutputMemoryFormats";

template <typename MemoryFormat>
class MemoryFormats : public ov::RuntimeAttribute {
protected:
    std::string memory_format;

public:
    MemoryFormats() = default;
    explicit MemoryFormats(std::string _memory_format) : memory_format(std::move(_memory_format)) {}
    [[nodiscard]] std::string to_string() const override {
        return memory_format;
    };
    [[nodiscard]] bool is_copyable(const std::shared_ptr<ov::Node>& to) const override {
        return (!ov::op::util::is_constant(to));
    }

    [[nodiscard]] ov::Any merge(const ov::NodeVector& nodes) const override {
        std::set<std::string> unique_mem_format;

        for (auto& node : nodes) {
            auto it_info = node->get_rt_info().find(MemoryFormat::get_type_info_static());
            if (it_info != node->get_rt_info().end()) {
                std::string mem_format = it_info->second.template as<MemoryFormat>().to_string();
                if (!mem_format.empty()) {
                    unique_mem_format.insert(mem_format);
                }
            }
        }

        if (unique_mem_format.size() > 1) {
            OPENVINO_THROW(std::string(MemoryFormat::get_type_info_static().name) +
                           " no rule defined for multiple values.");
        }

        std::string final_mem_format;
        if (unique_mem_format.size() == 1) {
            final_mem_format = *unique_mem_format.begin();
        }
        return MemoryFormat{final_mem_format};
    }
};

class InputMemoryFormats : public MemoryFormats<InputMemoryFormats> {
public:
    OPENVINO_RTTI(InputMemoryFormatsAttr);
    InputMemoryFormats() = default;
    explicit InputMemoryFormats(const std::string& _memory_format) : MemoryFormats(_memory_format) {}
    ~InputMemoryFormats() override;
};

std::string getInputMemoryFormats(const std::shared_ptr<ov::Node>& node);

class OutputMemoryFormats : public MemoryFormats<OutputMemoryFormats> {
public:
    OPENVINO_RTTI(OutputMemoryFormatsAttr);
    OutputMemoryFormats() = default;
    explicit OutputMemoryFormats(const std::string& _memory_format) : MemoryFormats(_memory_format) {}
    ~OutputMemoryFormats() override;
};

std::string getOutputMemoryFormats(const std::shared_ptr<ov::Node>& node);

}  // namespace ov::intel_cpu
