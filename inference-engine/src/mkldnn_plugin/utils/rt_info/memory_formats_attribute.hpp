// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace ngraph {

constexpr const char *MLKDNNInputMemoryFormatsAttr = "MLKDNNInputMemoryFormats";
constexpr const char *MLKDNNOutputMemoryFormatsAttr = "MLKDNNOutputMemoryFormats";

template<typename MemoryFormat>
class MLKDNNMemoryFormats : public Variant {
protected:
    std::string memory_format;

public:
    MLKDNNMemoryFormats() = default;
    explicit MLKDNNMemoryFormats(const std::string &_memory_format) : memory_format(_memory_format) {}
    std::string getMemoryFormats() const { return memory_format; }

    ov::Any merge(const ngraph::NodeVector & nodes) override {
        std::set<std::string> unique_mem_format;

        for (auto &node : nodes) {
            auto it_info = node->get_rt_info().find(MemoryFormat::get_type_info_static().name);
            if (it_info != node->get_rt_info().end()) {
                if (auto ptr = it_info->second.template as<std::shared_ptr<MemoryFormat>>()) {
                    std::string mem_format = ptr->getMemoryFormats();
                    if (!mem_format.empty()) {
                        unique_mem_format.insert(mem_format);
                    }
                }
            }
        }

        if (unique_mem_format.size() > 1) {
            throw ngraph::ngraph_error(
                std::string(MemoryFormat::get_type_info_static().name) +
                " no rule defined for multiple values.");
        }

        std::string final_mem_format;
        if (unique_mem_format.size() == 1) {
            final_mem_format = *unique_mem_format.begin();
        }
        return std::make_shared<MemoryFormat>(final_mem_format);
    }
};


class MLKDNNInputMemoryFormats : public MLKDNNMemoryFormats<MLKDNNInputMemoryFormats> {
public:
    OPENVINO_RTTI(MLKDNNInputMemoryFormatsAttr);
    MLKDNNInputMemoryFormats() = default;
    explicit MLKDNNInputMemoryFormats(const std::string &_memory_format) : MLKDNNMemoryFormats(_memory_format) {}
    ~MLKDNNInputMemoryFormats() override;
};

std::string getMLKDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

class MLKDNNOutputMemoryFormats : public MLKDNNMemoryFormats<MLKDNNOutputMemoryFormats> {
public:
    OPENVINO_RTTI(MLKDNNOutputMemoryFormatsAttr);
    MLKDNNOutputMemoryFormats() = default;
    explicit MLKDNNOutputMemoryFormats(const std::string &_memory_format) : MLKDNNMemoryFormats(_memory_format) {}
    ~MLKDNNOutputMemoryFormats() override;
};
std::string getMLKDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

}  // namespace ngraph
