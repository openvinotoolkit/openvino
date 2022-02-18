// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace ngraph {

constexpr const char *MKLDNNInputMemoryFormatsAttr = "MKLDNNInputMemoryFormats";
constexpr const char *MKLDNNOutputMemoryFormatsAttr = "MKLDNNOutputMemoryFormats";

template<typename MemoryFormat>
class MKLDNNMemoryFormats : public ov::RuntimeAttribute {
protected:
    std::string memory_format;

public:
    MKLDNNMemoryFormats() = default;
    explicit MKLDNNMemoryFormats(const std::string &_memory_format) : memory_format(_memory_format) {}
    std::string getMemoryFormats() const { return memory_format; }

    ov::Any merge(const ngraph::NodeVector & nodes) const override {
        std::set<std::string> unique_mem_format;

        for (auto &node : nodes) {
            auto it_info = node->get_rt_info().find(MemoryFormat::get_type_info_static());
            if (it_info != node->get_rt_info().end()) {
                std::string mem_format = it_info->second.template as<MemoryFormat>().getMemoryFormats();
                if (!mem_format.empty()) {
                    unique_mem_format.insert(mem_format);
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
        return MemoryFormat{final_mem_format};
    }
};


class MKLDNNInputMemoryFormats : public MKLDNNMemoryFormats<MKLDNNInputMemoryFormats> {
public:
    OPENVINO_RTTI(MKLDNNInputMemoryFormatsAttr);
    MKLDNNInputMemoryFormats() = default;
    explicit MKLDNNInputMemoryFormats(const std::string &_memory_format) : MKLDNNMemoryFormats(_memory_format) {}
    ~MKLDNNInputMemoryFormats() override;
};

std::string getMKLDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

class MKLDNNOutputMemoryFormats : public MKLDNNMemoryFormats<MKLDNNOutputMemoryFormats> {
public:
    OPENVINO_RTTI(MKLDNNOutputMemoryFormatsAttr);
    MKLDNNOutputMemoryFormats() = default;
    explicit MKLDNNOutputMemoryFormats(const std::string &_memory_format) : MKLDNNMemoryFormats(_memory_format) {}
    ~MKLDNNOutputMemoryFormats() override;
};
std::string getMKLDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

}  // namespace ngraph
