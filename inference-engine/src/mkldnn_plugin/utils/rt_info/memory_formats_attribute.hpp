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

class MLKDNNMemoryFormats {
protected:
    std::string memory_format;

public:
    MLKDNNMemoryFormats() = default;
    explicit MLKDNNMemoryFormats(const std::string &_memory_format) : memory_format(_memory_format) {}
    std::string getMemoryFormats() const { return memory_format; }
};

template <typename MemoryFormatsType>
class MLKDNNMemoryFormatsHelper : public VariantImpl<MemoryFormatsType> {
public:
    MLKDNNMemoryFormatsHelper(const MemoryFormatsType& value) : VariantImpl<MemoryFormatsType>(value) {}

    static std::string getMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
        const auto &rtInfo = node->get_rt_info();
        using MemoryFormatsWrapper = VariantWrapper<MemoryFormatsType>;
        if (!rtInfo.count(MemoryFormatsWrapper::type_info.name)) return "";
        const auto &attr = rtInfo.at(MemoryFormatsWrapper::type_info.name);
        MemoryFormatsType mem_format = as_type_ptr<MemoryFormatsWrapper>(attr)->get();
        return mem_format.getMemoryFormats();
    }

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override {
        std::set<std::string> unique_mem_format;

        for (auto &node : nodes) {
            std::string mem_format = getMemoryFormats(node);
            if (!mem_format.empty()) unique_mem_format.insert(mem_format);
        }

        if (unique_mem_format.size() > 1) {
            throw ngraph_error(std::string(VariantWrapper<MemoryFormatsType>::type_info.name) + " no rule defined for multiple values.");
        }

        std::string final_mem_format;
        if (unique_mem_format.size() == 1) {
            final_mem_format = *unique_mem_format.begin();
        }
        return std::make_shared<VariantWrapper<MemoryFormatsType>>(MemoryFormatsType(final_mem_format));
    }

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override {
        throw ngraph_error(std::string(VariantWrapper<MemoryFormatsType>::type_info.name) + " has no default initialization.");
    }
};

class MLKDNNInputMemoryFormats : public MLKDNNMemoryFormats {
public:
    MLKDNNInputMemoryFormats() = default;
    explicit MLKDNNInputMemoryFormats(const std::string &_memory_format) : MLKDNNMemoryFormats(_memory_format) {}
};

extern template class MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>;

template<>
class VariantWrapper<MLKDNNInputMemoryFormats> : public MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats> {
public:
    static constexpr VariantTypeInfo type_info{MLKDNNInputMemoryFormatsAttr, 0};
    const VariantTypeInfo &get_type_info() const override { return type_info; }

    VariantWrapper(const MLKDNNInputMemoryFormats &value) : MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>(value) {}
};

std::string getMLKDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

class MLKDNNOutputMemoryFormats : public MLKDNNMemoryFormats {
public:
    MLKDNNOutputMemoryFormats() = default;
    explicit MLKDNNOutputMemoryFormats(const std::string &_memory_format) : MLKDNNMemoryFormats(_memory_format) {}
};

extern template class MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>;

template<>
class VariantWrapper<MLKDNNOutputMemoryFormats> : public MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats> {
public:
    static constexpr VariantTypeInfo type_info{MLKDNNOutputMemoryFormatsAttr, 0};
    const VariantTypeInfo &get_type_info() const override { return type_info; }

    VariantWrapper(const MLKDNNOutputMemoryFormats &value) : MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>(value) {}
};

std::string getMLKDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

}  // namespace ngraph
