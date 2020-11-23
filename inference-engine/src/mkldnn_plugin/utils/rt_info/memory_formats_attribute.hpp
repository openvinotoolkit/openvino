// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace ngraph {

constexpr const char *InputMemoryFormatsAttr = "InputMemoryFormats";
constexpr const char *OutputMemoryFormatsAttr = "OutputMemoryFormats";

class MemoryFormats {
protected:
    std::string memory_format;

public:
    MemoryFormats() = default;
    explicit MemoryFormats(const std::string &_memory_format) : memory_format(_memory_format) {}
    std::string getMemoryFormats() const { return memory_format; }
};

template <typename MemoryFormatsType>
class MemoryFormatsHelper : public VariantImpl<MemoryFormatsType> {
public:
    MemoryFormatsHelper(const MemoryFormatsType& value) : VariantImpl<MemoryFormatsType>(value) {}

    static std::string getMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
        const auto &rtInfo = node->get_rt_info();
        using MemoryFormatsWraper = VariantWrapper<MemoryFormatsType>;
        if (!rtInfo.count(MemoryFormatsWraper::type_info.name)) return "";
        const auto &attr = rtInfo.at(MemoryFormatsWraper::type_info.name);
        MemoryFormatsType mem_format = as_type_ptr<MemoryFormatsWraper>(attr)->get();
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

class InputMemoryFormats : public MemoryFormats {
public:
    InputMemoryFormats() = default;
    explicit InputMemoryFormats(const std::string &_memory_format) : MemoryFormats(_memory_format) {}
};

extern template class MemoryFormatsHelper<InputMemoryFormats>;

template<>
class VariantWrapper<InputMemoryFormats> : public MemoryFormatsHelper<InputMemoryFormats> {
public:
    static constexpr VariantTypeInfo type_info{InputMemoryFormatsAttr, 0};
    const VariantTypeInfo &get_type_info() const override { return type_info; }

    VariantWrapper(const InputMemoryFormats &value) : MemoryFormatsHelper<InputMemoryFormats>(value) {}
};

std::string getInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

class OutputMemoryFormats : public MemoryFormats {
public:
    OutputMemoryFormats() = default;
    explicit OutputMemoryFormats(const std::string &_memory_format) : MemoryFormats(_memory_format) {}
};

extern template class MemoryFormatsHelper<OutputMemoryFormats>;

template<>
class VariantWrapper<OutputMemoryFormats> : public MemoryFormatsHelper<OutputMemoryFormats> {
public:
    static constexpr VariantTypeInfo type_info{OutputMemoryFormatsAttr, 0};
    const VariantTypeInfo &get_type_info() const override { return type_info; }

    VariantWrapper(const OutputMemoryFormats &value) : MemoryFormatsHelper<OutputMemoryFormats>(value) {}
};

std::string getOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node);

}  // namespace ngraph
