// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <type_traits>
#include <unordered_map>
#include <ostream>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>

namespace cldnn {
class json_base;
using json_key = std::string;
using json_base_ptr = std::shared_ptr<json_base>;
using json_map = std::unordered_map<json_key, json_base_ptr>;

class json_base {
public:
    virtual void dump(std::ostream& out, int offset) const = 0;
    virtual ~json_base() = default;
};

template <class Type>
class json_leaf : public json_base {
private:
    Type value;

public:
    explicit json_leaf(const Type& val) : value(val) {}
    explicit json_leaf(Type&& val) : value(std::move(val)) {}
    void dump(std::ostream& out, int) const override { out << value << ",\n"; }
};

template <class Type>
class json_basic_array : public json_base {
private:
    std::vector<Type> values;

public:
    explicit json_basic_array(const std::vector<Type>& arr) : values(arr) {}
    explicit json_basic_array(std::vector<Type>&& arr) : values(std::move(arr)) {}
    void dump(std::ostream& out, int) const override {
        const char* delim = "";
        for (size_t i = 0; i < values.size(); i++) {
            out << delim << values[i];
            delim = ",";
        }
        out << ",\n";
    }
};

class json_composite : public json_base {
private:
    json_map children;

public:
    void dump(std::ostream& out, int offset = 1) const override {
        std::string spaces_brackets(std::max(offset - 1, 0) * 4, ' ');
        std::string spaces_content(offset * 4, ' ');
        out << "\n" << spaces_brackets << "{\n";

        for (const auto& it : children) {
            out << spaces_content << it.first << " : ";
            it.second->dump(out, offset + 1);
        }

        out << spaces_brackets << "}\n";
    }

    template <class Type>
    void add(json_key key, Type value) {
        children[key] = std::make_shared<json_leaf<Type>>(value);
    }
    void add(json_key key, json_composite comp) { children[key] = std::make_shared<json_composite>(comp); }
    template <class Type>
    void add(json_key key, std::vector<Type> array) {
        children[key] = std::make_shared<json_basic_array<Type>>(array);
    }
};

}  // namespace cldnn
