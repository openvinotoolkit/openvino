// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <yaml-cpp/yaml.h>

#include "utils/logger.hpp"

#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

class ConfigNode {
public:
    ConfigNode(YAML::Node&& node, bool isRoot = false): _node(std::move(node)), _isRoot(isRoot) {
        collectKeys();
    }

    ~ConfigNode() {
        if (_isRoot) {
            std::vector<std::string> unusedKeys = getUnusedKeys();
            if (!unusedKeys.empty()) {
                std::string unusedKeysStr = "Unused config keys: ";
                for (size_t i = 0; i < unusedKeys.size(); ++i) {
                    unusedKeysStr += unusedKeys[i];
                    if (i < unusedKeys.size() - 1) unusedKeysStr += " ";
                }
                LOG_WARN() << unusedKeysStr << std::endl;
            }
        }
    }

    ConfigNode(const ConfigNode&) = delete;
    ConfigNode& operator=(const ConfigNode&) = delete;

    ConfigNode(ConfigNode&& other) = delete;
    ConfigNode& operator=(ConfigNode&& other)  = delete;

    bool IsSequence() const { return _node.IsSequence(); }
    bool IsMap() const { return _node.IsMap(); }
    std::size_t size() const { return _node.size(); }

    template <typename T>
    T as() const {
        // For primitive types, use YAML's converter
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
            return _node.as<T>();
        } else {
            T result;
            if (!YAML::convert<T>::decode(*this, result)) {
                THROW_ERROR("Failed to convert config node to requested type");
            }
            return result;
        }
    }

    template <typename Key>
    ConfigNode& operator[](const Key& key);

    template <typename Key>
    const ConfigNode& operator[](const Key& key) const;

    struct ConfigKey {
        YAML::Node _node;

        template <typename T>
        T as() const { return _node.as<T>(); }
    };

    struct KeyValueProxy {
        ConfigKey first;
        const ConfigNode& second;

        const KeyValueProxy* operator->() const { return this; }

        bool IsSequence() const { return second.IsSequence(); }
        bool IsMap() const { return second.IsMap(); }
        std::size_t size() const { return second.size(); }

        template <typename T>
        T as() const { return second.as<T>(); }

        template <typename Key>
        const ConfigNode& operator[](const Key& key) const { return second[key]; }

        explicit operator bool() const { return static_cast<bool>(second); }

        operator const ConfigNode&() const { return second; }
    };

    class const_iterator {
    public:
        const_iterator(ConfigNode& parent, std::size_t index)
            : _parent(parent), _index(index) {
            if (_parent._node.IsMap()) {
                cacheKeysAndNodes();
            }
        }

        KeyValueProxy operator*() const {
            if (_parent._node.IsMap()) {
                const auto& [keyStr, keyNode] = _cachedKeys[_index];
                return KeyValueProxy{ConfigKey{keyNode}, _parent[keyStr]};
            } else {
                return KeyValueProxy{ConfigKey{YAML::Node()}, _parent[_index]};
            }
        }

        KeyValueProxy operator->() const {
            return operator*();
        }

        const_iterator& operator++() {
            ++_index;
            return *this;
        }

        bool operator!=(const const_iterator& other) const {
            return _index != other._index;
        }

    private:
        void cacheKeysAndNodes() {
            if (_cachedKeys.empty()) {
                for (const auto& kv : _parent._node) {
                    _cachedKeys.emplace_back(kv.first.as<std::string>(), kv.first);
                }
            }
        }

        ConfigNode& _parent;
        std::size_t _index;
        std::vector<std::pair<std::string, YAML::Node>> _cachedKeys;
    };

    const_iterator begin() const {
        return const_iterator(const_cast<ConfigNode&>(*this), 0);
    }

    const_iterator end() const {
        return const_iterator(const_cast<ConfigNode&>(*this), _node.size());
    }

    explicit operator bool() const { return _node.IsDefined() && !_node.IsNull(); }

private:
    void collectKeys() {
        if (_node.IsMap()) {
            for (const auto& kv : _node) {
                _keys.insert(kv.first.as<std::string>());
            }
        }
    }

    std::vector<std::string> getUnusedKeys() const {
        std::vector<std::string> unused(_keys.begin(), _keys.end());
        for (const auto& [key, child] : _children) {
            auto childUnused = child->getUnusedKeys();
            unused.insert(unused.end(), childUnused.begin(), childUnused.end());
        }
        return unused;
    }

    YAML::Node _node;
    bool _isRoot;
    mutable std::unordered_set<std::string> _keys;
    mutable std::unordered_map<std::string, std::unique_ptr<ConfigNode>> _children;
};

template <typename Key>
const ConfigNode& ConfigNode::operator[](const Key& key) const {
    std::string keyStr;
    if constexpr (std::is_convertible_v<Key, std::string>) {
        keyStr = std::string(key);
    } else {
        keyStr = std::to_string(key);
    }

    _keys.erase(keyStr);

    auto it = _children.find(keyStr);
    if (it == _children.end()) {
        auto child = std::make_unique<ConfigNode>(YAML::Clone(_node[key]));
        auto [inserted_it, success] = _children.emplace(keyStr, std::move(child));
        return *inserted_it->second;
    }

    return *it->second;
}

template <typename Key>
ConfigNode& ConfigNode::operator[](const Key& key) {
    return const_cast<ConfigNode&>(std::as_const(*this)[key]);
}
