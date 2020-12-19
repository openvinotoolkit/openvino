// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cstdint>

#include <details/ie_exception.hpp>

namespace InferenceEngine {

class NetworkCompilationContext final {
    template <typename T>
    static std::size_t hash_combine(std::size_t seed, const T & a) {
        std::size_t hash = std::hash<T>()(a);
        return hash ^ (seed << 1);
    }

public:
    // network structure (ngraph::Function description)
    std::vector<uint8_t> m_constants;
    std::string m_model;

    std::string computeHash() const {
        validate();

        // compute hash
        size_t seed {};
        seed = hash_combine(seed, m_model);

        // TODO: more values to hash
        // seed = hash_combine(seed, m_constants);

        return std::to_string(seed);
    }

private:
    void validate() const {
        // m_constants can be empty because we can have models w/o weights

        if (m_model.empty())
            THROW_IE_EXCEPTION << "Model string is empty";
    }
};

}  // namespace InferenceEngine
