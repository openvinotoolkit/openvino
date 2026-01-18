#pragma once

#include <functional>
#include "isection.hpp"

class Registry {
    public:
    using Evaluator = std::function<bool()>;
    
    static Registry& instance()
    {
        static Registry registry;
        return registry;
    }
    
    // CRE should always return true
    void registryEvaluator(const uint16_t type, Evaluator evaluator)
    {
        // std::cout << "Register function for " << type << std::endl;
        _evaluator[type] = std::move(evaluator);
    }

    bool check(uint16_t& type) const
    {
        auto it = _evaluator.find(type);
        if (it == _evaluator.end())
        {
            // throw std::runtime_error("Unknown section: " + header.type);
            return false;
        }

        // std::cout << "Calling create function for " << header.type << std::endl;
        return it->second();
    }

    void clean() {
        _evaluator.clear();
    }

private:
    std::unordered_map<uint16_t, Evaluator> _evaluator;
};
