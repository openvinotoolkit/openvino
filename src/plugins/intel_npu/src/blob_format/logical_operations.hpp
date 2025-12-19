#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <map>
#include <vector>
#include <cstring>
#include <sstream>
#include <mutex>
#include <unordered_set>
#include <string_view>
#include <optional>
#include <functional>
#include <assert.h>

#include "registry.hpp"
#include "isection.hpp"

// thought: shouldnt we add compile time checks for expression validation? 
// for example this shouldnt be valid: OPEN AND ELF
// *
// thought 2: should operators work with just one operand?
std::unordered_set<uint16_t> SUPPORTED = {CRE, ELF, WS, BS, IO_LAYOUTS};

std::unordered_set<uint16_t> OPERATORS = {AND, OR, OPEN, CLOSE};
enum class Delimiter {
    PARENTHESIS,
    OPERATOR
};

class Evaluator {
private:
    int cursor = 0;

    static inline bool and_function(bool a, bool b) {
        return a && b;
    }
    static inline bool or_function(bool a, bool b) {
        return a || b;
    }

    inline bool end_condition(uint16_t* expression, const Delimiter end_delimiter) {
        return (end_delimiter == Delimiter::PARENTHESIS) ? (expression[cursor] == CLOSE) : OPERATORS.count(expression[cursor]);
    }

    bool eval(uint16_t* expression, const Delimiter end_delimiter) {
        std::function<bool(bool, bool)> logical_function;
        bool base;
        switch (expression[cursor++]) {
        case AND:
            logical_function = and_function;
            base = true;
            break;
        case OR:
            logical_function = or_function;
            base = false;
            break;
        default:
            throw std::logic_error("no.");
        }

        while (!end_condition(expression, end_delimiter)) {
            if (expression[cursor] == OPEN) {
                ++cursor;
                base = logical_function(base, eval(expression, Delimiter::PARENTHESIS));
                ++cursor;
            } else if (OPERATORS.count(expression[cursor])) {
                base = logical_function(base, eval(expression, Delimiter::OPERATOR));
            } else {
                // base = logical_function(base, supported.count(expression[cursor]));
                base = logical_function(base, Registry::instance().check(expression[cursor]));
                ++cursor;
            }
        }
    
        return base;
    }
public:
    static Evaluator& instance() {
        static Evaluator evaluator;
        return evaluator;
    }

    bool evaluate(std::vector<uint16_t>& expression) {
        cursor = 0;
        assert(expression[0] == OPEN);
        assert(expression[expression.size() - 1] == CLOSE);
        return eval(expression.data() + 1, Delimiter::PARENTHESIS);
    }
};


// int main() {
//     std::vector<uint16_t> expression = {OPEN, AND, L1, L2, OPEN, OR, L3, L4, AND, L6, L7, L8, CLOSE, L5, CLOSE};
//     std::cout << evaluate(expression) << std::endl;

//     return 0;
// }
