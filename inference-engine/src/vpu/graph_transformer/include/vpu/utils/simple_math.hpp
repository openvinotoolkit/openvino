// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <utility>

//
// Simple integer arithmetics to be used for the work sizes calculation.
// Supported operations : +,-,*,/,%,(,)
// no unary -,+
// Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
//

namespace vpu {

class SimpleMathExpression final {
public:
    void setVariables(const std::map<char, int>& vars) { _vars = vars; }

    void parse(const std::string& expression);

    int evaluate() const;

private:
    struct Token final {
        enum TokenType {
            Value,
            Operator,
        };

        TokenType type;
        int value;
        char op;

        explicit Token(TokenType t = Value, int v = 0, char o = 0) : type(t), value(v), op(o) {}
    };

private:
    std::map<char, int> _vars;
    std::vector<Token> _parsedTokens;
};

}  // namespace vpu
