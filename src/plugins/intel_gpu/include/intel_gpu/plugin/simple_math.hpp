// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <utility>

/*
*   Simple integer arithmetics to be used for the work sizes calculation
*   Supported ops: +,-,*,/,%,(,)
*   * no unary -,+
*   Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
*/


class SimpleMathExpression {
public:
    SimpleMathExpression() :m_parsed(false) {}
    void SetVariables(const std::map<char, int>& vars);
    bool SetExpression(const std::string& expression);
    bool IsParsed()const { return m_parsed; }
    int Evaluate()const;  // undefined behavior if not parsed properly

private:
    std::map<char, int> m_variables;
    std::string m_expression;
    bool m_parsed;
    bool Parse();

    struct Token {
        enum TokenType {
            Value,
            Operator,
        } type;
        int value;
        char op;
        explicit Token(TokenType t = Value, int v = 0, char o = 0) :type(t), value(v), op(o) {}
    };
    std::vector<Token> m_parsedTokens;

    static const std::set<char> whitespaces;
    using Operator = std::pair<int, std::function<int(int, int)>>;  // priority, function
    static const std::map<char, Operator> operators;
};
