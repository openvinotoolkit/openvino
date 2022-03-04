// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/simple_math.hpp>

#include <cctype>

#include <string>
#include <set>
#include <stack>
#include <map>
#include <functional>

#include <vpu/utils/error.hpp>

namespace vpu {

namespace {

using ValueType = details::IntOrFloat;

struct Operator {
    int priority;
    std::function<ValueType(ValueType, ValueType)> op;
};

static const std::map<std::string, Operator> operators = {
    { "+", { 0, std::plus<ValueType>() }},
    { "-", { 0, std::minus<ValueType>() }},
    { "*", { 1, std::multiplies<ValueType>() }},
    { "/", { 1, std::divides<ValueType>() }},
    { "%", { 1, std::modulus<ValueType>() }}
};

static const std::map<std::string, std::function<ValueType(ValueType)>> function = {
        {"floor", [](ValueType x) { return ValueType{std::floor(x.toFloat())}; }},
        {"ceil" , [](ValueType x) { return ValueType{std::ceil(x.toFloat())}; }},
        {"round", [](ValueType x) { return ValueType{std::round(x.toFloat())}; }},
        {"abs"  , [](ValueType x) { return ValueType{std::abs(x.toFloat())}; }},
        {"sqrt" , [](ValueType x) { return ValueType{std::sqrt(x.toFloat())}; }}
};

bool isFunction(const std::string& token) {
    return function.find(token) != function.end();
}
bool isOperator(const std::string& token) {
    return operators.find(token) != operators.end();
}
int opPriority(const std::string& token) {
    return operators.at(token).priority;
}

}  // namespace

void MathExpression::parse(const std::string& expression) {
    _parsedTokens.clear();
    std::stack<std::string> tokenStack;

    for (auto it = begin(expression); it != end(expression); ++it) {
        if (*it == ' ' || *it == '\t') {
            continue;
        }

        // parse number
        if (std::isdigit(*it)) {
            size_t len = 0;
            // parse number and use its length
            const auto value = std::stof(&*it, &len);
            (void) value;
            // copy sub string that represents a number
            auto substring = std::string{it, it + len};

            auto token = Token{TokenType::Value, ValueType{substring}, ""};
            _parsedTokens.push_back(std::move(token));

            std::advance(it, len - 1);
            continue;
        }

        // parse variable/function
        if (std::isalpha(*it)) {
            const auto end_token = std::find_if_not(it, end(expression),
                [](char c) { return std::isalnum(c) || c == '_'; });
            const auto token = std::string(it, end_token);
            std::advance(it, token.length() - 1);

            if (isFunction(token)) {
                tokenStack.push(token);
                continue;
            }

            if (_vars.find(token) != _vars.end()) {
                _parsedTokens.emplace_back(TokenType::Value, ValueType{_vars.at(token)}, "");
                continue;
            }
        }

        // parse operator
        if (isOperator(std::string(1, *it))) {
            while (!tokenStack.empty()
                   && (isFunction(tokenStack.top())
                       || (isOperator(tokenStack.top())
                           && opPriority(tokenStack.top()) >= opPriority(std::string(1, *it))))) {
                const auto tokenType = isOperator(tokenStack.top()) ? TokenType::Operator
                                                                    : TokenType::Function;
                _parsedTokens.emplace_back(tokenType, ValueType{0}, tokenStack.top());
                tokenStack.pop();
            }

            tokenStack.push(std::string(1, *it));
            continue;
        }

        if (*it == '(') {
            tokenStack.push("(");
            continue;
        }

        if (*it == ')') {
            while (!tokenStack.empty() && tokenStack.top() != "(") {
                const auto tokenType = isOperator(tokenStack.top()) ? TokenType::Operator
                                                                    : TokenType::Function;
                _parsedTokens.emplace_back(tokenType, ValueType{0}, tokenStack.top());
                tokenStack.pop();
            }

            if (!tokenStack.empty()) {
                tokenStack.pop();
            } else {
                VPU_THROW_EXCEPTION << "Mismatched parentheses in " << expression;
            }

            continue;
        }

        VPU_THROW_EXCEPTION << "Unknown token " << *it << " in " << expression;
    }

    while (!tokenStack.empty()) {
        if (tokenStack.top() == "(") {
            VPU_THROW_EXCEPTION << "Mismatched parentheses in " << expression;
        }
        const auto tokenType = isOperator(tokenStack.top()) ? TokenType::Operator
                                                            : TokenType::Function;
        _parsedTokens.emplace_back(tokenType, ValueType{0}, tokenStack.top());
        tokenStack.pop();
    }
}

int MathExpression::evaluate() const {
    std::stack<ValueType> values;

    for (const auto& token : _parsedTokens) {
        switch (token.type) {
            case TokenType::Value:
                values.push(token.value);
                break;
            case TokenType::Operator: {
                if (values.size() < 2) {
                    VPU_THROW_EXCEPTION << "Illegal expression: not enough values for operator evaluation";
                }

                auto val2 = values.top();
                values.pop();

                auto val1 = values.top();
                values.pop();

                values.push(operators.at(token.opName).op(val1, val2));
                break;
            }
            case TokenType::Function: {
                if (values.empty()) {
                    VPU_THROW_EXCEPTION << "Illegal expression: not enough values for function evaluation";
                }
                auto val1 = values.top();
                values.pop();

                values.push(function.at(token.opName)(val1));
                break;
            }
            default:
                VPU_THROW_EXCEPTION << "Illegal expression: unhandled token";
        }
    }

    if (values.size() != 1) {
        VPU_THROW_EXCEPTION << "Illegal expression: not enough operators";
    }

    return values.top().toInt();
}

}  // namespace vpu
