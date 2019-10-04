// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/simple_math.hpp>

#include <cctype>

#include <string>
#include <set>
#include <stack>
#include <map>
#include <stdexcept>
#include <utility>
#include <functional>

#include <vpu/utils/extra.hpp>

namespace vpu {

namespace {

const std::set<char> whitespaces = {
    ' ',
    '\t',
};

// priority, function
using Operator = std::pair<int, std::function<int(int, int)>>;

const std::map<char, Operator> operators = {
    { '+', { 0, std::plus<int>() } },
    { '-', { 0, std::minus<int>() } },
    { '*', { 1, std::multiplies<int>() } },
    { '/', { 1, std::divides<int>()  } },
    { '%', { 1, std::modulus<int>()  } },
};

}  // namespace

void SimpleMathExpression::parse(const std::string& expression) {
    _parsedTokens.clear();

    std::stack<char> operatorStack;

    // While there are tokens to be read.
    for (size_t i = 0; i != expression.length(); i++) {
        // Ignore whitespaces;
        while (whitespaces.find(expression[i]) != whitespaces.end()) {
            i++;
        }

        // Read a token.
        auto curr = expression[i];

        // If the token is a number, then push it to the output queue.
        if (std::isdigit(curr)) {
            size_t len = 0;
            auto value = std::stoi(expression.substr(i), &len);

            _parsedTokens.emplace_back(Token(Token::Value, value, 0));

            i += (len - 1);

            continue;
        }

        // If the token is a variable, then push it's value to the output queue.
        if (_vars.find(curr) != _vars.end()) {
            _parsedTokens.emplace_back(Token(Token::Value, _vars.at(curr), 0));

            continue;
        }

        // If the token is an operator, then:
        if (operators.find(curr) != operators.end()) {
            // While there is an operator at the top of the operator stack with
            //   greater than or equal to precedence:
            //     pop operators from the operator stack, onto the output queue;
            while (!operatorStack.empty() &&
                   (operators.find(operatorStack.top()) != operators.end()) &&
                   (operators.at(operatorStack.top()).first >= operators.at(curr).first)) {
                auto op = operatorStack.top();
                operatorStack.pop();

                _parsedTokens.emplace_back(Token(Token::Operator, 0, op));
            }

            //     push the read operator onto the operator stack.
            operatorStack.push(curr);

            continue;
        }

        // If the token is a left bracket (i.e. "("), then:
        //   push it onto the operator stack.
        if (curr == '(') {
            operatorStack.push(curr);

            continue;
        }

        // If the token is a right bracket (i.e. ")"), then:
        if (curr == ')') {
            // While the operator at the top of the operator stack is not a left bracket:
            //   pop operators from the operator stack onto the output queue;
            while (!operatorStack.empty() &&
                   operatorStack.top() != '(') {
                _parsedTokens.emplace_back(Token(Token::Operator, 0, operatorStack.top()));

                operatorStack.pop();
            }

            //   pop the left bracket from the stack.
            // If the stack runs out without finding a left bracket, then there are mismatched parentheses.
            if (!operatorStack.empty() &&
                operatorStack.top() == '(') {
                operatorStack.pop();
            } else {
                VPU_THROW_EXCEPTION << "Mismatched parentheses in " << expression;
            }

            continue;
        }

        // Unknown token
        VPU_THROW_EXCEPTION << "Unknown token " << curr << " in " << expression;
    }

    // If there are no more tokens to read:
    //   while there are still operator tokens on the stack:
    //     if the operator token on the top of the stack is a bracket, then
    //       there are mismatched parentheses;
    //     pop the operator onto the output queue.
    while (!operatorStack.empty()) {
        if (operatorStack.top() == '(') {
            VPU_THROW_EXCEPTION << "Mismatched parentheses in " << expression;
        }

        _parsedTokens.emplace_back(Token(Token::Operator, 0, operatorStack.top()));

        operatorStack.pop();
    }
}

int SimpleMathExpression::evaluate() const {
    std::stack<int> values;
    for (const auto& t : _parsedTokens) {
        switch (t.type) {
        case Token::Value:
            values.push(t.value);
            break;
        case Token::Operator: {
            if (values.size() < 2) {
                VPU_THROW_EXCEPTION << "Illegal expression: not enough values for operator evaluation";
            }

            // pop last 2 values and apply operand
            auto val2 = values.top();
            values.pop();

            auto val1 = values.top();
            values.pop();

            values.push(operators.at(t.op).second(val1, val2));

            break;
        }
        default:
            VPU_THROW_EXCEPTION << "Illegal expression: unhandled token";
        }
    }

    if (values.size() != 1) {
        VPU_THROW_EXCEPTION << "Illegal expression: not enough operators";
    }

    return values.top();
}

}  // namespace vpu
