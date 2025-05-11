// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/simple_math.hpp"
#include <cctype>
#include <string>
#include <set>
#include <stack>
#include <map>
#include <stdexcept>

// Using the algorithm from: https://en.wikipedia.org/wiki/Shunting-yard_algorithm

const std::set<char> SimpleMathExpression::whitespaces = {
    ' ',
    '\t',
};
const std::map<char, SimpleMathExpression::Operator> SimpleMathExpression::operators = {
    { '+', { 0, std::plus<int>() } },
    { '-', { 0, std::minus<int>() } },
    { '*', { 1, std::multiplies<int>() } },
    { '/', { 1, std::divides<int>()  } },
    { '%', { 1, std::modulus<int>()  } },
};

void SimpleMathExpression::SetVariables(const std::map<char, int>& vars) {
    m_variables = vars;
}

bool SimpleMathExpression::SetExpression(const std::string & expression) {
    m_expression = expression;
    m_parsed = Parse();
    return m_parsed;
}

int SimpleMathExpression::Evaluate() const {
    if (!m_parsed) {
        throw std::runtime_error("Evaluation error: not parsed yet");
    }

    std::stack<int> values;
    for (Token t : m_parsedTokens) {
        switch (t.type) {
        case Token::Value:
            values.push(t.value);
            break;
        case Token::Operator: {
            if (values.size() < 2) {
                throw std::runtime_error("Illegal expression: not enough values for operator evaluation");
            }
            // pop last 2 values and apply operand
            int val2 = values.top();
            values.pop();
            int val1 = values.top();
            values.pop();
            values.push(operators.at(t.op).second(val1, val2));
        }
            break;
        default:
            throw std::runtime_error("Illegal expression: unhandled token");
        }
    }
    if (values.size() != 1) {
        throw std::runtime_error("Illegal expression: not enough operators");
    }
    return values.top();
}

bool SimpleMathExpression::Parse() {
    std::stack<char> operatorStack;
    m_parsedTokens.clear();

    // while there are tokens to be read:
    for (size_t i = 0; i != m_expression.length(); i++) {
        //  read a token.
        while (whitespaces.find(m_expression.at(i)) != whitespaces.end()) i++;  // ignore whitespaces
        char curr = m_expression.at(i);

        //  if the token is a number, then push it to the output queue.
        if (isdigit(curr)) {
            size_t len;
            int value = std::stoi(m_expression.substr(i), &len);
            m_parsedTokens.push_back(Token(Token::Value, value, 0));
            i += (len - 1);
            continue;
        }

        //  if the token is a variable, then push it's value to the output queue.
        if (m_variables.find(curr) != m_variables.end()) {
            m_parsedTokens.push_back(Token(Token::Value, m_variables.at(curr), 0));
            continue;
        }

        //  if the token is an operator, then:
        if (operators.find(curr) != operators.end()) {
        //    while there is an operator at the top of the operator stack with
        //      greater than or equal to precedence:
        //        pop operators from the operator stack, onto the output queue;
            while ( !operatorStack.empty() &&
                    (operators.find(operatorStack.top()) != operators.end()) &&
                    (operators.at(operatorStack.top()).first >= operators.at(curr).first)) {
                char op = operatorStack.top();
                operatorStack.pop();
                m_parsedTokens.push_back(Token(Token::Operator, 0, op));
            }
        //      push the read operator onto the operator stack.
            operatorStack.push(curr);
            continue;
        }

        //  if the token is a left bracket (i.e. "("), then:
        //    push it onto the operator stack.
        if (curr == '(') {
            operatorStack.push(curr);
            continue;
        }

        //  if the token is a right bracket (i.e. ")"), then:
        if (curr == ')') {
            //    while the operator at the top of the operator stack is not a left bracket:
            //      pop operators from the operator stack onto the output queue.
            while (!operatorStack.empty() && operatorStack.top() != '(') {
                m_parsedTokens.push_back(Token(Token::Operator, 0, operatorStack.top()));
                operatorStack.pop();
            }
            //    pop the left bracket from the stack.
            //    /* if the stack runs out without finding a left bracket, then there are
            //       mismatched parentheses. */
            if (!operatorStack.empty() && operatorStack.top() == '(') {
                operatorStack.pop();
            } else {
                return false;
            }
            continue;
        }

        // unknown token
        return false;
    }
    // if there are no more tokens to read:
    //  while there are still operator tokens on the stack:
    //    /* if the operator token on the top of the stack is a bracket, then
    //      there are mismatched parentheses. */
    //    pop the operator onto the output queue.
    while (!operatorStack.empty()) {
        if (operatorStack.top() == '(') {
            return false;
        }
        m_parsedTokens.push_back(Token(Token::Operator, 0, operatorStack.top()));
        operatorStack.pop();
    }

    // exit.
    m_parsed = true;
    return true;
}
