// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header file for the user input exception
 * \file user_exception.hpp
 */
#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <list>
#include <functional>

/**
 * @def THROW_USER_EXCEPTION
 * @brief A macro used to throw the exception with a notable description
 */
#define THROW_USER_EXCEPTION(exitCode) \
    throw UserException(exitCode)

/**
 * @class UserException
 * @brief The UserException class implements an exception appearing as a result of user input
 */
class UserException : public std::exception {
    mutable std::string errorDesc;
    std::shared_ptr<std::stringstream> exception_stream;
    int _exitCode;

public:
    /**
     * @brief A C++ std::exception API member
     * @return An exception description with a file name and file line
     */
    const char *what() const noexcept override {
        if (errorDesc.empty() && exception_stream) {
            errorDesc = exception_stream->str();
        }
        return errorDesc.c_str();
    }

    /**
     * @brief A constructor. Creates a UserException object
     */
    explicit UserException(int exitCode) : _exitCode(exitCode) {
    }

    UserException(int exitCode, std::string msg) : _exitCode(exitCode) {
        *this << msg;
    }

    /**
     * @brief A stream output operator to be used within exception
     * @param arg Object for serialization in the exception message
     */
    template<class T>
    UserException &operator<<(const T &arg) {
        if (!exception_stream) {
            exception_stream.reset(new std::stringstream());
        }
        (*exception_stream) << arg;
        return *this;
    }

    int exitCode() const { return _exitCode; }
};

class UserExceptions : public std::exception {
    std::list<UserException> _list;
    mutable std::string msg;

public:
    UserExceptions &operator<<(const UserException &arg) {
        _list.push_back(arg);
        return *this;
    }

    const char *what() const noexcept override {
        std::stringstream ss;

        if (_list.size() == 1) {
            ss << _list.back().what();
        } else {
            auto iter = _list.begin();
            for (int i = 0; i < _list.size() - 1; i++) {
                ss << "\t* " << (*iter++).what() << std::endl;
            }
            ss << "\t* " << _list.back().what();
        }

        msg = ss.str();
        return msg.c_str();
    }

    const std::list<UserException>& list() const {
        return _list;
    }

    bool empty() const { return _list.empty(); }
};

