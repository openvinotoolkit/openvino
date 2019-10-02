// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the main Inference Engine exception
 * \file ie_exception.hpp
 */
#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <utility>

/**
 * @def THROW_IE_EXCEPTION
 * @brief A macro used to throw the exception with a notable description
 */
#define THROW_IE_EXCEPTION\
    throw InferenceEngine::details::InferenceEngineException(__FILE__, __LINE__)\

/**
 * @def IE_ASSERT
 * @brief Uses assert() function if NDEBUG is not defined, InferenceEngine exception otherwise
 */
#ifdef NDEBUG
    #define IE_ASSERT(EXPRESSION)\
    if (!(EXPRESSION)) throw InferenceEngine::details::InferenceEngineException(__FILE__, __LINE__) << "AssertionFailed: " << #EXPRESSION  // NOLINT
#else
#include <cassert>

class NullStream {
 public :
    template <class T>
    NullStream & operator << (const T &) noexcept {
        return *this;
    }

    NullStream &  operator<< (std::ostream & (*)(std::ostream &)) noexcept {
        return *this;
    }
};

#define IE_ASSERT(EXPRESSION)\
    assert((EXPRESSION)); NullStream()
#endif  // NDEBUG

namespace InferenceEngine {
enum StatusCode: int;
namespace details {

/**
 * @brief The InferenceEngineException class implements the main Inference Engine exception
 */
class InferenceEngineException : public std::exception {
    mutable std::string errorDesc;
    StatusCode status_code = static_cast<StatusCode>(0);
    std::string _file;
    int _line;
    std::shared_ptr<std::stringstream> exception_stream;
    bool save_to_status_code = false;

public:
    /**
     * @brief A C++ std::exception API member
     * @return An exception description with a file name and file line
     */
    const char *what() const noexcept override {
        if (errorDesc.empty() && exception_stream) {
            errorDesc = exception_stream->str();
#ifndef NDEBUG
            errorDesc +=  "\n" + _file + ":" + std::to_string(_line);
#endif
        }
        return errorDesc.c_str();
    }

    /**
     * @brief A constructor. Creates an InferenceEngineException object from a specific file and line
     * @param filename File where exception has been thrown
     * @param line Line of the exception emitter
     */
    InferenceEngineException(const std::string &filename, const int line)
        : _file(filename), _line(line) {
    }

    /**
     * @brief noexcept required for copy ctor
     * @details The C++ Standard, [except.throw], paragraph 3 [ISO/IEC 14882-2014]
     */
    InferenceEngineException(const InferenceEngineException & that) noexcept {
        errorDesc = that.errorDesc;
        status_code = that.status_code;
        _file = that._file;
        _line = that._line;
        exception_stream = that.exception_stream;
    }

    /**
     * @brief A stream output operator to be used within exception
     * @param arg Object for serialization in the exception message
     */
    template<class T>
    InferenceEngineException& operator<<(const T &arg) {
        if (save_to_status_code) {
            auto can_convert =  status_code_assign(arg);
            save_to_status_code = false;
            if (can_convert.second) {
                this->status_code = can_convert.first;
                return *this;
            }
        }
        if (!exception_stream) {
            exception_stream.reset(new std::stringstream());
        }
        (*exception_stream) << arg;
        return *this;
    }

    /**
     * @brief Manipulator to indicate that next item has to be converted to StatusCode to save
     * @param iex InferenceEngineException object
     */
    friend InferenceEngineException& as_status(InferenceEngineException& iex) {
        iex.save_to_status_code = true;
        return iex;
    }

    /**
     * @brief A stream output operator to catch InferenceEngineException manipulators
     * @param manip InferenceEngineException manipulator to call
     */
    InferenceEngineException& operator<<(InferenceEngineException& (*manip)(InferenceEngineException &)) {
        return manip(*this);
    }

    /**  @brief Check if it has StatusCode value */
    bool hasStatus() const {
        return this->status_code == 0 ? false : true;
    }

    /** @brief Get StatusCode value */
    StatusCode getStatus() const {
        return this->status_code;
    }

private:
    std::pair<StatusCode, bool> status_code_assign(const StatusCode& status) {
        return {status, true};
    }

    template <typename T>
    std::pair<StatusCode, bool> status_code_assign(const T &) {
        return {static_cast<StatusCode>(0), false};
    }
};

InferenceEngineException& as_status(InferenceEngineException& iex);

static_assert(std::is_nothrow_copy_constructible<InferenceEngineException>::value,
              "InferenceEngineException must be nothrow copy constructible");
}  // namespace details
}  // namespace InferenceEngine
