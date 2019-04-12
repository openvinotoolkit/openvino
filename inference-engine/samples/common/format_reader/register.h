// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
 * \brief Register for readers
 * \file register.h
 */
#pragma once

#include <format_reader.h>
#include <functional>
#include <vector>
#include <string>

namespace FormatReader {
/**
 * \class Registry
 * \brief Create reader from fabric
 */
class Registry {
private:
    typedef std::function<Reader *(const std::string &filename)> CreatorFunction;
    static std::vector<CreatorFunction> _data;
public:
    /**
     * \brief Create reader
     * @param filename - path to input data
     * @return Reader for input data or nullptr
     */
    static Reader *CreateReader(const char *filename);

    /**
     * \brief Registers reader in fabric
     * @param f - a creation function
     */
    static void RegisterReader(CreatorFunction f);
};

/**
 * \class Register
 * \brief Registers reader in fabric
 */
template<typename To>
class Register {
public:
    /**
     * \brief Constructor creates creation function for fabric
     * @return Register object
     */
    Register() {
        Registry::RegisterReader([](const std::string &filename) -> Reader * {
            return new To(filename);
        });
    }
};
}  // namespace FormatReader