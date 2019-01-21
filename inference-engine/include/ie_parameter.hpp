// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the CNNNetworkIterator class
 * @file ie_cnn_network_iterator.hpp
 */
#pragma once

#include <details/ie_exception.hpp>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cctype>
#include <string>
#include <map>

namespace InferenceEngine {

/**
 * @brief This class represents an object to work with different parameters
 */
class Parameter {
public:
    /**
     * @brief Default constructor
     */
    Parameter() = default;

    /**
     * @brief The constructor creates a Parameter object with string value
     * @param value string value
     */
    Parameter(const std::string& value): initialized(true), value(value) {}         // NOLINT

    /**
     * @brief The constructor creates a Parameter object with template value
     * @param value template value
     */
    template <class T>
    Parameter(const T& value): initialized(true), value(std::to_string(value)) {}   // NOLINT

    /**
     * @brief The constructor creates a Parameter object with a vector of template values
     * @param values vector of template values
     */
    template <class T>
    Parameter(const std::vector<T>& values): initialized(true) {                    // NOLINT
        for (const auto& val : values) {
            if (!value.empty())
                value += ",";
            value += std::to_string(val);
        }
    }

    /**
     * @brief The cast to string object
     * Throws exception if parameter was not found.
     * @return string value
     */
    operator std::string() const {                                                  // NOLINT
        return asString();
    }

    /**
     * @brief Returns a string value for the given parameter or returns the default one
     * @param def Default value of the parameter if not found
     * @return A string value
     */
    std::string asString(std::string def) const {
        if (!initialized) {
            return def;
        }
        return value;
    }

    /**
     * @brief Returns a string value for the given parameter.
     * Throws exception if parameter was not found.
     * @return A string value
     */
    std::string asString() const {
        if (!initialized) {
            THROW_IE_EXCEPTION << "Parameter was not initialized!";
        }
        return value;
    }

    /**
     * @brief Gets float value for the given parameter
     * @param def - default value of the parameter if not found
     * @return float value
     */
    float asFloat(float def) const {
        std::string val = asString(std::to_string(def));
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a float value for the given layer parameter
     * @return A float value for the specified parameter
     */
    float asFloat() const {
        std::string val = asString();
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a vector of float values for the given parameter or returns the default value
     * @param def Default value of the parameter if not found
     * @return vector of float values
     */
    std::vector<float> asFloats(std::vector<float> def) const {
        std::string vals = asString("");
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of float values for the given parameter
     * @return vector of float values
     */
    std::vector<float> asFloats() const {
        std::string vals = asString();
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns an integer value for the given parameter or returns the default value
     * @param def Default value of the parameter if not found
     * @return An int value for the specified parameter
     */
    int asInt(int def) const {
        std::string val = asString(std::to_string(def));
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Value " << val << " cannot be casted to int.";
        }
    }

    /**
     * @brief Returns an integer value for the given parameter
     * @return An int value for the specified parameter
     */
    int asInt() const {
        std::string val = asString();
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Value " << val << " cannot be casted to int.";
        }
    }


    /**
     * @brief Returns a vector of int values for the given parameter or returns the default value
     * @param def Default value of the parameter if not found
     * @return vector of int values
     */
    std::vector<int> asInts(std::vector<int> def) const {
        std::string vals = asString("");
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Value " << vals << " cannot be casted to ints.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of int values for the given parameter
     * @return vector of int values
     */
    std::vector<int> asInts() const {
        std::string vals = asString();
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Value " << vals <<  " cannot be casted to ints.";
            }
        }
        return result;
    }
    /**
     * @brief Returns an unsigned integer value for the given parameter or returns the default value
     * @param def Default value of the parameter if not found
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int asUInt(unsigned int def) const {
        std::string val = asString(std::to_string(def));
        std::string message = "Value " + val + " cannot be casted to unsigned int.";
        try {
            int value = std::stoi(val);
            if (value < 0) {
                THROW_IE_EXCEPTION << message;
            }
            return static_cast<unsigned int>(value);
        } catch (...) {
            THROW_IE_EXCEPTION << message;
        }
    }

    /**
     * @brief Returns an unsigned integer value for the given parameter
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int asUInt() const {
        std::string val = asString();
        std::string message = "Value " + val + " cannot be casted to unsigned int.";
        try {
            int value = std::stoi(val);
            if (value < 0) {
                THROW_IE_EXCEPTION << message;
            }
            return static_cast<unsigned int>(value);
        } catch (...) {
            THROW_IE_EXCEPTION << message;
        }
    }


    /**
     * @brief Returns a vector of unsigned int values for the given parameter or returns the default value
     * @param def Default value of the parameter if not found
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> asUInts(std::vector<unsigned int> def) const {
        std::string vals = asString("");
        std::vector<unsigned int> result;
        std::istringstream stream(vals);
        std::string str;
        std::string message = "Value " + vals +  " cannot be casted to unsigned ints.";
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                int value = std::stoi(str);
                if (value < 0) {
                    THROW_IE_EXCEPTION << message;
                }
                result.push_back(static_cast<unsigned int>(value));
            } catch (...) {
                THROW_IE_EXCEPTION << message;
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of unsigned int values for the given parameter
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> asUInts() const {
        std::string vals = asString();
        std::vector<unsigned int> result;
        std::istringstream stream(vals);
        std::string str;
        std::string message = "Value " + vals +  " cannot be casted to unsigned ints.";
        while (getline(stream, str, ',')) {
            try {
                int value = std::stoi(str);
                if (value < 0) {
                    THROW_IE_EXCEPTION << message;
                }
                result.push_back(static_cast<unsigned int>(value));
            } catch (...) {
                THROW_IE_EXCEPTION << message;
            }
        }
        return result;
    }

    /**
     * @brief Returns an boolean value for the given parameter.
     * The valid values are (true, false, 1, 0).
     * @param def Default value of the parameter if not found
     * @return An bool value for the specified parameter
     */
    bool asBool(bool def) const {
        std::string val = asString(std::to_string(def));
        std::string loweredCaseValue;
        std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
            return std::tolower(value);
        });

        bool result = false;

        if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
            // attempting parse using non alpha bool
            return static_cast<bool>(asInt(def));
        }

        return result;
    }

    /**
     * @brief Returns an boolean value for the given parameter.
     * The valid values are (true, false, 1, 0).
     * @return An bool value for the specified parameter
     */
    bool asBool() const {
        std::string val = asString();
        std::string loweredCaseValue;
        std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
            return std::tolower(value);
        });

        bool result = false;

        if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
            // attempting parse using non alpha bool
            return static_cast<bool>(asInt());
        }

        return result;
    }

private:
    bool initialized;
    std::string value;
};

}  // namespace InferenceEngine
