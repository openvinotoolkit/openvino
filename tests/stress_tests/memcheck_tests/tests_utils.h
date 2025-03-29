// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../common/tests_utils.h"

#include <array>
#include <pugixml.hpp>

// Measure values
enum MeasureValue {
    VMRSS = 0, VMHWM, VMSIZE, VMPEAK, THREADS, MeasureValueMax
};
// Measure values headers
const std::array<std::string, MeasureValueMax> MeasureValueHeader{"VMRSS", "VMHWM", "VMSIZE", "VMPEAK", "THREADS"};

namespace util {
    template<typename Type>
    inline std::string get_measure_values_as_str(const std::array<Type, MeasureValueMax> &array,
                                                 const std::string &delimiter = "\t\t") {
        std::string str = std::to_string(*array.begin());
        for (auto it = array.begin() + 1; it != array.end(); it++)
            str += delimiter + std::to_string(*it);
        return str;
    }

    inline std::string get_measure_values_headers(const std::string &delimiter = "\t\t") {
        std::string str = *MeasureValueHeader.begin();
        for (auto it = MeasureValueHeader.begin() + 1; it != MeasureValueHeader.end(); it++)
            str += delimiter + *it;
        return str;
    }
}

class MemCheckEnvironment {
private:
    pugi::xml_document _refs_config;

    MemCheckEnvironment() = default;

    MemCheckEnvironment(const MemCheckEnvironment &) = delete;

    MemCheckEnvironment &operator=(const MemCheckEnvironment &) = delete;

public:
    static MemCheckEnvironment &Instance() {
        static MemCheckEnvironment env;
        return env;
    }

    const pugi::xml_document &getRefsConfig() {
        return _refs_config;
    }

    void setRefsConfig(const pugi::xml_document &refs_config) {
        _refs_config.reset(refs_config);
    }
};

class TestReferences {
private:
    std::vector<std::string> model_name_v, test_name_v, device_v, precision_v;
    std::vector<long> vmsize_v, vmpeak_v, vmrss_v, vmhwm_v;
public:
    std::array<long, MeasureValueMax> references;

    TestReferences() {
        std::fill(references.begin(), references.end(), -1);

        // Parse RefsConfig from MemCheckEnvironment
        const pugi::xml_document &refs_config = MemCheckEnvironment::Instance().getRefsConfig();
        auto values = refs_config.child("attributes").child("models");
        for (pugi::xml_node node = values.first_child(); node; node = node.next_sibling()) {
            for (pugi::xml_attribute_iterator ait = node.attributes_begin(); ait != node.attributes_end(); ait++) {
                if (strncmp(ait->name(), "path", strlen(ait->name())) == 0) {
                    model_name_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "precision", strlen(ait->name())) == 0) {
                    precision_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "test", strlen(ait->name())) == 0) {
                    test_name_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "device", strlen(ait->name())) == 0) {
                    device_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "vmsize", strlen(ait->name())) == 0) {
                    vmsize_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmpeak", strlen(ait->name())) == 0) {
                    vmpeak_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmrss", strlen(ait->name())) == 0) {
                    vmrss_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmhwm", strlen(ait->name())) == 0) {
                    vmhwm_v.push_back(std::atoi(ait->value()));
                }
            }
        }
    }

    void collect_vm_values_for_test(std::string test_name, TestCase test_params) {
        for (size_t i = 0; i < test_name_v.size(); i++)
            if (test_name_v[i] == test_name)
                if (model_name_v[i] == test_params.model_name)
                    if (device_v[i] == test_params.device)
                        if (precision_v[i] == test_params.precision) {
                            references[VMSIZE] = vmsize_v[i];
                            references[VMPEAK] = vmpeak_v[i];
                            references[VMRSS] = vmrss_v[i];
                            references[VMHWM] = vmhwm_v[i];
                        }
    }
};
