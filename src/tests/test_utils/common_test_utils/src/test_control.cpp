// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_control.hpp"

#include "openvino/util/common_util.hpp"
#include "precomp.hpp"

using namespace std;

static unordered_set<string>& get_blacklist(const string& backend) {
    static unordered_map<string, unordered_set<string>> s_blacklists;
    return s_blacklists[backend];
}

string ov::prepend_disabled(const string& backend_name, const string& test_name, const string& manifest) {
    string rc = test_name;
    unordered_set<string>& blacklist = get_blacklist(backend_name);
    if (blacklist.empty() && !manifest.empty()) {
        ifstream f(manifest);
        string line;
        while (getline(f, line)) {
            size_t pound_pos = line.find('#');
            line = (pound_pos > line.size()) ? line : line.substr(0, pound_pos);
            line = ov::util::trim(line);
            if (line.size() > 1) {
                blacklist.insert(line);
            }
        }
    }
    string compound_test_name = backend_name + "." + test_name;
    if (blacklist.find(test_name) != blacklist.end() || blacklist.find(compound_test_name) != blacklist.end()) {
        rc = "DISABLED_" + test_name;
    }
    return rc;
}

string ov::combine_test_backend_and_case(const string& backend_name, const string& test_casename) {
    if (backend_name == test_casename) {
        return backend_name;
    } else {
        return backend_name + "/" + test_casename;
    }
}
