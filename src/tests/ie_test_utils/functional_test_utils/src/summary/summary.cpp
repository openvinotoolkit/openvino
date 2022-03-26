// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/summary.hpp"

using namespace ov::test::utils;

bool Summary::extendReport = false;
bool Summary::saveReportWithUniqueName = false;
const char* Summary::outputFolder = ".";

void Summary::setDeviceName(std::string device) {
    deviceName = device;
}

std::map<std::string, PassRate> Summary::getStatisticFromReport() {
    return std::map<std::string, PassRate>();
}

std::string Summary::getDeviceName() const {
    return deviceName;
}

void Summary::setReportFilename(const std::string& val) {
    reportFilename = val.c_str();
}