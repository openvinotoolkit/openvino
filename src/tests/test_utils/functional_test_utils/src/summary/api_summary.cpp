// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/api_summary.hpp"

#include <pugixml.hpp>

#include "common_test_utils/file_utils.hpp"

using namespace ov::test::utils;

#ifdef _WIN32
#    define getpid _getpid
#endif

ApiSummary* ApiSummary::p_instance = nullptr;
ApiSummaryDestroyer ApiSummary::destroyer;
const std::map<ov_entity, std::string> ApiSummary::apiInfo({
    {ov_entity::ov_infer_request, "ov_infer_request"},
    {ov_entity::ov_plugin, "ov_plugin"},
    {ov_entity::ov_compiled_model, "ov_compiled_model"},
    {ov_entity::undefined, "undefined"},
});

ApiSummaryDestroyer::~ApiSummaryDestroyer() {
    delete p_instance;
}

void ApiSummaryDestroyer::initialize(ApiSummary* p) {
    p_instance = p;
}

ApiSummary::ApiSummary() : apiStats() {
    reportFilename = ov::test::utils::API_REPORT_FILENAME;
}

ApiSummary& ApiSummary::getInstance() {
    if (!p_instance) {
        p_instance = new ApiSummary();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

void ApiSummary::updateStat(ov_entity entity,
                            const std::string& target_device,
                            PassRate::Statuses status,
                            double rel_influence_coef) {
    if (apiStats.empty()) {
        std::string outputFilePath = outputFolder + std::string(ov::test::utils::FileSeparator) + reportFilename +
                                     ov::test::utils::REPORT_EXTENSION;
        const bool fileExists = ov::test::utils::fileExists(outputFilePath);
        if (extendReport && !isReported && fileExists) {
            getStatisticFromReport(outputFilePath);
        }
    }
    std::string real_device = target_device.substr(0, target_device.find(':'));
    if (deviceName.empty()) {
        deviceName = real_device == target_device ? target_device : target_device.substr(target_device.find(':') + 1);
    }
    if (apiStats.find(entity) == apiStats.end()) {
        apiStats.insert({entity, {{real_device, PassRate()}}});
    }
    auto& cur_stat = apiStats[entity];
    if (cur_stat.find(real_device) == cur_stat.end()) {
        cur_stat.insert({real_device, PassRate()});
    }
    if (cur_stat[real_device].isCrashReported) {
        cur_stat[real_device].crashed--;
        cur_stat[real_device].isCrashReported = false;
    } else {
        cur_stat[real_device].rel_all += rel_influence_coef;
    }
    if (cur_stat[real_device].isHangReported) {
        cur_stat[real_device].isHangReported = false;
        return;
    }
    switch (status) {
    case PassRate::Statuses::SKIPPED: {
        cur_stat[real_device].skipped++;
        break;
    }
    case PassRate::Statuses::PASSED: {
        if (!cur_stat[real_device].isImplemented) {
            cur_stat[real_device].isImplemented = true;
        }
        cur_stat[real_device].passed++;
        cur_stat[real_device].rel_passed += rel_influence_coef;
        break;
    }
    case PassRate::Statuses::HANGED: {
        cur_stat[real_device].hanged++;
        cur_stat[real_device].isHangReported = true;
        break;
    }
    case PassRate::Statuses::FAILED: {
        cur_stat[real_device].failed++;
        break;
    }
    case PassRate::Statuses::CRASHED:
        cur_stat[real_device].crashed++;
        cur_stat[real_device].isCrashReported = true;
        break;
    }
}

ov_entity ApiSummary::getOvEntityByName(const std::string& name) {
    for (const auto& api : apiInfo) {
        if (name == api.second) {
            return api.first;
        }
    }
    return ov_entity::undefined;
}

void ApiSummary::getStatisticFromReport(const std::string& filePath) {
    pugi::xml_document doc;

    doc.load_file(filePath.c_str());
    pugi::xml_node root = doc.child("report");

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.child(deviceName.c_str());
    for (auto& entityNode : currentDeviceNode.children()) {
        std::string entityName = entityNode.name();
        ov_entity entity = getOvEntityByName(entityName);
        for (const auto& realDeviceNode : entityNode.children()) {
            std::string realDeviceName = realDeviceNode.name();
            auto p = std::stoi(realDeviceNode.attribute("passed").value());
            auto f = std::stoi(realDeviceNode.attribute("failed").value());
            auto s = std::stoi(realDeviceNode.attribute("skipped").value());
            auto c = std::stoi(realDeviceNode.attribute("crashed").value());
            auto h = std::stoi(realDeviceNode.attribute("hanged").value());
            auto rel_p = std::stoi(realDeviceNode.attribute("relative_passed").value());
            auto rel_all = std::stoi(realDeviceNode.attribute("relative_all").value());
            PassRate entity_stat(p, f, s, c, h, rel_p, rel_all);
            if (apiStats.find(entity) == apiStats.end()) {
                apiStats.insert({entity, {}});
            }
            apiStats[entity].insert({realDeviceName, entity_stat});
        }
    }
}

void ApiSummary::saveReport() {
    std::string filename = reportFilename;
    if (saveReportWithUniqueName) {
        auto processId = std::to_string(getpid());
        filename += "_" + processId + "_" + ts;
    }
    filename += ov::test::utils::REPORT_EXTENSION;

    if (!ov::util::directory_exists(outputFolder)) {
        ov::util::create_directory_recursive(outputFolder);
    }

    std::string outputFilePath = outputFolder + std::string(ov::test::utils::FileSeparator) + filename;

    auto& summary = ApiSummary::getInstance();
    auto stats = summary.getApiStats();

    pugi::xml_document doc;

    const bool fileExists = ov::test::utils::fileExists(outputFilePath);

    time_t rawtime;
    struct tm* timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime);  // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (fileExists) {
        doc.load_file(outputFilePath.c_str());
        root = doc.child("report");
        // Ugly but shorter than to write predicate for find_atrribute() to update existing one
        root.remove_attribute("timestamp");
        root.append_attribute("timestamp").set_value(timeNow);

        root.remove_child("api_list");
        root.child("results").remove_child(deviceName.c_str());
    } else {
        root = doc.append_child("report");
        root.append_attribute("timestamp").set_value(timeNow);
        root.append_child("results");
    }

    pugi::xml_node opsNode = root.append_child("api_list");
    for (const auto& api : apiInfo) {
        std::string name = api.second;
        pugi::xml_node entry = opsNode.append_child(name.c_str());
        (void)entry;
    }

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.append_child(summary.deviceName.c_str());
    std::unordered_set<std::string> opList;
    for (const auto& stat_entity : stats) {
        pugi::xml_node currentEntity = currentDeviceNode.append_child(apiInfo.at(stat_entity.first).c_str());
        for (const auto& stat_device : stat_entity.second) {
            pugi::xml_node entry = currentEntity.append_child(stat_device.first.c_str());
            entry.append_attribute("implemented").set_value(stat_device.second.isImplemented);
            entry.append_attribute("passed").set_value(static_cast<unsigned long long>(stat_device.second.passed));
            entry.append_attribute("failed").set_value(static_cast<unsigned long long>(stat_device.second.failed));
            entry.append_attribute("skipped").set_value(static_cast<unsigned long long>(stat_device.second.skipped));
            entry.append_attribute("crashed").set_value(static_cast<unsigned long long>(stat_device.second.crashed));
            entry.append_attribute("hanged").set_value(static_cast<unsigned long long>(stat_device.second.hanged));
            entry.append_attribute("passrate").set_value(stat_device.second.getPassrate());
            entry.append_attribute("relative_passed")
                .set_value(static_cast<unsigned long long>(stat_device.second.rel_passed));
            entry.append_attribute("relative_all")
                .set_value(static_cast<unsigned long long>(stat_device.second.rel_all));
            entry.append_attribute("relative_passrate").set_value(stat_device.second.getRelPassrate());
        }
    }

    auto exitTime = std::chrono::system_clock::now() + std::chrono::seconds(saveReportTimeout);
    bool result = false;
    do {
        result = doc.save_file(outputFilePath.c_str());
    } while (!result && std::chrono::system_clock::now() < exitTime);

    if (!result) {
        std::string errMessage = "Failed to write report to " + outputFilePath;
        throw std::runtime_error(errMessage);
    } else {
        isReported = true;
    }
}
