// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pugixml.hpp>

#include "functional_test_utils/summary/api_summary.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace ov::test::utils;

#ifdef _WIN32
# define getpid _getpid
#endif

ApiSummary *ApiSummary::p_instance = nullptr;
ApiSummaryDestroyer ApiSummary::destroyer;
const std::map<ov_entity, std::string> ApiSummary::apiInfo({
    { ov_entity::ov_infer_request, "ie_infer_request"},
    { ov_entity::ov_plugin, "ov_plugin"},
    { ov_entity::ov_compiled_model, "ov_compiled_model"},
    { ov_entity::ie_infer_request, "ie_infer_request"},
    { ov_entity::ie_plugin, "ie_plugin"},
    { ov_entity::ie_executable_network, "ie_executable_network"},
});

ApiSummaryDestroyer::~ApiSummaryDestroyer() {
    delete p_instance;
}

void ApiSummaryDestroyer::initialize(ApiSummary *p) {
    p_instance = p;
}

ApiSummary::ApiSummary() : apiStats() {
    reportFilename = CommonTestUtils::API_REPORT_FILENAME;
}

ApiSummary &ApiSummary::getInstance() {
    if (!p_instance) {
        p_instance = new ApiSummary();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

void ApiSummary::updateStat(ov_entity entity, const std::string& target_device, PassRate::Statuses status) {
    if (extendReport && !isReported) {
//        apiStats = getStatisticFromReport();
    }
    std::string real_device = target_device.substr(0, target_device.find(':'));
    if (apiStats.find(entity) == apiStats.end()) {
        apiStats.insert({entity, {{real_device, PassRate()}}});
    }
    auto& cur_stat = apiStats[entity];
    if (cur_stat.find(real_device) == cur_stat.end()) {
        cur_stat.insert({real_device, PassRate()});
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
            cur_stat[real_device].crashed--;
            break;
        }
        case PassRate::Statuses::HANGED: {
            cur_stat[real_device].hanged++;
            cur_stat[real_device].crashed--;
            break;
        }
        case PassRate::Statuses::FAILED: {
            cur_stat[real_device].failed++;
            cur_stat[real_device].crashed--;
            break;
        }
        case PassRate::Statuses::CRASHED:
            cur_stat[real_device].crashed++;
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

void ApiSummary::getStatisticFromReport() {
    pugi::xml_document doc;

    std::ifstream file;
    file.open(reportFilename);

    pugi::xml_node root;
    doc.load_file(reportFilename);
    root = doc.child("report");

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.child(deviceName.c_str());
//    for (const auto& ie_entity_node : currentDeviceNode) {
//        for (const auto& attr : ie_entity_node.attributes()) {
//            std::cout << attr.name() << " " << attr.value() << std::endl;
//        }
//        std::string ie_entity_name = ie_entity_node.value();
//        ov_entity entity = getOvEntityByName(ie_entity_name);
//        for (const auto& real_device_node : ie_entity_node) {
//            std::string real_device_name = real_device_node.name();
//            auto p = std::stoi(real_device_node.attribute("passed").value());
//            auto f = std::stoi(real_device_node.attribute("failed").value());
//            auto s = std::stoi(real_device_node.attribute("skipped").value());
//            auto c = std::stoi(real_device_node.attribute("crashed").value());
//            auto h = std::stoi(real_device_node.attribute("hanged").value());
//            PassRate entity_stat(p, f, s, c, h);
//            if (apiStats.find(entity) == apiStats.end()) {
//                apiStats.insert({entity, {}});
//            }
//            apiStats[entity].insert({real_device_name, entity_stat});
//        }
//    }
}

void ApiSummary::saveReport() {
    std::string filename = reportFilename;
    if (saveReportWithUniqueName) {
        auto processId = std::to_string(getpid());
        filename += "_" + processId + "_" + std::string(CommonTestUtils::GetTimestamp());
    }
    filename += CommonTestUtils::REPORT_EXTENSION;

    if (!CommonTestUtils::directoryExists(outputFolder)) {
        CommonTestUtils::createDirectoryRecursive(outputFolder);
    }

    std::string outputFilePath = outputFolder + std::string(CommonTestUtils::FileSeparator) + filename;

    auto &summary = ApiSummary::getInstance();
    auto stats = summary.getApiStats();

    pugi::xml_document doc;

    const bool fileExists = CommonTestUtils::fileExists(outputFilePath);

    if (extendReport && fileExists && !isReported) {
        getStatisticFromReport();
    }

    time_t rawtime;
    struct tm *timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime); // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (fileExists) {
        doc.load_file(outputFilePath.c_str());
        root = doc.child("report");
        //Ugly but shorter than to write predicate for find_atrribute() to update existing one
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
    for (const auto &api : apiInfo) {
        std::string name = api.second;
        pugi::xml_node entry = opsNode.append_child(name.c_str());
        (void) entry;
    }

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.append_child(summary.deviceName.c_str());
    std::unordered_set<std::string> opList;
    for (const auto &stat_entity : stats) {
        pugi::xml_node currentEntity = currentDeviceNode.append_child(apiInfo.at(stat_entity.first).c_str());
        for (const auto& stat_device : stat_entity.second) {
            pugi::xml_node entry = currentEntity.append_child(stat_device.first.c_str());
            entry.append_attribute("implemented").set_value(stat_device.second.isImplemented);
            entry.append_attribute("passed").set_value(stat_device.second.passed);
            entry.append_attribute("failed").set_value(stat_device.second.failed);
            entry.append_attribute("skipped").set_value(stat_device.second.skipped);
            entry.append_attribute("crashed").set_value(stat_device.second.crashed);
            entry.append_attribute("hanged").set_value(stat_device.second.hanged);
            entry.append_attribute("passrate").set_value(stat_device.second.getPassrate());
        }
    }

//    if (extendReport && fileExists) {
//        auto opStataFromReport = summary.getStatisticFromReport();
//        for (auto &item : opStataFromReport) {
//            pugi::xml_node entry;
//            if (opList.find(item.first) == opList.end()) {
//                entry = currentDeviceNode.append_child(item.first.c_str());
//                entry.append_attribute("implemented").set_value(item.second.isImplemented);
//                entry.append_attribute("passed").set_value(item.second.passed);
//                entry.append_attribute("failed").set_value(item.second.failed);
//                entry.append_attribute("skipped").set_value(item.second.skipped);
//                entry.append_attribute("crashed").set_value(item.second.crashed);
//                entry.append_attribute("hanged").set_value(item.second.hanged);
//                entry.append_attribute("passrate").set_value(item.second.getPassrate());
//            } else {
//                entry = currentDeviceNode.child(item.first.c_str());
//                auto implStatus = entry.attribute("implemented").value() == std::string("true") ? true : false;
//                auto p = std::stoi(entry.attribute("passed").value()) + item.second.passed;
//                auto f = std::stoi(entry.attribute("failed").value()) + item.second.failed;
//                auto s = std::stoi(entry.attribute("skipped").value()) + item.second.skipped;
//                auto c = std::stoi(entry.attribute("crashed").value()) + item.second.crashed;
//                auto h = std::stoi(entry.attribute("hanged").value()) + item.second.hanged;
//                PassRate obj(p, f, s, c, h);
//
//                (implStatus || obj.isImplemented)
//                ? entry.attribute("implemented").set_value(true)
//                : entry.attribute("implemented").set_value(false);
//                entry.attribute("passed").set_value(obj.passed);
//                entry.attribute("failed").set_value(obj.failed);
//                entry.attribute("skipped").set_value(obj.skipped);
//                entry.attribute("crashed").set_value(obj.crashed);
//                entry.attribute("hanged").set_value(obj.hanged);
//                entry.attribute("passrate").set_value(obj.getPassrate());
//            }
//        }
//    }

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

