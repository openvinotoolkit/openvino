#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
    std::string modelPath = "/tmp/myModel.xml";
    std::string deviceName = "GNA";
    std::map<std::string, std::string> deviceConfig;
    InferenceEngine::Core ie;
//! [part3]
    // Get list of supported metrics
    std::vector<std::string> keys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));

    // Find 'IMPORT_EXPORT_SUPPORT' metric in supported metrics
    auto it = std::find(keys.begin(), keys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));

    // If metric 'IMPORT_EXPORT_SUPPORT' exists, check it's value
    bool cachingSupported = (it != keys.end()) && ie.GetMetric(deviceName, METRIC_KEY(IMPORT_EXPORT_SUPPORT));
//! [part3]
    return 0;
}
