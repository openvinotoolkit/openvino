const { IECore } = require('bindings')('InferenceEngine');

const ieCore = new IECore();

const availableDevices = ieCore.getAvailableDevices();


console.log('Available devices:');
for (device of availableDevices) {
    console.log("\tDevice: ", device);
    console.log("\tMetrics:");
    for (supportedMetricName of ieCore.getMetric(device, 'SUPPORTED_METRICS')) {
        try {
            const metric = ieCore.getMetric(device, supportedMetricName);
            console.log(`\t\t${supportedMetricName}: ${metric}`);
        } catch (e) {
            console.log(`\t\t${supportedMetricName}: UNSUPPORTED TYPE`);
        }
    }
}
