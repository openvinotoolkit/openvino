const { IECore } = require('bindings')('InferenceEngineAddon');

const ieCore = new IECore();

const availableDevices = ieCore.getAvailableDevices();

console.log(availableDevices);
