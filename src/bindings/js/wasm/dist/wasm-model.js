"use strict";
var __classPrivateFieldSet = (this && this.__classPrivateFieldSet) || function (receiver, state, value, kind, f) {
    if (kind === "m") throw new TypeError("Private method is not writable");
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
    return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
};
var __classPrivateFieldGet = (this && this.__classPrivateFieldGet) || function (receiver, state, kind, f) {
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a getter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
    return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var _WASMModel_ov, _WASMModel_originalModel;
Object.defineProperty(exports, "__esModule", { value: true });
const openvino_wasm_js_1 = __importDefault(require("../bin/openvino_wasm.js"));
const openvinojs_common_1 = require("openvinojs-common");
const helpers_js_1 = require("./helpers.js");
class WASMModel {
    constructor(originalModel) {
        _WASMModel_ov.set(this, openvino_wasm_js_1.default);
        _WASMModel_originalModel.set(this, void 0);
        __classPrivateFieldSet(this, _WASMModel_originalModel, originalModel, "f");
    }
    infer(tensorOrDataArray, shape) {
        const tensor = tensorOrDataArray instanceof openvinojs_common_1.Tensor
            ? tensorOrDataArray
            : new openvinojs_common_1.Tensor('uint8', tensorOrDataArray, shape);
        const wrapper = new Promise((resolve, reject) => {
            let outputTensor;
            try {
                outputTensor = runInference(__classPrivateFieldGet(this, _WASMModel_ov, "f"), __classPrivateFieldGet(this, _WASMModel_originalModel, "f"), tensor);
            }
            catch (e) {
                return reject(e);
            }
            outputTensor ? resolve(outputTensor) : reject(null);
        });
        return wrapper;
    }
}
_WASMModel_ov = new WeakMap(), _WASMModel_originalModel = new WeakMap();
async function loadModel(xmlPath, binPath, shapeData, layout) {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
        throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');
    const xmlData = await (0, helpers_js_1.getFileDataAsArray)(xmlPath);
    const binData = await (0, helpers_js_1.getFileDataAsArray)(binPath);
    const timestamp = Date.now();
    const xmlFilename = `m${timestamp}.xml`;
    const binFilename = `m${timestamp}.bin`;
    // Uploading and creating files on virtual WASM filesystem
    (0, helpers_js_1.uploadFile)(openvino_wasm_js_1.default, xmlFilename, xmlData);
    (0, helpers_js_1.uploadFile)(openvino_wasm_js_1.default, binFilename, binData);
    const shape = shapeData instanceof openvinojs_common_1.Shape ? shapeData : new openvinojs_common_1.Shape(...shapeData);
    const originalShape = (0, helpers_js_1.convertShape)(openvino_wasm_js_1.default, shape);
    const originalModel = new openvino_wasm_js_1.default.Session(xmlFilename, binFilename, originalShape.obj, layout);
    return new WASMModel(originalModel);
}
exports.default = loadModel;
function runInference(ov, model, tensor) {
    let originalTensor;
    let originalOutputTensor;
    try {
        console.time('== Inference time');
        originalTensor = (0, helpers_js_1.convertTensor)(ov, tensor);
        originalOutputTensor = model.infer(originalTensor.obj);
        console.timeEnd('== Inference time');
    }
    finally {
        if (originalTensor)
            originalTensor.free();
    }
    return originalOutputTensor ? (0, helpers_js_1.parseOriginalTensor)(ov, originalOutputTensor) : null;
}
//# sourceMappingURL=wasm-model.js.map