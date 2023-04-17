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
var _WASMModel_ov, _WASMModel_originalModel;
import ov from './openvino_wasm.mjs';
import { Tensor, Shape } from 'openvinojs-common';
import { getFileDataAsArray, uploadFile, convertShape, convertTensor, parseOriginalTensor, } from './helpers.mjs';
class WASMModel {
    constructor(originalModel) {
        _WASMModel_ov.set(this, ov);
        _WASMModel_originalModel.set(this, void 0);
        __classPrivateFieldSet(this, _WASMModel_originalModel, originalModel, "f");
    }
    infer(tensorOrDataArray, shape) {
        const tensor = tensorOrDataArray instanceof Tensor
            ? tensorOrDataArray
            : new Tensor('uint8', tensorOrDataArray, shape);
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
export default async function loadModel(xmlPath, binPath, shapeData, layout) {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
        throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');
    const xmlData = await getFileDataAsArray(xmlPath);
    const binData = await getFileDataAsArray(binPath);
    const timestamp = Date.now();
    const xmlFilename = `m${timestamp}.xml`;
    const binFilename = `m${timestamp}.bin`;
    // Uploading and creating files on virtual WASM filesystem
    uploadFile(ov, xmlFilename, xmlData);
    uploadFile(ov, binFilename, binData);
    const shape = shapeData instanceof Shape ? shapeData : new Shape(...shapeData);
    const originalShape = convertShape(ov, shape);
    const originalModel = new ov.Session(xmlFilename, binFilename, originalShape.obj, layout);
    return new WASMModel(originalModel);
}
function runInference(ov, model, tensor) {
    let originalTensor;
    let originalOutputTensor;
    try {
        console.time('== Inference time');
        originalTensor = convertTensor(ov, tensor);
        originalOutputTensor = model.infer(originalTensor.obj);
        console.timeEnd('== Inference time');
    }
    finally {
        if (originalTensor)
            originalTensor.free();
    }
    return originalOutputTensor ? parseOriginalTensor(ov, originalOutputTensor) : null;
}
//# sourceMappingURL=wasm-model.mjs.map