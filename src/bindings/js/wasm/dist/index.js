"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDescriptionString = exports.getVersionString = exports.Shape = exports.Tensor = exports.loadModel = void 0;
const openvino_wasm_js_1 = __importDefault(require("../bin/openvino_wasm.js"));
const wasm_model_js_1 = __importDefault(require("./wasm-model.js"));
exports.loadModel = wasm_model_js_1.default;
const openvinojs_common_1 = require("openvinojs-common");
Object.defineProperty(exports, "Tensor", { enumerable: true, get: function () { return openvinojs_common_1.Tensor; } });
Object.defineProperty(exports, "Shape", { enumerable: true, get: function () { return openvinojs_common_1.Shape; } });
function getVersionString() {
    return openvino_wasm_js_1.default.getVersionString();
}
exports.getVersionString = getVersionString;
function getDescriptionString() {
    return openvino_wasm_js_1.default.getDescriptionString();
}
exports.getDescriptionString = getDescriptionString;
//# sourceMappingURL=index.js.map