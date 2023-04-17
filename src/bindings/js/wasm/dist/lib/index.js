"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.init = void 0;
const openvino_wasm_js_1 = __importDefault(require("../bin/openvino_wasm.js"));
const wasm_model_js_1 = __importDefault(require("./wasm-model.js"));
const openvinojs_common_1 = require("openvinojs-common");
async function init() {
    const ov = await (0, openvino_wasm_js_1.default)();
    function getVersionString() {
        return ov.getVersionString();
    }
    function getDescriptionString() {
        return ov.getDescriptionString();
    }
    return {
        loadModel: wasm_model_js_1.default,
        Tensor: openvinojs_common_1.Tensor,
        Shape: openvinojs_common_1.Shape,
        getVersionString,
        getDescriptionString,
    };
}
exports.init = init;
;
//# sourceMappingURL=index.js.map