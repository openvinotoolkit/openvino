import ov from './openvino_wasm.mjs';
import loadModel from './wasm-model.mjs';
import { Tensor, Shape } from 'openvinojs-common';
export { loadModel, Tensor, Shape, getVersionString, getDescriptionString, };
function getVersionString() {
    return ov.getVersionString();
}
function getDescriptionString() {
    return ov.getDescriptionString();
}
//# sourceMappingURL=index.mjs.map