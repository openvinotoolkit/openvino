import loadModel from './wasm-model.js';
import { Tensor, Shape } from 'openvinojs-common';
export declare function init(): Promise<{
    loadModel: typeof loadModel;
    Tensor: typeof Tensor;
    Shape: typeof Shape;
    getVersionString: () => string;
    getDescriptionString: () => string;
}>;
