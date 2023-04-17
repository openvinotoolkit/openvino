import loadModel from './wasm-model.js';
import { Tensor, Shape } from 'openvinojs-common';
export { loadModel, Tensor, Shape, getVersionString, getDescriptionString, };
declare function getVersionString(): string;
declare function getDescriptionString(): string;
