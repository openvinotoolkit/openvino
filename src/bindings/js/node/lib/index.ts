const ov = require('../build/Release/ov_node_addon.node');
import loadModel from './node-model';
import { Tensor, Shape } from 'openvinojs-common'

export async function init() { 
  return {
    loadModel, 
    Tensor,
    Shape,
    getDescriptionString,
    getModelName,
  };
};

function getDescriptionString(): string {
    return ov.getDescriptionString();
};

function getModelName(model_path: string): string {
    const model = new ov.Model().read_model(model_path);
    return model.get_name();
}
