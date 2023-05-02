import { Tensor, Shape } from 'openvinojs-common';

import type { ovNodeModule, NodeModel, NodeTensor } from './types';
import type { ITensor, IShape, IModel, PrecisionSupportedType } from 'openvinojs-common';

const ovNode: ovNodeModule = require('../build/Release/ov_node_addon.node');

export default async function loadModel(xmlPath: string, binPath: string): Promise<IModel> {
    if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
        throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');

    const model = new ovNode.Model().read_model(xmlPath).compile("CPU");
    return new CommonModel(ovNode, model);
}

class CommonModel implements IModel {
    #ovNode: ovNodeModule;
    #nodeModel: NodeModel;

    constructor(ovNode: ovNodeModule, nodeModel: NodeModel) {
        this.#ovNode = ovNode;
        this.#nodeModel = nodeModel;
    }

    async infer(tensorOrDataArray: ITensor | number[], shape: IShape): Promise<ITensor> {
        const tensor_data = tensorOrDataArray instanceof Array
            ? Float32Array.from(tensorOrDataArray) : tensorOrDataArray.data;
        const precision = tensorOrDataArray instanceof Tensor
            ? tensorOrDataArray.precision : "f32";

        const nodeTensor = new this.#ovNode.Tensor(precision, shape.data, tensor_data);
        const output = this.#nodeModel.infer(nodeTensor);

        return parseNodeTensor(output);

    }
}

function parseNodeTensor(nodeTensor: NodeTensor): Tensor {
    const precision = nodeTensor.getPrecision() as PrecisionSupportedType;
    const data = nodeTensor.data;
    const shape = new Shape(nodeTensor.getShape().getData());
    return new Tensor(precision, data, shape);
}
