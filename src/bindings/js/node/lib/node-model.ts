import { Tensor, TypedArray, Shape } from 'openvinojs-common';
import type { ITensor, IShape, IModel } from 'openvinojs-common';
const ovNode = require('../build/Release/ov_node_addon.node');


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

        const nodeTensor = new this.#ovNode.Tensor("f32", Int32Array.from(shape.data), tensor_data);
        // TO_DO check if I have to use int32array for shape
        // TO_DO use precision from ITensor

        const output = this.#nodeModel.infer(nodeTensor);

        return parseNodeTensor(output);

    }
}

function parseNodeTensor(output: NodeTensor): Tensor {
    const precision = "float32";
    const data = output.data;
    const shape = new Shape(output.getShape().getData()); // TO_DO output.getShape() returns ShapeLite() from Node

    return new Tensor(precision, data, shape);
}


export interface ovNodeModule {
    Tensor: new (precision: string, shape: number[] | Int32Array, tensor_data: TypedArray) => NodeTensor, //TO_DO tensor_data  from ITensor 
    Model: new () => NodeModel,
    Shape: new (dimension: number, data: Uint32Array) => ShapeLite,
    getDescriptionString(): string
};

export interface NodeTensor {
    data: TypedArray;
    getData(): TypedArray;
    getPrecision(): string;
    getShape(): ShapeLite;
};

export interface ShapeLite {
    getDim(): number;
    getData(): number;
    shapeSize(): number;
};

export interface NodeModel {
    read_model(): NodeModel;
    infer(tensor: NodeTensor): NodeTensor;
}
