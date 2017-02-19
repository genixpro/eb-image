/*
 Electric Brain is an easy to use platform for machine learning.
 Copyright (C) 2016 Electric Brain Software Corporation

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

"use strict";

const
    assert = require('assert'),
    EBNeuralNetworkComponentBase = require('./../../../shared/components/architecture/EBNeuralNetworkComponentBase'),
    EBTorchModule = require('../../../shared/models/EBTorchModule'),
    EBTorchNode = require('../../../shared/models/EBTorchNode'),
    EBTensorSchema = require('../../../shared/models/EBTensorSchema'),
    sharp = require('sharp'),
    underscore = require('underscore');

/**
 * The neural network image component is used to process images.
 */
class EBNeuralNetworkImageComponent extends EBNeuralNetworkComponentBase
{
    /**
     * Constructor
     */
    constructor()
    {
        super();
    }


    /**
     * Method returns the tensor schema data of this format
     *
     * @param {EBSchema} schema The regular schema from which we will determine the tensor schema
     * @returns {EBTensorSchema} The mapping of tensors.
     */
    getTensorSchema(schema)
    {
        const size = EBNeuralNetworkImageComponent.getImageSizeForSchema(schema);
        const channels = EBNeuralNetworkImageComponent.getImageChannelsForSchema(schema);
        
        return new EBTensorSchema({
            "type": "tensor",
            "variableName": schema.variableName,
            "tensorDimensions": [
                {
                    "size": size.width,
                    "label": "width"
                },
                {
                    "size": size.height,
                    "label": "height"
                },
                {
                    "size": channels,
                    "label": "channels"
                }
            ],
            "tensorMap": {
                [schema.variableName]: {
                    start: 1,
                    size: size.width * size.height * channels
                }
            }
        });
    }


    /**
     * Method generates Lua code to create a tensor from the JSON of this variable
     *
     * @param {EBSchema} schema The schema to generate this conversion code for.
     * @param {string} name The name of the lua function to be generated.
     */
    generateTensorInputCode(schema, name)
    {
        // First, ensure that the schema we are dealing with is an image
        assert(schema.isField);
        assert(schema.metadata.mainInterpretation === 'image');

        let code = '';
        code += `local ${name} = function (input)\n`;
        code += `    local decoded = mime.unb64(input)\n`;
        code += `    local bytes = torch.ByteStorage()\n`;
        code += `    bytes:string(decoded)\n`;
        code += `    local tensor = torch.ByteTensor(bytes)\n`;
        code += `    return image.decompress(tensor, 3)\n`;
        code += `end\n`;
        return code;
    }


    /**
     * Method generates Lua code to turn this variable back into a tensor
     */
    generateTensorOutputCode()
    {
        throw new Error("Cant create output code for the image neural network component");
    }


    /**
     * Method generates Lua code that can prepare a combined batch tensor from
     * multiple samples.
     *
     * @param {EBSchema} schema The schema to generate this conversion code for
     * @param {string} name The name of the Lua function to be generated
     */
    generatePrepareBatchCode(schema, name)
    {
        // First, ensure that the schema we are dealing with is an image
        assert(schema.isField);
        assert(schema.metadata.mainInterpretation === 'image');

        let code = '';

        code += `local ${name} = function (input)\n`;
        code += `    local height, width = input[1]:size(2), input[1]:size(3)\n`;
        code += `    local batch = torch.zeros(#samples,3,height ,width )\n`;
        code += `    for k,v in pairs(input) do\n`;
        code += `        batch:narrow(1, k , 1):copy(input[k]:view(1, 3, height, width))\n`;
        code += `    end\n`;
        code += `    return batch\n`;
        code += `end\n`;

        return code;
    }


    /**
     * Method generates Lua code that can takes a batch and breaks it apart
     * into the individual samples
     *
     * @param {EBSchema} schema The schema to generate this unwinding code for
     * @param {string} name The name of the Lua function to be generated
     */
    generateUnwindBatchCode(schema, name)
    {
        // First, ensure that the schema we are dealing with is an image
        assert(schema.isField);
        assert(schema.metadata.mainInterpretation === 'image');

        let code = '';

        code += `local ${name} = function (input)\n`;
        code += `    local samples = {}\n`;
        code += `    for k,v in pairs(input) do\n`;
        code += `        table.insert(samples, input[k]:narrow(2, k, 1))\n`;
        code += `    end\n`;
        code += `    return samples\n`;
        code += `end\n`;

        return code;
    }


    /**
     * This method should generate an input stack for this variable
     *
     * @param {EBSchema} schema The schema to generate this stack for
     * @param {EBTorchNode} inputNode The input node for this variable
     * @returns {object} An object with the following structure:
     *                      {
     *                          "outputNode": EBTorchNode || null,
     *                          "outputTensorSchema": EBTensorSchema || null,
     *                          "additionalModules": [EBCustomModule]
     *                      }
     */
    generateInputStack(schema, inputNode)
    {
        const name = schema.variableName;
        const configuration = schema.coniguration.interpretation;


        const size = EBNeuralNetworkImageComponent.getImageSizeForSchema(schema);
        let width = size.width;
        let height = size.height;

        const layer1 = new EBTorchNode(new EBTorchModule("nn.Sequential", [], [
            new EBTorchModule('nn.SpatialConvolution', [3, 32, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [32, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.SpatialConvolution', [32, 32, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [32, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.Dropout',[0.4]),
            new EBTorchModule('nn.SpatialMaxPooling',[2,2,2,2])
        ]), inputNode, `${name}_convNetLayer1`);

        // Chop width and height in half after the pooling
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);

        const layer2 = new EBTorchNode(new EBTorchModule("nn.Sequential", [], [
            new EBTorchModule('nn.SpatialConvolution', [32, 64, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [64, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.SpatialConvolution', [64, 64, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [64, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.Dropout',[0.4]),
            new EBTorchModule('nn.SpatialMaxPooling',[2,2,2,2])
        ]), layer1, `${name}_convNetLayer2`);

        // Chop width and height in half after the pooling
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);

        const layer3 = new EBTorchNode(new EBTorchModule("nn.Sequential", [], [
            new EBTorchModule('nn.SpatialConvolution', [64, 128, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [128, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.SpatialConvolution', [128, 128, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [128, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.Dropout',[0.4]),
            new EBTorchModule('nn.SpatialMaxPooling',[2,2,2,2])
        ]), layer2, `${name}_convNetLayer3`);

        // Chop width and height in half after the pooling
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);

        const layer4 = new EBTorchNode(new EBTorchModule("nn.Sequential", [], [
            new EBTorchModule('nn.SpatialConvolution', [128, 256, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [256, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.SpatialConvolution', [256, 256, 3,3,1,1,1,1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [256, 1e-3]),
            new EBTorchModule('nn.ReLU',['true']),
            new EBTorchModule('nn.Dropout',[0.4]),
            new EBTorchModule('nn.SpatialMaxPooling',[2,2,2,2])
        ]), layer3, `${name}_convNetLayer4`);

        // Chop width and height in half after the pooling
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);

        const layer5 = new EBTorchNode(new EBTorchModule("nn.Sequential", [], [
            new EBTorchModule('nn.SpatialConvolution', [256, 512, 3, 3, 1, 1, 1, 1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [512, 1e-3]),
            new EBTorchModule('nn.ReLU', ['true']),
            new EBTorchModule('nn.SpatialConvolution', [512, 512, 3, 3, 1, 1, 1, 1]),
            new EBTorchModule('nn.SpatialBatchNormalization', [512, 1e-3]),
            new EBTorchModule('nn.ReLU', ['true']),
            new EBTorchModule('nn.Dropout', [0.4]),
            new EBTorchModule('nn.SpatialMaxPooling', [2, 2, 2, 2])
        ]), layer4, `${name}_convNetLayer5`);


        // Chop width and height in half after the pooling
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);

        const outputSize = 512 * width * height;

        const reshape = new EBTorchNode(new EBTorchModule("nn.Reshape", [outputSize]), layer5, `${name}_reshape`);

        return {
            outputNode: reshape,
            outputTensorSchema: EBTensorSchema.generateDataTensorSchema(outputSize, `${name}_convNetOutput`),
            additionalModules: []
        };
    }



    /**
     * This method should generate the output stack for this variable
     *
     * @param {EBTorchNode} input The input node for this variable
     */
    generateOutputStack(input)
    {
        throw new Error("Cant create output stack for the image neural network component");
    }


    /**
     * This method returns the image size for the given schema
     *
     * @param {EBSchema} schema The schema to get the image size for
     */
    static getImageSizeForSchema(schema)
    {
        // TODO: Fix this.
        // Should be derived from the data.
        return {
            width: 100,
            height: 100
        };
    }


    /**
     * This method returns the number of channels for images from the given schema
     *
     * @param {EBSchema} schema The schema to get the channels for
     */
    static getImageChannelsForSchema(schema)
    {
        // TODO: Fix this.
        // Should be derived from the data.
        return 3;
    }
}

module.exports = EBNeuralNetworkImageComponent;