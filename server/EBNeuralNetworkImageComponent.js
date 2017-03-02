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
        const Nconfiguration = schema.configuration.interpretation;

        const size = EBNeuralNetworkImageComponent.getImageSizeForSchema(schema);
        let width = size.width;
        let height = size.height;
        let lastConvLayerKernelSize = null;
        const torchModules = [];
        Nconfiguration.layers.forEach(function(entry) {
            if (entry.layerType === 'convolution')
            {
                const convModule = new EBTorchModule('nn.SpatialConvolution', [entry.nInputPlane, entry.nOutputPlane, entry.kernelWidth,entry.kernelHeight,entry.stepWidth,entry.stepHeight,entry.paddingWidth,entry.paddingHeight]);
                // console.log(convModule);
                torchModules.push(convModule);
                lastConvLayerKernelSize = entry.nOutputPlane;
            }
            else if (entry.layerType === 'batchnormalization')
            {
                const batchNormal = new EBTorchModule('nn.SpatialBatchNormalization', [entry.nInputFeatures]);
                torchModules.push(batchNormal);
            }
            else if (entry.layerType === 'relu')
            {
                const reluModule = new EBTorchModule('nn.ReLU', [entry.nState]);
                torchModules.push(reluModule);
            }
            else if (entry.layerType === 'dropout')
            {
                const dropoutModule = new EBTorchModule('nn.Dropout', [entry.nRatio]);
                torchModules.push(dropoutModule);
            }
            else if (entry.layerType === 'maxpooling')
            {
                const maxpoolModule = new EBTorchModule('nn.SpatialMaxPooling', [entry.nKernelWidth, entry.nKernelHeight, entry.nStepWidth,entry.nStepHeight]);
                torchModules.push(maxpoolModule);

                // Chop width and height in half after the pooling
                width = Math.floor(width / 2);
                height = Math.floor(height / 2);
            }
        });

        const convStack = new EBTorchNode(new EBTorchModule("nn.Sequential", [], torchModules), inputNode, `${name}_convStack`);

        const outputSize = lastConvLayerKernelSize * width * height;

        const reshape = new EBTorchNode(new EBTorchModule("nn.Reshape", [outputSize]), convStack, `${name}_reshape`);

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