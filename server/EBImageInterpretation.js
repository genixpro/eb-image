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
    fileType = require('file-type'),
    fs = require('fs'),
    EBFieldAnalysisAccumulatorBase = require('./../../../server/components/datasource/EBFieldAnalysisAccumulatorBase'),
    EBFieldMetadata = require('../../../shared/models/EBFieldMetadata'),
    EBInterpretationBase = require('./../../../server/components/datasource/EBInterpretationBase'),
    EBNumberHistogram = require('../../../shared/models/EBNumberHistogram'),
    jimp = require('jimp'),
    sharp = require('sharp'),
    Promise = require('bluebird'),
    underscore = require('underscore');

let imageN = 0;

/**
 * The image interpretation applies when you have binary data containing some sort of image
 */
class EBImageInterpretation extends EBInterpretationBase
{
    /**
     * Constructor
     */
    constructor()
    {
        super('image');
    }


    /**
     * This method should return the list of interpretations that this interpretation is dependent
     * on. This interpretation won't be checked unless the dependent interpretation is the next
     * one up in the chain.
     *
     * If the list of dependencies is empty, then this interpretation is assumed to be able to
     * operate directly on raw-values from JSON.
     *
     * @return [String] An array of strings with the type-names for each of the higher interpretations
     */
    getUpstreamInterpretations()
    {
        return ['binary'];
    }




    /**
     * This method returns the raw javascript type of value that this interpretation applies to.
     *
     * @return {string} Can be one of: 'object', 'array', 'number', 'string', 'boolean', 'binary'
     */
    getJavascriptType()
    {
        return 'binary';
    }



    /**
     * This method should look at the given value and decide whether it can be handled by this
     * interpretation.
     *
     * @param {*} value Can be practically anything.
     * @return {Promise} A promise that resolves to either true or false on whether that value
     *                   can be handled by that interpretation.
     */
    checkValue(value)
    {
        if (value instanceof Buffer)
        {
            // Check if the data has some sort of image mime-type
            const type = fileType(value);
            if (type.mime.indexOf('image') === 0)
            {
                return Promise.resolve(true);
            }
        }

        return Promise.resolve(false);
    }



    /**
     * This method should transform a given schema for a value following this interpretation.
     * It should return a new schema for the interpreted version.
     *
     * @param {EBSchema} schema The schema for a field that wants to be interpreted by this interpretation.
     * @return {Promise} A promise that resolves to a new EBSchema object.
     */
    transformSchema(schema)
    {
        return Promise.resolve(schema);
    }




    /**
     * This method should transform a given value, assuming its following this interpretation.
     *
     * @param {*} value The value to be transformed
     * @return {Promise} A promise that resolves to a new value.
     */
    transformValue(value)
    {
        return Promise.resolve(value);
    }



    /**
     * This method should transform an example into a value that is small enough to be
     * stored with the schema and shown on the frontend. Information can be destroyed
     * in this transformation in order to allow the data to be stored easily.
     *
     * @param {*} value The value to be transformed
     * @return {Promise} A promise that resolves to a new object that is similar to the old one to a human, but with size truncated for easy storage.
     */
    transformExample(value)
    {
        return jimp.read(value).then((imageObj) =>
        {
            return Promise.fromCallback((next) =>
            {
                imageObj.contain(100,100).getBuffer(jimp.MIME_JPEG, next);
            }).then((buffer) =>
            {
                return buffer.toString('base64');
            });
        });
    }


    /**
     * This method should transform the given schema for input to the neural network.
     *
     * @param {EBSchema} schema The schema to be transformed
     * @return {Promise} A promise that resolves to a new value.
     */
    transformSchemaForNeuralNetwork(schema)
    {
        return schema;
    }


    /**
     * This method should prepare a given value for input into the neural network
     *
     * @param {*} value The value to be transformed
     * @param {EBSchema} schema The schema for the value to be transformed
     * @return {Promise} A promise that resolves to a new value.
     */
    transformValueForNeuralNetwork(value, schema)
    {
        const configuration = schema.configuration.interpretation;
        return jimp.read(value).then((imageObj) =>
        {
            let mirrorHorizontal = false;
            if (configuration.mirrorHorizontally)
            {
                if (Math.random() > 0.5)
                {
                    mirrorHorizontal = true;
                }
            }
            if (mirrorHorizontal)
            {
                imageObj.flip(true, false);
            }

            if (configuration.stretchImage)
            {
                // Determine the amount of horizontal scaling
                const scalingFactor = 0.3;
                const widthDiff = ((Math.random() * scalingFactor * 2) - scalingFactor) * 100;
                const heightDiff = ((Math.random() * scalingFactor * 2) - scalingFactor) * 100;

                const newWidth = 100 + widthDiff;
                const newHeight = 100 + heightDiff;

                const cropPointX = Math.max(0, Math.floor(Math.random() * widthDiff));
                const cropPointY = Math.max(0, Math.floor(Math.random() * heightDiff));

                imageObj.resize(newWidth, newHeight);
                imageObj.crop(cropPointX, cropPointY, Math.min(newWidth, 100), Math.min(newHeight, 100) );
                imageObj.contain(100, 100);
            }

            if (configuration.rotateImage)
            {
                const rotationFactor = 15;
                const degrees = (Math.random() * rotationFactor * 2) - rotationFactor;
                imageObj.rotate(degrees);
                imageObj.resize(100, 100);
            }

            return Promise.fromCallback((next) =>
            {
                imageObj.getBuffer(jimp.MIME_JPEG, next);
            }).then((buffer) =>
            {
                return buffer.toString('base64');
            });
        });
    }


    /**
     * This method should take output from the neural network and transform it back
     *
     * @param {*} value The value to be transformed
     * @param {EBSchema} schema The schema for the value to be transformed
     * @return {Promise} A promise that resolves to a new value
     */
    transformValueBackFromNeuralNetwork(value, schema)
    {
        return new Buffer(value, 'base64');
    }


    /**
     * This method should generate the default configuration for the given schema
     *
     * @param {EBSchema} schema The schema for the value to be transformed
     * @return {object} An object which follows the schema returned from configurationSchema
     */
    generateDefaultConfiguration(schema)
    {
        return {
            rotateImage: true,
            stretchImage: true,
            mirrorHorizontally: true
        };
    }


    /**
     * This method should create a new field accumulator, a subclass of EBFieldAnalysisAccumulatorBase.
     *
     * This accumulator can be used to analyze a bunch of values through the lens of this interpretation,
     * and calculate statistics that the user may use to analyze the situation.
     *
     * @return {EBFieldAnalysisAccumulatorBase} An instantiation of a field accumulator.
     */
    createFieldAccumulator()
    {
        // Create a subclass and immediately instantiate it.
        return new (class extends EBFieldAnalysisAccumulatorBase
        {
            constructor()
            {
                super();
                this.widths = [];
                this.heights = [];
            }

            accumulateValue(value)
            {
                const image = sharp(value);
                return image.metadata().then((metadata) =>
                {
                    // Get the width and height of the image
                    this.widths.push(metadata.width);
                    this.heights.push(metadata.height);
                });
            }

            getFieldStatistics()
            {
                return {
                    imageWidthHistogram: EBNumberHistogram.computeHistogram(this.widths),
                    imageHeightHistogram: EBNumberHistogram.computeHistogram(this.heights)
                };
            }
        })();
    }


    /**
     * This method should return a schema for the metadata associated with this interpretation
     *
     * @return {jsonschema} A schema representing the metadata for this interpretation
     */
    static statisticsSchema()
    {
        return {
            "id": "EBFieldMetadata",
            "type": "object",
            "properties": {
                imageWidthHistogram: EBNumberHistogram.schema(),
                imageHeightHistogram: EBNumberHistogram.schema()
            }
        };
    }


    /**
     * This method should return a schema for the configuration for this interpretation
     *
     * @return {jsonschema} A schema representing the configuration for this interpretation
     */
    static configurationSchema()
    {
        return {
            "id": "EBImageInterpretation.configurationSchema",
            "type": "object",
            "properties": {
                rotateImage: {"type": "boolean"},
                stretchImage: {"type": "boolean"},
                mirrorHorizontally: {"type": "boolean"}
            }
        };
    }


    /**
     * This method should return a schema for accumulating accuracy results from values in this interpretation
     *
     * @return {jsonschema} A schema representing whatever is needed to store results
     */
    static resultsSchema()
    {
        return {
            "id": "EBImageInterpretation.resultsSchema",
            "type": "object",
            "properties": {}
        };
    }
}

module.exports = EBImageInterpretation;