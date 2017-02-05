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
    EBFieldAnalysisAccumulatorBase = require('./../../../server/components/datasource/EBFieldAnalysisAccumulatorBase'),
    EBFieldMetadata = require('../../../shared/models/EBFieldMetadata'),
    EBInterpretationBase = require('./../../../server/components/datasource/EBInterpretationBase'),
    EBNumberHistogram = require('../../../shared/models/EBNumberHistogram'),
    jimp = require('jimp'),
    sharp = require('sharp'),
    underscore = require('underscore');

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
     * This method should return information about fields that need to be graphed on
     * the frontend for this interpretation.
     *
     * @param {*} value The value to be transformed
     * @return {Promise} A promise that resolves to an array of statistics
     */
    listStatistics(value)
    {
        return Promise.resolve([]);
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
        return jimp.read(value)
            .then(function (imageObj)
            {
                return Promise.fromCallback(function(next)
                {
                    imageObj.contain(100,100).getBuffer(jimp.MIME_JPEG, next);
                });
            });
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

            getFieldMetadata()
            {
                const metadata = new EBFieldMetadata();

                metadata.types.push('binary');

                metadata.imageWidthHistogram = EBNumberHistogram.computeHistogram(this.widths);
                metadata.imageHeightHistogram = EBNumberHistogram.computeHistogram(this.heights);

                return metadata;
            }
        })();
    }


    /**
     * This method should return a schema for the metadata associated with this interpretation
     *
     * @return {jsonschema} A schema representing the metadata for this interpretation
     */
    static metadataSchema()
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
}

module.exports = EBImageInterpretation;