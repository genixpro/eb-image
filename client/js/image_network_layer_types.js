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

angular.module('eb').service('EBImageNetworkLayerTypes', function EBImageNetworkLayerTypes()
{
    var service = {};


    service.layerTypes = [
        {
            "id": 1,
            "title": "Convolution",
            "layerType": "convolution",
            "nInputPlane": 3,
            "nOutputPlane": 32,
            "kernelWidth": 3,
            "kernelHeight": 3,
            "stepWidth": 1,
            "stepHeight": 1,
            "paddingWidth": 1,
            "paddingHeight": 1
        }
    ]

    return service;
});