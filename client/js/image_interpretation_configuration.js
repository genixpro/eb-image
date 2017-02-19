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

/**
 * Represents a single field being configured within the neural network
 */
angular.module('eb').directive('ebImageInterpretationConfiguration', function ebImageInterpretationConfiguration($timeout, EBDataSourceService)
{
    function controller($scope, $element, $attrs)
    {
        $scope.removeConvNetLayer = function(scope)
        {
            console.log($scope.data.length);
            if ($scope.data.length > 1)
            {
                scope.remove();
            }
        };
        $scope.toggleConvNetDetails = function(scope)
        {
            scope.toggle();
        };
        $scope.moveLastToTheBeginning = function ()
        {
            // var a = $scope.data.pop();
            // $scope.data.splice(0,0, a);
        };
        $scope.newSubItem = function(scope)
        {
            var nodeData = scope.$modelValue;
            var tempData = {
                "id": nodeData.id,
                "title": nodeData.title,
                "layerType": nodeData.layerType
            };
            $scope.data.push(tempData);
        };
        $scope.collapseAll = function() {
            $scope.$broadcast('collapseAll');
        };
        $scope.expandAll = function() {
            $scope.$broadcast('expandAll');
        };
        $scope.data = [
            {
                "id": 1,
                "title": "Convolution",
                "layerType": "convolution"
            },
            {
                "id": 2,
                "title": "MAxPooling",
                "layerType": "maxpooling"
            },
            {
                "id": 3,
                "title": "ReLu",
                "layerType": "relu"
            },
            {
                "id": 4,
                "title": "BatchNormalization",
                "layerType": "batchnormalization"
            },
            {
                "id": 5,
                "title": "Dropout",
                "layerType": "dropout"
            }
        ];
    }

    return {
        templateUrl: "/plugins/image/views/image_interpretation_configuration.html",
        controller,
        restrict: "A",
        scope: {
            field: '='
        }

    };
});
