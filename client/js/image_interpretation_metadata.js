
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
 * This directive provides the menu for viewing metadata about the field
 */
angular.module('eb').directive('ebImageInterpretationMetadata', function ebImageInterpretationMetadata($timeout)
{
    function controller($scope, $element, $attrs)
    {

    }

    return {
        templateUrl: "/plugins/image/views/image_interpretation_metadata.html",
        controller,
        restrict: "A",
        scope: {
            field: '='
        }
    };
});
