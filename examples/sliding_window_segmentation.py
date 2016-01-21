#!/usr/bin/env python
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Copyright (C) 2015 National Institutes of Health

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Written by:  Christopher Coletta (github.com/colettace)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

import numpy as np

from wndcharm import __version__ as wndcharm_version
print "WND-CHARM version " + wndcharm_version
from wndcharm.FeatureSpace import FeatureSpace
from wndcharm.FeatureVector import SlidingWindow
from wndcharm.FeatureWeights import FisherFeatureWeights
from wndcharm.FeatureSpacePrediction import FeatureSpaceClassification

import argparse

parser = argparse.ArgumentParser( description='Use a classifier to find regions within images ("segmentation"). Slides a window of user-defined size over the image and performs a classification at each window location using the user-provided classifier. Outputs a grayscale "heatmap" image for each class in the classifier where pixel intensity corresponds with liklihood of that class being in the given location.' )

# positional arguments (indicated to argparse by excluding leading dash in arg name)
parser.add_argument( 'input_image', help='Path to input tiff image to be scanned.', nargs=1,
                     type=str, metavar='<image path>')
parser.add_argument( 'classifier', help='Path to WND-CHARM feature space file (.fit/.fof), or path to top-level directory containing subdirectories of training images where each subdirectory is a class.',
                     nargs=1, type=str, metavar='<classifier path>' )

# "Optional" arguments (N.B. leading dash(es) in arg names below), made required by setting "required=True".
parser.add_argument( '--outdir', help='Destination directory for resulting heatmap images (default is directory containing input image)',
                     nargs='?', default=None )
parser.add_argument( '-f', help='Classifier tuning parameter. Fraction of top-ranked features in feature space to use for classification, on interval (0.0,1.0] (defaults to using top %%15)',
                     type=float, metavar='<float>',default=0.15)
parser.add_argument( '-F', help='An alternate way of specifying -f above, instead using an integer number of features',
                     type=int, metavar='<integer>', default=None )
parser.add_argument( '--n_jobs', help='Number of cores to use to calculate image features on input image ROIs (default = all of them)',
                     metavar='<integer>', default=None)
parser.add_argument( '--classifier_args', nargs='*', help='Classifier sampling args directly passed to FeatureSpace.NewFrom* factory methods (see respective docstrings for more info).' )
parser.add_argument( '--window_args', nargs='*', help='Args directly passed to SlidingWindow constructor (see docstring for more info).' )
parser.add_argument('--verbose', '-v', help='Give more output, useful for debugging.', action='store_true' )

args = parser.parse_args()
print args

quiet = not args.verbose

def parse_args( inargs ):
    outargs = []
    outkwargs = {}
    for _arg in inargs:
        if '=' in _arg:
            key, val = _arg.split('=')
            outkwargs[ key ] = eval( val )
        else:
            outargs.append( _arg )
    return outargs, outkwargs

cl_args, cl_kwargs = parse_args( args.classifier_args )
# positional args come in as a list from argparse
classifier_path = args.classifier[0]
input_image = args.input_image[0]

# Step 1: Load classifier features and (or generate if they don't exist)
if classifier_path.endswith( '.fit' ):
    classifier = FeatureSpace.NewFromFitFile( classifier_path, *cl_args, quiet=quiet, **cl_kwargs )
elif classifier.endswith( ('.fof', '.fof.tsv', '.tsv' ) ):
    classifier = FeatureSpace.NewFromFileOfFiles( classifier_path, *cl_args, quiet=quiet, **cl_kwargs )
else:
    classifier = FeatureSpace.NewFromDirectory( classifier_path, *cl_args, quiet=quiet, **cl_kwargs )

if isinstance( args.F, int ):
    n_features = args.F
elif isinstance( args.f, float ):
    n_features = int( round( args.f * classifier.num_features ) )
else:
    # All of them
    n_features =  classifier.num_features

classifier.Normalize( inplace=True, quiet=quiet )
weights = FisherFeatureWeights.NewFromFeatureSpace( classifier ).Threshold( n_features )
classifier.FeatureReduce( weights, inplace=True, quiet=quiet )

w_args, w_kwargs = parse_args( args.window_args )

# Sliding window constructor doesn't take non-keyword args as of right now
window = SlidingWindow( *w_args, source_filepath=input_image, **w_kwargs )
print '\n', window, "Number of samples = " + str( window.num_positions ), '\n'

image_features = FeatureSpace.NewFromSlidingWindow( window, n_jobs=args.n_jobs, quiet=quiet )
#image_features.ToFitFile( sort=False )
image_features.FeatureReduce( weights, inplace=True, quiet=quiet )
image_features.Normalize( classifier, inplace=True, quiet=quiet )

exp = FeatureSpaceClassification.NewWND5( classifier, image_features, weights, quiet=quiet )

# Create a list of zero'd out, image-sized, 2-D byte numpys
img_shape = ( window.preprocessed_full_px_plane_width, window.preprocessed_full_px_plane_height )
masks = [] 

for i in xrange( classifier.num_classes ):
    masks.append( np.full( img_shape, np.nan ) )

