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


import wndcharm
import numpy as np
from . import feature_vector_major_version
from . import feature_vector_minor_version_from_num_features
from .utils import normalize_by_columns

class WrongFeatureSetVersionError( Exception ):
    pass

class IncompleteFeatureSetError( Exception ):
    pass

from . import feature_vector_minor_version_from_num_features
ver_to_num_feats_map = dict((v, k) for k, v in feature_vector_minor_version_from_num_features.iteritems())
# Couldn't get this "Python singleton inherited from swig-wrapped C++ object" to work:
#*** NotImplementedError: Wrong number or type of arguments for overloaded function 'FeatureComputationPlan_add'.
#  Possible C/C++ prototypes are:
#    FeatureComputationPlan::add(std::string const &)
#    FeatureComputationPlan::add(FeatureGroup const *)
# "self" below was of type "<wndcharm.FeatureVector.PyFeatureComputationPlan;  >"
# when what was required was a SWIG proxy object to translate native python strings
# into std::string
# "<wndcharm.wndcharm.FeatureComputationPlan; proxy of <Swig Object of type 'FeatureComputationPlan *' at 0x111263cf0> >"
##================================================================
#class PyFeatureComputationPlan( wndcharm.FeatureComputationPlan ):
#    """Contains a cache to save memory, as there may be tens of thousands of samples
#    and therefore the same number of of redundant instances of the same computation plan."""
#
#    plan_cache = {}
#
#    def __new__( cls, feature_list, name='custom' ):
#        """Takes list of feature strings and chops off bin number at the first
#        space on right, e.g., "feature alg (transform()) [bin]" """
#
#        feature_groups = frozenset( [ feat.rsplit(" ",1)[0] for feat in feature_list ] )
#
#        if feature_groups in cls.plan_cache:
#            return cls.plan_cache[ feature_groups ]
#
#        self = super( PyFeatureComputationPlan, cls ).__new__( cls, name )
#        [ self.add( family ) for family in feature_groups ]
#
#        cls.plan_cache[ feature_groups ] = self
#        return self

# instead implement with a global dict to serve as feature plan cache

plan_cache = {}

def GenerateFeatureComputationPlan( feature_list, name='custom' ):
    """Takes list of feature strings and chops off bin number at the first
    space on right, e.g., "feature alg (transform()) [bin]" """

    global plan_cache
    feature_groups = frozenset( [ feat.rsplit(" ",1)[0] for feat in feature_list ] )

    if feature_groups in plan_cache:
        return plan_cache[ feature_groups ]

    obj = wndcharm.FeatureComputationPlan( name )
    for family in feature_groups:
        obj.add( family )

    plan_cache[ feature_groups ] = obj
    return obj


#############################################################################
# class definition of FeatureVector
#############################################################################
class FeatureVector( object ):
    """
    FeatureVector is the container object for image features from a
    single sample, as well as for the 5D image sampling parameters and feature
    metadata that was used to produce the features. Info from FeatureVectors
    are stacked horizontally and vertically to form FeatureSpaces. The general
    workflow is to set sampling and feature attributes and call
    self.GenerateFeatures() to either load corresponding features from disk
    or obtain the pixel plane called out and calculate features outright.

    Instance Attributes:
    ============================

    General Attributes:
    -------------------

        self.name - str - Sample (row) name, common across all channels
        self.source_filepath - str
            Filesystem path or other handle, e.g. OMERO obj id
        self.original_px_plane - wndcharm.ImageMatrix
            Cached pristine original image as loaded from disk.
        self.original_px_plane_width - int
        self.original_px_plane_height - int
            Dimensions of orig img kept if cache is cleared, useful for deriving
            tiling, cropping params.
        self.preprocessed_full_px_plane - wndcharm.ImageMatrix
            Full image with downsample/mean/std shift done & WITHOUT any cropping/tiling
        self.preprocessed_local_px_plane - wndcharm.ImageMatrix
            Immediate substrate/local pixel plane upon which features will be calculated.
        self.auxiliary_feature_storage - str - Path to storage on file system,
            currently an ASCII text representation of data and metadata, written
            in a format used by the classic C++ WND-CHARM implementation. Has .sig
            file extension, and contains 5D sampling options as part of the filename.
            In future, could be a path to a hdf5 or sql file.
        self.basename - str - Part of filename, a substring of self.auxiliary_feature_storage,
            with all the sampling options and .sig file extension stripped out.
        self.ground_truth_label - str - stringified ground truth/category
        self.ground_truth_value - float - numeric representation of ground truth, if any.
        self.fs_col - int - "FeatureSpace column index". FeatureVectors (and FeatureSpaces)
            can be horizontally stacked to create FeatureSpaces with higher
            dimensionality. Taking each FeatureVector to be one indivisible stackable
            unit, this integer represents the index of this unit within that horizontal
            stack, i.e. the feature set column to which this FeatureVector belongs.

    Feature Metadata Attributes (iterables of length N features, unless noted):
    -------------------------------------------------

        self.feature_names - list - machine-parsable, human readable strings
            specifying the feature extraction algorithm, pixel plane transform
            chain, channel, and intra-feature family index.
        self.values - numpy.ndarray containing features, order corresponds with
            feature_names. Can be raw extracted values, or values normalized to be on
            interval [0,100] relative to a FeatureSpace.
        self.num_features - int - should always be == len( self.feature_names ) == len( self.values )
        self.feature_maxima - list - if values are normalized, can uses these to back out
            original, unnormalized features.
        self.feature_minima - list - if values are normalized, can uses these to back out
            original, unnormalized features.
        self.normalized_against - wndcharm.FeatureSpace - if not None, is the source
            FeatureSpace against which features contained herein are normalized.

    Feature Set Metadata Attributes (scalars):
    ---------------------

        The following attributes are specific to the WND-CHARM feature bank,
        but other feature sets can reuse these, or you can add other member attributes
        specific to that feature set:

        self.feature_set_version - str - in the form "X.Y", where X is the major version
            and Y is the minor version.
        self.color - bool - Extract additional features specific to RGB images.
        self.long - bool - If evals to True, extract WND-CHARM "long" features set =
            2919 features; if False calculate 1059 features.
        self.feature_computation_plan - instance of wndcharm.FeatureComputationPlan.

    Sampling Options Attributes:
    ----------------------------

        self.channel - string - channel name
        self.time_index - whatever you want
        self.sample_group_id - int - sample group id, identifies groups of samples
            which is indivisible across train/test splits
        self.sample_sequence_id - int, index within sample group. If no
            ROI/image subsample, whole image is tile 1 of 1 in this sample group.
        self.downsample  - int, in percents
        self.pixel_intensity_mean - int - Mean shift pixel intensities. New pixel
            intensities are clipped to interval [0 , (bit depth)^2-1], THEREFORE
            RESULTING MEAN MAY BE DIFFERENT THAN WHAT'S ASKED FOR BY USER!
        self.pixel_intensity_stdev - int
        self.rot - bool - Calculate entire feature set for each pixel plane rotated
            4 times, 90 degrees each time.

        The following attributes are for calling out a single ROI within an image for
        feature extraction:
        self.x - int - xi
        self.w - int - delta x
        self.y - int - yi
        self.h - int - delta y
        self.z - int - zi
        self.z_delta - int - delta z
        self.roi - str - string pulled out of the file name sampling options in the
            form of "-B<x>_<y>_<w>_<h>". Strong target for deprecation!

        The following attributes are for setting up a tiling scheme for extracting
        features from 2 or more repeating subsampled regions:
        self.tiling_scheme - str - string pulled out of the file name sampling options
            in the form of "-t<num_cols>_<num_rows>_<col_index>_<row_index>". Strong
            target for deprecation!
        self.tile_num_cols - int - default is 1
        self.tile_num_rows - int - default is 1
        self.tile_row_index - int, indices count from 0
        self.tile_col_index - int, indices count from 0
"""


    import re
    sig_filename_parser = re.compile( ''.join( [
    # basename:
    r'(?P<basename>.+?)',
    # ROI:
    r'(?P<roi>-B(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)_(?P<h>\d+))?',
    # downsampling:
    r'(?:-d(?P<downsample>\d+))?',
    # pixel intensity adjustment:
    r'(?:-S(?P<pixel_intensity_mean>\d+)(?:_(?P<pixel_intensity_stdev>\d+))?)?',
    # rotations:
    r'(?:-R_(?P<rot>\d))?',
    # tiling info: (I know the order is a bit wacky: num_cols, num_rows, col index, row_index
    r'(?P<tiling_scheme>-t(?P<tile_num_cols>\d+)(?:x(?P<tile_num_rows>\d+))?_(?P<tile_col_index>\d+)_(?P<tile_row_index>\d+))?',
    # color features:
    r'(?P<color>-c)?',
    # long feature set:
    r'(?P<long>-l)?',
    # extension: .sig or .pysig
    r'\.(?:py)?sig$' ] ) )

    members_of_type_int = [ 'tile_row_index', 'tile_col_index', 'tile_num_rows', 'tile_num_cols', 'sample_group_id', 'sample_sequence_id', 'x','y','w','h','z','z_delta', 'downsample', 'pixel_intensity_mean', 'pixel_intensity_stdev',]

    #==============================================================
    def __init__( self, **kwargs ):
        #: Row name, common across all channels
        self.name = None
        #: str - Filesystem path or other handle, e.g. OMERO obj id
        self.source_filepath = None
        #: wndcharm.ImageMatrix - Cached pristine original image as loaded from disk.
        self.original_px_plane = None
        #: Keep separate track of raw px plane height width apart from width and height
        #: members of wndcharm.ImageMatrix object in case the latter object isn't cached.
        self.original_px_plane_width = None
        self.original_px_plane_height = None
        #: wndcharm.ImageMatrix - full image with downsample/mean/std shift done.
        self.preprocessed_full_px_plane = None
        self.preprocessed_full_px_plane_width = None
        self.preprocessed_full_px_plane_height = None
        #: wndcharm.ImageMatrix - the immediate substrate/local pixel plane
        #: upon which features will be calculated.
        self.preprocessed_local_px_plane = None
        #: str - Path to .sig file, in future hdf/sql file
        self.auxiliary_feature_storage = None
        #: str - prefix string to which sampling options will be appended to form .sig filepath
        self.basename = None
        #: ground_truth_label is stringified ground truth
        self.ground_truth_label = None
        self.ground_truth_value = None

        #: feature names
        self.feature_names = None
        #: Used for loading features from disk but not all requested were there
        self.temp_names = None
        #: feature vector
        self.values = None
        #: Used for loading features from disk but not all requested were there
        self.temp_values = None

        #: feature_maxima, feature_minima, and normalized_against are members
        #: this object shares with the FeatureSpaceObject, as well as implementations
        #: of FeatureReduce() and Normalize(), and CompatibleFeatureSetVersion()
        self.feature_maxima = None
        self.feature_minima = None
        self.feature_means = None
        self.feature_stdevs = None
        self.normalized_against = None

        self.sample_group_id = None
        self.channel = None
        self.time_index = None
        self.tiling_scheme = None
        #: If no ROI image subsample, whole image is tile 1 of 1 in this sample group.
        self.tile_num_rows = None
        self.tile_num_cols = None
        #: indices count from 0
        self.tile_row_index = None
        self.tile_col_index = None
        self.sample_sequence_id = None
        #: downsample (in percents)
        self.downsample = 0
        self.pixel_intensity_mean = None
        self.pixel_intensity_stdev = None
        self.roi = None
        self.h = None
        self.w = None
        self.x = None
        self.y = None
        self.z = None
        self.z_delta = None
        self.rot = None
        self.fs_col = 0

        #: self.num_features should always be == len( self.feature_names ) == len( self.values )
        self.num_features = None

        # WND-CHARM feature bank-specific params:
        self.color = None
        self.long = None
        self.feature_set_version = None
        self.feature_computation_plan = None

        self.Update( **kwargs )

    #==============================================================
    def __len__( self ):
        try:
            length = len( self.values )
        except:
            length = 0
        return length

    #==============================================================
    def __str__( self ):
        outstr = '<' + self.__class__.__name__
        if self.name is not None:
            if len(self.name) > 30:
                name = '...' + self.name[-30:]
            else:
                name = self.name
            outstr += ' "' + name + '"'
        if self.x is not None and self.y is not None and self.w is not None and self.h is not None:
            outstr += ' ROI={0}x{1}+{2}+{3}"'.format( self.w, self.h, self.x, self.y)
        if self.ground_truth_label is not None:
            outstr += ' label="' + self.ground_truth_label + '"'
        if self.values is not None:
            outstr += ' n_features=' + str( len( self ) )
        if self.sample_group_id is not None:
            outstr += ' grp=' + str( self.sample_group_id )
        if self.sample_sequence_id is not None:
            outstr += ' seq=' + str( self.sample_sequence_id )
        if self.fs_col is not None:
            outstr += ' fs_col=' + str( self.fs_col )
        return outstr + '>'

    #==============================================================
    def __repr__( self ):
        return str(self)

    #==============================================================
    def Update( self, force=False, **kwargs ):
        """force - if True then kwargs coming in with None values will overwrite
        members in self, even if those members are non-None before calling Update."""

        self_namespace = vars( self )
        for key, val in kwargs.iteritems():
            # don't bother setting a val if it's None unless force
            if val is not None or force:  
                if key in self_namespace:
                    #if self_namespace[ key ] != None and self_namespace[ key ] != val:
                    #    from warnings import warn
                    #    warn( "Overwriting attrib {0} old val {1} new val {2}".format( key, self_namespace[ key ], val ) )
                    self_namespace[ key ] = val
                else:
                    raise AttributeError( 'No instance variable named "{0}" in class {1}'.format(
                        key, self.__class__.__name__ ) )

        def ReturnNumFeaturesBasedOnMinorFeatureVectorVersion( minor_fvv ):
            if major == 1:
                num_feats_dict = feature_vector_minor_version_from_num_features_v1
            else:
                num_feats_dict = feature_vector_minor_version_from_num_features

            for num_feats, version in num_feats_dict.iteritems():
                if version == minor_fvv:
                    return num_feats
            return None

        # FIXME: feature_set_version refers to WND-CHARM specific feature set specifications.
        # Want to be able to handle other feature sets from other labs in the future.
        major = feature_vector_major_version # global

        # The feature_set_version helps describe what features contained in the set.
        # Major version has to do with fixing bugs in the WND_CHARM algorithm code base.
        # The minor version describes the composition of features in the feature set.
        # Minor versions 1-4 have specific combination of WND-CHARM features.
        # Minor version 0 refers to user-defined combination of features.

        # Check to see if there is a user-defined set of features for this feature vector:
        if self.feature_computation_plan or self.feature_names:
            # set num_features
            if self.feature_names:
                self.num_features = len( self.feature_names )
            else:
                self.num_features = self.feature_computation_plan.n_features

            if self.num_features not in feature_vector_minor_version_from_num_features:
                minor = 0
            else:
                # FIXME: If features are out of order, should have a minor version of 0
                minor = feature_vector_minor_version_from_num_features[ len( self.feature_names ) ]
        else:
            if not self.long:
                if not self.color:
                    minor = 1
                else:
                    minor = 3
            else:
                if not self.color:
                    minor = 2
                else:
                    minor = 4
            self.num_features = ReturnNumFeaturesBasedOnMinorFeatureVectorVersion( minor )
        self.feature_set_version = '{0}.{1}'.format( major, minor )

        # When reading in sampling opts from the path, they get pulled out as strings
        # instead of ints:
        for member in self.members_of_type_int:
            val = getattr( self, member )
            if val is not None and not isinstance( val, int ):
                setattr( self, member, int( val ) )

        # sequence order has historically been (e.g. 3x3):
        # 0 3 6
        # 1 4 7
        # 2 5 8
        if self.sample_sequence_id is None and self.tile_col_index is not None \
                and self.tile_num_rows is not None and self.tile_row_index is not None:
            self.sample_sequence_id = \
                    (self.tile_col_index * self.tile_num_rows ) + self.tile_row_index
        return self

    #==============================================================
    def Derive( self, base=None, **kwargs ):
        """Make a copy of this FeatureVector, except members passed as kwargs"""

        from copy import deepcopy
        if base:
            new_obj = self.__class__.__bases__[0]()
        else:
            new_obj = self.__class__()
        self_namespace = vars( self )
        new_obj_namespace = vars( new_obj )

        # skip these (if any):
        convenience_view_members = []

        # Are all keys in kwargs valid instance attribute names?
        invalid_kwargs = set( kwargs.keys() ) - set( self_namespace.keys() )
        if len( invalid_kwargs ) > 0:
            raise ValueError( "Invalid keyword arg(s) to Derive: {0}".format( invalid_kwargs ) )

        # Go through all of self's members and copy them to new_fs
        # unless a key-val pair was passed in as kwargs
        for key in self_namespace:
            if key in convenience_view_members:
                continue
            if key in kwargs:
                new_obj_namespace[key] = kwargs[key]
            else:
                new_obj_namespace[key] = deepcopy( self_namespace[key] )
        return new_obj

    #==============================================================
    def __deepcopy__( self, memo ):
        """Make a deepcopy of this FeatureVector
        memo - arg required by deepcopy package"""
        return self.Derive()

    #==============================================================
    def GenerateSigFilepath( self ):
        """The C implementation of wndchrm placed feature metadata
        in the filename in a specific order, recreated here."""

        from os.path import splitext

        # FIXME: sigpaths for FeatureVectors with different channels
        # may have sig file names that will collide/overwrite each other.
        if self.basename:
            base = self.basename
        elif isinstance( self.source_filepath, wndcharm.ImageMatrix ) and \
                self.source_filepath.source:
            base, ext = splitext( self.source_filepath.source )
            self.basename = base
        elif isinstance( self.source_filepath, str ) and self.source_filepath:
            base, ext = splitext( self.source_filepath )
            self.basename = base
        elif self.name:
            # ext may be nothing, that's ok
            base, ext = splitext( self.name )
            self.basename = base
        else:
            raise ValueError( 'Need for "basename" or "source_filepath" or "name" attribute in FeatureVector object to be set to generate sig filepath.')

        if self.x is not None and self.y is not None and self.w is not None and self.h is not None:
            base += "-B{0}_{1}_{2}_{3}".format( self.x, self.y, self.w, self.h )
        if self.downsample:
            base += "-d" + str(self.downsample)
        if self.pixel_intensity_mean is not None:
            base += "-S" + str(self.pixel_intensity_mean)
            if self.pixel_intensity_stdev is not None:
                base += "-S" + str(self.pixel_intensity_stdev)
        if self.rot is not None:
            base += "-R_" + str(self.rot)
        # the historical tile notation order is: num_cols, num_rows, col index, row_index
        if self.tile_num_cols and self.tile_num_cols != 1:
            base += "-t" + str(self.tile_num_cols)
            if self.tile_num_rows and self.tile_num_rows != 1 \
                    and self.tile_num_rows != self.tile_num_cols:
                # 6x6 tiling scheme is simply represented as -t6
                base +="x" + str(self.tile_num_rows)
            if self.tile_col_index is not None and self.tile_row_index is not None:
                base += "_{0}_{1}".format( self.tile_col_index, self.tile_row_index )
            else:
                raise ValueError('Need to specify tile_row_index and tile_col_index in self for tiling params')
        if self.color:
            base += '-c'
        if self.long:
            base += '-l'

        return base + '.sig'

    #================================================================
    def GetOriginalPixelPlane( self, cache=False ):
        """Gets cached image pixels exactly as loaded from disk, i.e.,
        without cropping/std shift/downsample etc."""

        if self.original_px_plane is not None:
            return self.original_px_plane

        if self.source_filepath is not None:
            # Load from disk
            from .PyImageMatrix import PyImageMatrix
            original_px_plane = PyImageMatrix()
            retval = original_px_plane.OpenImage( self.source_filepath )
            if 1 != retval:
                errmsg = 'Could not build a wndchrm.PyImageMatrix from {0}, check the path.'
                raise ValueError( errmsg.format( self.source_filepath ) )
        else:
            errmsg = "Could not load pixel plane required for calculating features: " + \
                    'Neither members "source_filepath" or "raw_px_plane" are set.'
            raise ValueError( errmsg )

        self.original_px_plane_width = original_px_plane.width
        self.original_px_plane_height = original_px_plane.height

        if cache:
            self.original_px_plane = original_px_plane

        return original_px_plane

    #================================================================
    def GetPreprocessedFullPixelPlane( self, cache=False ):
        """Gets cached pre-cropped pixel plane INCLUDING std shift/downsample, etc."""

        if self.preprocessed_full_px_plane is not None:
            return self.preprocessed_full_px_plane

        original_px_plane = self.GetOriginalPixelPlane(cache=cache)

        if not self.downsample and not self.pixel_intensity_mean:
            self.preprocessed_full_px_plane_width = original_px_plane.width
            self.preprocessed_full_px_plane_height = original_px_plane.height

            if cache:
                self.preprocessed_full_px_plane = original_px_plane
            return original_px_plane

        # Do a copy to avoid possibly corrupting source FULL_px_plane
        from .PyImageMatrix import PyImageMatrix
        preprocessed_full_px_plane = PyImageMatrix()

        if self.downsample:
            d = float( self.downsample ) / 100
            preprocessed_full_px_plane.Downsample( original_px_plane, d, d )
        else:
            preprocessed_full_px_plane.copy( original_px_plane )

        if self.pixel_intensity_mean:
            mean = float( self.pixel_intensity_mean )
            std = float( self.pixel_intensity_stdev ) if self.pixel_intensity_stdev else 0.0
            # void normalize(double min, double max, long range, double mean, double stdev);
            preprocessed_full_px_plane.normalize( -1, -1, -1, mean, std )

        self.preprocessed_full_px_plane_width = preprocessed_full_px_plane.width
        self.preprocessed_full_px_plane_height = preprocessed_full_px_plane.height

        if cache:
            self.preprocessed_full_px_plane = preprocessed_full_px_plane
        return preprocessed_full_px_plane

    #================================================================
    def GetPreprocessedLocalPixelPlane( self, cache=False ):
        """obtains a pixel plane in accordance with 5D sampling options as prescribed
        by instance attributes."""

        # ImageMatrix::submatrix() has a funky signature:
        #   void ImageMatrix::submatrix (const ImageMatrix &matrix, const unsigned int x1, const unsigned int y1, const unsigned int x2, const unsigned int y2);
        #   where x2 and y2 are INCLUSIVE, i.e., must subtract 1 from both

        if self.preprocessed_local_px_plane is not None:
            return self.preprocessed_local_px_plane

        from .PyImageMatrix import PyImageMatrix

        preprocessed_full_px_plane = self.GetPreprocessedFullPixelPlane( cache=cache )

        # submatrix for ROI/tiling
        if self.x is not None and self.y is not None and self.w is not None and self.h is not None:
            # For a bounding box:
            if (self.tile_num_cols and self.tile_num_cols > 1) or \
                    (self.tile_num_rows and self.tile_num_rows > 1 ):
                # Tiling pattern params ignored if bounding box params specified.
                #errmsg = "Specifing both ROI and tiling params currently not supported."
                #raise ValueError( errmsg )
                import warnings
                warnings.warn( "Tiling pattern params ignored if bounding box params specified")
            x1 = self.x
            y1 = self.y
            x2 = x1 + self.w - 1
            y2 = y1 + self.h - 1
            # C++ API calls for copying desired pixels into empty ImageMatrix instance:
            preprocessed_local_px_plane = PyImageMatrix()
            retval = preprocessed_local_px_plane.submatrix( preprocessed_full_px_plane, x1, y1, x2, y2 )
            if 1 != retval:
                e = 'Error cropping bounding box ({},{}),({},{}) from image "{}" w={} h={}'
                e = e.format( x1, y1, x2, y2, preprocessed_full_px_plane.source,
                        preprocessed_full_px_plane.width, preprocessed_full_px_plane.height )
                raise ValueError( e )
        elif( self.tile_num_cols and self.tile_num_cols > 1 ) or \
                    ( self.tile_num_rows and self.tile_num_rows > 1 ):
            # for tiling: figure out bounding box for this tile:
            # You have to use floor division here, as opposed to rounding as was previously
            # used; e.g. an 11x11 image with requested 3x3 tile scheme has a 3.66x3.66
            # dimension, you can't round up to 4x4 tiles.
            w = int( float( preprocessed_full_px_plane.width ) / self.tile_num_cols )
            h = int( float( preprocessed_full_px_plane.height ) / self.tile_num_rows )
            x1 = self.tile_col_index * w
            x2 = ( ( self.tile_col_index + 1 ) * w ) - 1
            y1 = self.tile_row_index * h
            y2 = ( ( self.tile_row_index + 1 ) * h ) - 1
            preprocessed_local_px_plane = PyImageMatrix()
            retval = preprocessed_local_px_plane.submatrix( preprocessed_full_px_plane, x1, y1, x2, y2 )
            if 1 != retval:
                e = 'Error cropping tile (col {},row {}) with tiling scheme {}col X {}row '
                e += 'with bounding box ({},{}),({},{}) from image "{}" w={} h={}'
                e = e.format( self.tile_col_index, self.tile_row_index, self.tile_num_cols,
                            self.tile_num_rows, x1, y1, x2, y2, preprocessed_full_px_plane.source,
                            preprocessed_full_px_plane.width, preprocessed_full_px_plane.height )
                raise ValueError( e )
        else:
            preprocessed_local_px_plane = preprocessed_full_px_plane

        if cache:
            self.preprocessed_local_px_plane = preprocessed_local_px_plane
        return preprocessed_local_px_plane

    #================================================================
    def GenerateFeatures( self, write_to_disk=True, update_samp_opts_from_pathname=None,
            cache=False, quiet=True ):
        """@brief Loads precalculated features, or calculates new ones, based on which instance
        attributes have been set, and what their values are.

        write_to_disk (bool) - save features to text file which by convention has extension ".sig"
        update_samp_opts_from_pathname (bool) - If a .sig file exists, don't overwrite
            self's sampling options from the sampling options in the .sig file pathname.
 
        Returns self for convenience."""

        # 0: What features does the user want?
        # 1: are there features already calculated somewhere?
        # 2: if so are they current/complete/expected/correct?
        # 3: if not, what's left to calculate?
        # 4: Calculate the rest
        # 5: Reduce the features down to what the user asked for

        if self.values is not None and len( self.values ) != 0:
            return self

        partial_load = False
        try:
            self.LoadSigFile( quiet=quiet, \
                    update_samp_opts_from_pathname=update_samp_opts_from_pathname )
            # FIXME: Here's where you'd calculate a small subset of features
            # and see if they match what was loaded from file. The file could be corrupted
            # incomplete, or calculated with different options, e.g., -S1441
            return self
        except IOError:
            # File doesn't exist
            pass
        except WrongFeatureSetVersionError:
            # File has different feature version than desired
            pass
        except IncompleteFeatureSetError:
            # LoadSigFile should create a FeatureComputationPlan
            if not quiet:
                print 'Loaded {0} features from disk for sample "{1}"'.format(
                        len( self.temp_names ), self.name )
            partial_load = True

        # All hope is lost, calculate features.

        # Use user-assigned feature computation plan, if provided=quiet:
        if self.feature_computation_plan != None:
            comp_plan = self.feature_computation_plan

            # I Commented the following out because the computation plan may only reflect
            # the subset of features that haven't been calculated yet:
            # comp_plan.feature_vec_type seems to only contain the minor version
            # i.e., number after the '.'. Assume major version is the latest.
            #self.feature_set_version = '{0}.{1}'.format( 
            #        feature_vector_major_version, comp_plan.feature_vec_type )
        else:
            major, minor = self.feature_set_version.split('.')
            if minor == '0':
                comp_plan = GenerateFeatureComputationPlan( self.feature_names )
            elif minor == '1':
                comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSet()
            elif minor == '2':
                comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSetLong()
            elif minor == '3':
                comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSetColor()
            elif minor == '4':
                comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSetLongColor()
            else:
                raise ValueError( "Not sure which features you want." )
            self.feature_computation_plan = comp_plan

        if self.rot is not None:
            # void Rotate (const ImageMatrix &matrix_IN, double angle);
            raise NotImplementedError( "FIXME: Implement rotations." )

        px_plane = self.GetPreprocessedLocalPixelPlane( cache=cache )

        # pre-allocate space where the features will be stored (C++ std::vector<double>)
        tmp_vec = wndcharm.DoubleVector( comp_plan.n_features )

        # Get an executor for this plan and run it
        plan_exec = wndcharm.FeatureComputationPlanExecutor( comp_plan )
        if not quiet:
            print "CALCULATING FEATURES FROM", self.source_filepath, self
        plan_exec.run( px_plane, tmp_vec, 0 )

        # get the feature names from the plan
        comp_names = [ comp_plan.getFeatureNameByIndex(i) for i in xrange( comp_plan.n_features ) ]

        # convert std::vector<double> to native python list of floats
        comp_vals = np.array( list( tmp_vec ) )

        # Feature Reduction/Reorder step:
        # Feature computation may give more features than are asked for by user, or out of order.
        if self.feature_names:
            if self.feature_names != comp_names:
                if partial_load:
                    # If we're here, we've already loaded some but not all of the features
                    # we need. Take what we've already loaded and slap it at the end 
                    # of what was calculated.  Doesn't matter if some of the features are
                    # redundant, because the .index() method returns the first item it finds.
                    # FIXME: if there is overlap between what was loaded and what was 
                    # calculated, check to see that they match.
                    comp_names.extend( self.temp_names )
                    comp_vals = np.hstack( (comp_vals,  self.temp_values ))
                    del self.temp_names
                    del self.temp_values
            comp_vals = np.array( [ comp_vals[ comp_names.index( name ) ] for name in self.feature_names ] )
        self.feature_names = comp_names
        self.values = comp_vals

        if not quiet:
            if len( comp_vals ) != len( self ):
                print "CALCULATED {0} TOTAL FEATURES, REDUCED TO: {1}".format(
                        len( comp_vals ), self )
            else:
                print "CALCULATED: " + str( self )

        # FIXME: maybe write to disk BEFORE feature reduce? Provide flag to let user decide?
        if write_to_disk:
            self.ToSigFile( quiet=quiet )

        # Feature names need to be modified for their sampling options.
        # Base case is that channel goes in the innermost parentheses, but really it's not
        # just channel, but all sampling options.
        # For now, let the FeatureSpace constructor code handle the modification of feature names
        # for its own self.feature_names
        return self

    #==============================================================
    def CompatibleFeatureSetVersion( self, version ):
        """Note that if either minor version is 0 (i.e not a standard feature vector)
        we return true while in fact, the compatibility is unknown"""

        try:
            version = version.feature_set_version
        except AttributeError:
            # Assume version is the version string then.
            pass

        if self.feature_set_version is None or version is None:
            err_str = "Can't tell if FeatureSpace {0} is compatible with version {1} because "
            if self.feature_set_version is None and version is None:
                err_str += "both are null."
            elif self.feature_set_version is None:
                err_str += "the FS instance's version string is null."
            else:
                err_str += "input version string is null."
            raise AttributeError( err_str.format( self.name, version ) )

        their_major, their_minor = [ int(v) for v in version.split('.',1) ]
        our_major, our_minor = [ int(v) for v in self.feature_set_version.split('.',1) ]

        if their_major != our_major:
            return False
        if our_minor and their_minor and our_minor != their_minor:
            return False

        return True

    #==============================================================
    def Normalize( self, reference_features=None, inplace=True, zscore=False,
            non_real_check=True, quiet=False ):
        """By convention, the range of feature values in the WND-CHARM algorithm are
        normalized on the interval [0,100]. Normalizing is useful in making the variation 
        of features human readable. Normalized samples are only comprable if they've been 
        normalized against the same feature maxima/minima."""

        if self.normalized_against:
            # I've already been normalized, and you want to normalize me again?
            raise ValueError( "{0} \"{1}\" has already been normalized against {2}.".format (
                self.__class__.__name__, self.name, self.normalized_against ) )

        newdata = {}
        mins = None
        maxs = None
        means = None
        stdevs = None

        if not reference_features:
            # Specific to FeatureVector implementation:
            # Doesn't make sense to Normalize a 1-D FeatureVector against itself
            # The FeatureSpace implementation of this function has stuff in this block
            err = "Can't normalize {0} \"{1}\" against itself (Normalize() called with blank arg)."
            raise ValueError( err.format( self.__class__.__name__, self.name ) )
        else:
            # Recalculate my feature space according to maxima/minima in reference_features
            if reference_features.feature_names != self.feature_names:
                err_str = "Can't normalize {0} \"{1}\" against {2} \"{3}\": Features don't match.".format(
                  self.__class__.__name__, self.name,
                    reference_features.__class__.__name__, reference_features.name )
                raise ValueError( err_str )
            if not self.CompatibleFeatureSetVersion( reference_features ):
                err_str = 'Incompatible feature versions: "{0}" ({1}) and "{2}" ({3})'
                raise ValueError( err_str.format( self.name, self.feature_set_version,
                    reference_features.name, reference_features.feature_set_version ) )

            if not quiet:
                # Specific to FeatureVector implementation:
                # no num_samples member:
                print 'Normalizing {0} "{1}" ({2} features) against {3} "{4}"'.format(
                    self.__class__.__name__, self.name, len( self.feature_names),
                    reference_features.__class__.__name__, reference_features.name )

            # Need to make sure there are feature minima/maxima to normalize against:
            if not reference_features.normalized_against:
                reference_features.Normalize( quiet=quiet, zscore=zscore,
                        non_real_check=non_real_check )

            if not zscore:
                mins = reference_features.feature_minima
                maxs = reference_features.feature_maxima
            else:
                means = reference_features.feature_means
                stdevs = reference_features.feature_stdevs

            newdata['normalized_against'] = reference_features

        if inplace:
            fv = self.values
        else:
            fv = np.copy( self.values )

        retval = normalize_by_columns( fv, mins, maxs, means, stdevs, zscore, non_real_check )
        newdata['values'] = fv
        newdata['feature_minima'], newdata['feature_maxima'], \
                newdata['feature_means'], newdata['feature_stdevs'] = retval

        if inplace:
            return self.Update( **newdata )
        return self.Derive( **newdata )

    #==============================================================
    def FeatureReduce( self, requested_features, inplace=False, quiet=False ):
        """Returns a new FeatureVector that contains a subset of the data by dropping
        features (columns), and/or rearranging columns.

        requested_features := an object with a "feature_names" member
            (FeatureVector/FeatureSpace/FeatureWeights) or an iterable containing
            strings that are feature names.

        Implementation detail: compares input "requested_features" to self.feature_names,
        and "requested_features" becomes the self.feature_names of the returned FeatureVector."""

        try:
            requested_features = requested_features.feature_names
        except AttributeError:
            # assume it's already a list then
            pass

        # Check that self's featurelist contains all the features in requested_features
        selfs_features = set( self.feature_names )
        their_features = set( requested_features )
        if not their_features <= selfs_features:
            missing_features_from_req = their_features - selfs_features
            err_str = "Feature Reduction error:\n"
            err_str += '{0} "{1}" is missing '.format( self.__class__.__name__, self.name )
            err_str += "{0}/{1} features that were requested in the feature reduction list.".format(\
                    len( missing_features_from_req ), len( requested_features ) )
            err_str += "\nDid you forget to convert the feature names into their modern counterparts?"
            raise IncompleteFeatureSetError( err_str )

        # The implementation of FeatureReduce here is similar to FeatureSpace.FeatureReduce
        # Here is where the implementations diverge"
        num_features = len( requested_features )

        if not quiet:
            orig_len = len( self )

        newdata = {}
        newdata[ 'name' ] = self.name + "(feature reduced)"
        newdata[ 'feature_names' ] = requested_features
        newdata[ 'num_features' ] = num_features

        new_order = [ self.feature_names.index( name ) for name in requested_features ]

        # N.B. 1-D version used here, contrast with FeatureSpace.FeatureReduce() implementation.
        newdata[ 'values' ] = self.values[ new_order ]

        if self.feature_maxima is not None:
            newdata[ 'feature_maxima' ] = self.feature_maxima[ new_order ]
        if self.feature_minima is not None:
            newdata[ 'feature_minima' ] = self.feature_minima[ new_order ]
        if self.feature_means is not None:
            newdata[ 'feature_means' ] = self.feature_means[ new_order ]
        if self.feature_stdevs is not None:
            newdata[ 'feature_stdevs' ] = self.feature_stdevs[ new_order ]

        # If the feature vectors sizes changed then they are no longer standard feature vectors.
        if self.feature_set_version is not None and num_features != self.num_features:
            newdata[ 'feature_set_version' ] = \
                    "{0}.0".format( self.feature_set_version.split('.',1)[0] )

        if inplace:
            newfv = self.Update( **newdata )
        else:
            newfv = self.Derive( **newdata )

        if not quiet:
            print "FEATURE VECTOR REDUCED (orig len {0}): {1}".format( orig_len, newfv )
        return newfv

    #================================================================
    def LoadSigFile( self, sigfile_path=None, update_samp_opts_from_pathname=None,
            quiet=False ):
        """Load computed features from a sig file.

        Desired features indicated by strings currently in self.feature_names.
        Desired feature set version indicated self.feature_set_version.

        Compare what got loaded from file with desired.

        update_samp_opts_from_pathname (bool) - If a .sig file exists, don't overwrite
            self's sampling options from the sampling options in the .sig file pathname"""

        import re

        if sigfile_path:
            path = sigfile_path
            if update_samp_opts_from_pathname is None:
                update_samp_opts_from_pathname = True
        elif self.auxiliary_feature_storage:
            path = self.auxiliary_feature_storage
            if update_samp_opts_from_pathname is None:
                update_samp_opts_from_pathname = True
        else:
            path = self.GenerateSigFilepath()
            update_samp_opts_from_pathname = False

        with open( path ) as infile:

            # First, check to see feature set versions match:
            firstline = infile.readline()
            m = re.match( r'^(\S+)\s*(\S+)?$', firstline )
            if not m:
                # Deprecate old-style naming support anyway, those features are pretty buggy
                # -CEC 20150104
                raise ValueError( "Can't read a WND-CHARM feature set version from file {0}. File my be corrupted or calculated by an unsupported version of WND-CHARM. Recalculate features and try again.".format( path ) )
                #input_major = 1
                # For ANCIENT sig files, with features calculated YEARS ago
                # Cleanup for legacy edge case:
                # Set the minor version to the vector type based on # of features
                # The minor versions should always specify vector types, but for
                # version 1 vectors, the version is not written to the file.
                #self.feature_set_version = "1." + str(
                #feature_vector_minor_version_from_num_features_v1.get( len( self.values ),0 ) )
                # This is really slow:
                #for i, name in enumerate( names ):
                #retval = wndcharm.FeatureNames.getFeatureInfoByName( name )
                #if retval:
                #    self.feature_names[i] = retval.name
                #else:
                # self.feature_names[i] = name
                # Use pure Python for old-style name translation
                #from wndcharm import FeatureNameMap
                #self.feature_names = FeatureNameMap.TranslateToNewStyle( feature_names )
            else:
                class_id, input_fs_version = m.group( 1, 2 )
                input_fs_major_ver, input_fs_minor_ver = input_fs_version.split('.')
            if self.feature_set_version:
                desired_fs_major_ver, desired_fs_minor_ver = self.feature_set_version.split('.')
                if desired_fs_major_ver != input_fs_major_ver:
                    errstr = 'Desired feature set version "{0}" different from "{1}" in file {2}'
                    raise WrongFeatureSetVersionError(
                            errstr.format( desired_fs_major_ver, input_fs_major_ver, path ) )

            # 2nd line is path to original tiff file, which may be nonsense
            # if sig file was moved post-feature calculation.
            orig_source_tiff_path = infile.readline()
            if self.source_filepath is None:
                from os.path import exists
                # FIXME: Maybe try a few directories?
                if exists( orig_source_tiff_path ):
                    self.source_filepath = orig_source_tiff_path

            # Load data into local variables:
            # Loading these sig files takes way too long.
            # Try to speed up by being more explicit
            # hardcode the num features check:
            if (input_fs_major_ver == '2' or input_fs_major_ver== '3') and \
                    input_fs_minor_ver != '0':
                # We know exactly how long these feature vectors are gonna be
                # so allocate just the right amount of memory:
                vec_len = ver_to_num_feats_map[ int( input_fs_minor_ver ) ]
                values = np.zeros( vec_len )
                names = [None] * vec_len
                for i, line in enumerate( infile ):
                    val, name = line.rstrip('\n').split( None, 1 )
                    values[i] = float( val )
                    names[i] = name
            else:
                # If we're here, then we don't know for sure how many features are
                # in this file, so do it the old, slow way:
                values, names = \
                    zip( *[ line.split( None, 1 ) for line in infile.read().splitlines() ] )
                values = [ float(_) for _ in values ]

        # Re: converting read-in text to numpy array of floats, np.fromstring is a 3x PIG:
        # %timeit out = np.array( [ float(val) for val in thing ] )
        # 10 loops, best of 3: 38.3 ms per loop
        # %timeit out = np.fromstring( " ".join( thing ), sep=" " )
        # 10 loops, best of 3: 98.1 ms per loop

        # By now we would know by know if there was a sigfile processing error,
        # e.g., file doesn't exist.
        # Safe to set this member now if not already set
        if not self.auxiliary_feature_storage:
            self.auxiliary_feature_storage = path

        # Check to see that the sig file contains all of the desired features:
        if self.feature_names:
            if self.feature_names == names:
                # Perfect! Do nothing.
                pass
            else:
                features_we_want = set( self.feature_names )
                features_we_have = set( names )
                if not features_we_want <= features_we_have:
                    # Need to calculate more features
                    missing_features = features_we_want - features_we_have
                    # create a feature computation plan based on missing features only:
                    self.feature_computation_plan = GenerateFeatureComputationPlan( missing_features )
                    # temporarily store loaded features in temp members to be used by 
                    # self.GenerateFeatures to create the final feature vector.
                    self.temp_names = names
                    #self.temp_values = [ float( val ) for val in values ]
                    self.temp_values = values
                    raise IncompleteFeatureSetError
                else:
                    # If you get to here, we loaded MORE features than asked for,
                    # or the features are out of desired order, or both.
                    values = np.array( [ values[ names.index( name ) ] for name in self.feature_names ] )
        else:
            # User didn't indicate what features they wanted.
            # It's a pretty dangerous assumption to make that the user just "got 
            # what they wanted" by loading the file, but danger is my ... middle name ;-)
            self.feature_names = list( names )

        #self.values = np.array( [ float( val ) for val in values ] )
        self.values = values

        if not self.name or update_samp_opts_from_pathname:
            # Subtract path so that path part doesn't become part of name
            from os.path import basename
            # Pull sampling options from filename
            path_removed = basename( path )
            self.name = path_removed

        if update_samp_opts_from_pathname:
            result = self.sig_filename_parser.search( path_removed )
            if result:
                self.Update( **result.groupdict() )
        else:
            self.Update()

        if not quiet:
            print "LOADED ", str( self )
        return self

    #================================================================
    @classmethod
    def NewFromSigFile( cls, sigfile_path, image_path=None, quiet=False ):
        """@return  - An instantiated FeatureVector class with feature names translated from
        the old naming convention, if applicable."""
        return cls( source_filepath=image_path ).LoadSigFile( sigfile_path, quiet )

    #================================================================
    def ToSigFile( self, path=None, quiet=False ):
        """Write features C-WND-CHARM .sig file format

        If filepath is specified, you get to name it whatever you want and put it
        wherever you want. Otherwise, it's named according to convention and placed 
        next to the image file in its directory."""
        from os.path import exists
        if path:
            self.auxiliary_feature_storage = path
        elif self.auxiliary_feature_storage is not None:
            path = self.auxiliary_feature_storage
        else:
            path = self.auxiliary_feature_storage = self.GenerateSigFilepath()

        if not quiet:
            if exists( path ):
                print "Overwriting {0}".format( path )
            else:
                print 'Writing signature file "{0}"'.format( path )
        
        with open( path, "w" ) as out:
            # FIXME: line 1 contains class membership and version
            # Just hardcode the class membership for now.
            out.write( "0\t{0}\n".format( self.feature_set_version ) )
            out.write( "{0}\n".format( self.source_filepath ) )
            for val, name in zip( self.values, self.feature_names ):
                out.write( "{0:0.8g}\t{1}\n".format( val, name ) )

# end definition class FeatureVector

#=============================================================================
class SlidingWindow( FeatureVector ):
    """Object for sliding window-style image analysis/feature calculation.

    self.sample() generator method yields a fully-qualified FeatureVector object with
    5D sampling parameters appropriate for given window location.

    Contstructor for this object without args yields classic "tile" behavior, i.e.,
    contiguous, non-overlapping ROIs specified by tile_num_rows & tile_num_cols.
    Overlapping/non-contiguous behavior enabled when user specifies instance attributes
    w, h, deltax and/or deltay."""

    count = 0

    def __init__( self, deltax=None, deltay=None, masks=None, default_label=None,
            verbose=False, *args, **kwargs ):
        """Will open source image to get dimensions to calculate number of window positions,
        UNLESS using "classic" tiling, a.k.a. contiguous, non-overlapping tiles. Passes
        args/kwargs straight to FeatureVector constructor.

        Arguments:
            deltax & deltay - int (default None)
                Number of pixels to move scanning window vertically/horizontally.

            masks - dict, default none
                Keys are names of classes, values are paths to mask images"""

        super( SlidingWindow, self ).__init__( *args, **kwargs )
        self.deltax = deltax
        self.deltay = deltay
        # list of sample sequence ids with the current scanning pattern that this window
        # should stop at. Allows user to skip window positions, or delegate certain
        # window positions to different processors.
        #self.desired_positions = None
        self.num_positions = None
        self.verbose = verbose


        # Mask attributes
        self.base_mask = None
        self.class_masks = None
        self.default_label = default_label

        if self.sample_group_id is None:
            # All sample group ids are the same for all sliding window positions
            self.sample_group_id = self.count
            self.count += 1

        if self.tile_num_rows is not None and self.tile_num_cols is not None: 
            # Case 1: Standard tiling. Sig files will use -tNxM_n_m notation.
            self.standard_tiling = True
            self.num_positions = self.tile_num_rows * self.tile_num_cols
            self.tile_col_index = None
            self.tile_row_index = None
        elif self.deltax is not None and self.deltay is not None and \
                self.w is not None and self.h is not None:
            self.standard_tiling = False
            # Case 2: Potential overlaping/non-contiguous window locations.
            # Sig files will use -B_x_y_w_h notation.
            if not self.preprocessed_full_px_plane_width or \
                    not self.preprocessed_full_px_plane_height:
                # No choice but to load pixel plane to get dimensions to derive num_positions.
                # For multiprocessing purposes don't cache the pixel plane.
                precropped_img = self.GetPreprocessedFullPixelPlane( cache=False )
 
            self.sliding_window_num_cols = int( ( precropped_img.width - self.w ) / self.deltax )
            self.sliding_window_num_rows = int( ( precropped_img.height - self.h ) / self.deltay )
            self.num_positions = self.sliding_window_num_rows * self.sliding_window_num_cols
            self.sliding_window_col_index = None
            self.sliding_window_row_index = None
        else:
            raise ValueError( "Could not obtain window/slide dimensions from instance atribute params provided." )

        if masks:
            from skimage.io import imread
            import numpy as np
            def LoadMask(mask_path):
                mask = imread( mask_path, as_grey=True ).astype( np.uint8 )
                return mask > 5

            base_mask_path = masks.get( 'base', None )
            if base_mask_path:
                self.base_mask = LoadMask( base_mask_path )
                del masks['base']

            if len( masks ) > 0:
                self.class_masks = {}
                for class_name, class_mask_path in masks.items():
                    self.class_masks[ class_name ] = LoadMask( class_mask_path )

    def increment_position( self ):

        if self.standard_tiling:
            if self.tile_row_index == None:
                self.tile_row_index = 0
            else:
                self.tile_row_index += 1

            if self.tile_col_index == None:
                self.tile_col_index = 0
            elif self.tile_row_index == self.tile_num_rows:
                self.tile_col_index += 1
                self.tile_row_index = 0

            if self.tile_col_index == self.tile_num_cols:
                raise StopIteration

        else:
            if self.sliding_window_row_index == None:
                self.sliding_window_row_index = 0
                self.y = 0
            else:
                self.sliding_window_row_index += 1
                self.y += self.deltay

            if self.sliding_window_col_index == None:
                self.sliding_window_col_index = 0
                self.x = 0
            elif ( (self.y + self.h - 1) > self.preprocessed_full_px_plane_height) or \
                    self.sliding_window_row_index == self.sliding_window_num_rows:
                self.sliding_window_col_index += 1
                self.x += self.deltax
                self.sliding_window_row_index = 0
                self.y = 0

            if ( (self.x + self.w - 1) > self.preprocessed_full_px_plane_width) or \
                    self.sliding_window_col_index == self.sliding_window_num_cols:
                raise StopIteration

    def get_next_position( self ):

        while True :
            self.increment_position()
            left = self.x
            right = self.x + self.w
            top = self.y
            bottom = self.y + self.h

            if self.base_mask is not None:
                if self.base_mask[ top:bottom, left:right ].all():
                    break
                else:
                    if self.verbose:
                        print "Pos={} row={}, col={}, x={}, y={}, w={}, h={} NOT in basemask".format(
                                self.sample_sequence_id, self.sliding_window_row_index, 
                                self.sliding_window_col_index, self.x, self.y, self.w, self.h )
            else:
                break

        if self.class_masks is not None:
            flag = False
            for class_name, class_mask in self.class_masks.items():
                if class_mask[ top:bottom, left:right ].all():
                    self.ground_truth_label = class_name
                    flag = True
                    break
            if not flag:
                self.ground_truth_label = self.default_label

        if self.verbose:
            print "Pos={} row={}, col={}, x={}, y={}, w={}, h={}, class={}".format(
                self.sample_sequence_id, self.sliding_window_row_index, 
                self.sliding_window_col_index, self.x, self.y, self.w, self.h, self.ground_truth_label )

        if self.sample_sequence_id == None:
            self.sample_sequence_id = 0
        else:
            self.sample_sequence_id += 1

        return self

    def sample( self ):

        self.values = None
        self.preprocessed_local_px_plane = None
        self.auxiliary_feature_storage = None

        while True:
            yield self.get_next_position().Derive( base=True )
