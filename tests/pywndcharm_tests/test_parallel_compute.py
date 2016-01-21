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

import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

import re
import numpy as np

from os.path import sep, dirname, realpath, join
from tempfile import mkdtemp
from shutil import rmtree

pychrm_test_dir = dirname( realpath( __file__ ) ) #WNDCHARM_HOME/tests/pychrm_tests
wndchrm_test_dir = dirname( pychrm_test_dir ) + sep + 'wndchrm_tests'

from wndcharm.FeatureSpace import FeatureSpace
from wndcharm.FeatureVector import FeatureVector

class Test_parallel_compute( unittest.TestCase ):
    """Parallel computation of image features"""
    maxDiff = None

    #@unittest.expectedFailure("")
    #@unittest.skip("")
    def test_ParallelTiling( self ):
        """Specify bounding box to FeatureVector, calc features, then compare
        with C++ implementation-calculated feats."""

        import zipfile
        from shutil import copy
        from tempfile import NamedTemporaryFile

        refdir = mkdtemp(prefix='ref') 
        targetdir = mkdtemp(prefix='target')

        try:
            reference_feats = pychrm_test_dir + sep + 'lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E_t6x5_REFERENCE_SIGFILES.zip'
            zf = zipfile.ZipFile( reference_feats, mode='r' )
            zf.extractall( refdir )

            img_filename = "lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E.tif"
            orig_img_filepath = pychrm_test_dir + sep + img_filename

            # copy the tiff to the tempdir so the .sig files end up there too
            copy( orig_img_filepath, targetdir )
            copy( orig_img_filepath, refdir )
            input_image_path = targetdir + sep + img_filename

            with NamedTemporaryFile( mode='w', dir=refdir, prefix='ref', delete=False ) as temp:
                ref_fof = temp.name
                temp.write( 'reference_samp\ttest_class\t{}\t{{}}\n'.format( refdir + sep + img_filename ) )
            with NamedTemporaryFile( mode='w', dir=targetdir, prefix='target', delete=False ) as temp:
                target_fof = temp.name
                temp.write( 'test_samp\ttest_class\t{}\t{{}}\n'.format( targetdir + sep + img_filename ) )

            global_sampling_options = \
                FeatureVector( long=True, tile_num_cols=6, tile_num_rows=5 )

            # Should just load reference sigs
            ref_fs = FeatureSpace.NewFromFileOfFiles( ref_fof, quiet=False,
                 global_sampling_options=global_sampling_options )
            target_fs = FeatureSpace.NewFromFileOfFiles( target_fof, n_jobs=True,
                 quiet=False, global_sampling_options=global_sampling_options )

            #from numpy.testing import assert_allclose
            #self.assertTrue( assert_allclose( ref_fs.data_matrix, target_fs.data_matrix ) )
            from wndcharm.utils import compare
            for row_num, (ref_row, test_row) in enumerate( zip( ref_fs.data_matrix, target_fs.data_matrix )):
                retval = compare( ref_row, test_row )
                if retval == False:
                    print "error in sample row", row_num
                    print "FIT: ", ref_fs._contiguous_sample_names[row_num], "FOF", target_fs._contiguous_sample_names[row_num]
                self.assertTrue( retval )
        finally:
            rmtree( refdir )
            rmtree( targetdir )

    def test_sliding_window_segmentation( self ):
        """Test utility script sliding_window_segmentation.py"""

        import zipfile
        from shutil import copy
        from tempfile import NamedTemporaryFile

        tempdir = mkdtemp()

        try:

            # Set up image to be scanned; prepopulate directory with precomputed sigs
            #import pdb; pdb.set_trace()
            img_filename = "lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E.tif"
            orig_img_filepath = pychrm_test_dir + sep + img_filename
            copy( orig_img_filepath, tempdir )
            input_image_path = tempdir + sep + img_filename
            reference_feats = pychrm_test_dir + sep + 'lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E_t6x5_REFERENCE_SIGFILES.zip'
            zf = zipfile.ZipFile( reference_feats, mode='r' )
            zf.extractall( tempdir )

            # Dump classifier in the same directory
            #zf = zipfile.ZipFile( 'lymphoma_iicbu2008_subset_EOSIN_ONLY_t5x6_v3.2features.fit.zip', mode='r' )
            #classifier_path = tempdir + sep + 'lymphoma_iicbu2008_subset_eosin_t5x6_v3.2features.fit'
            classifier_path = pychrm_test_dir + sep + 'lymphoma_iicbu2008_subset_EOSIN_ONLY_t6x5-l.fit'
            #zf.extractall( tempdir )

            # note the difference in tile args, due to differences in test data sampling:
            classifier_args = 'long=True, tile_num_cols=5, tile_num_rows=6'
            sliding_window_args = 'name="sliding_window_classifier", long=True, tile_num_cols=6, tile_num_rows=5'

            import subprocess

            #print subprocess.check_output( ['ls', tempdir ] )

            sp_args = ['../../examples/sliding_window_segmentation.py',
                    input_image_path,
                    classifier_path,
                    '--verbose',
                    '--window_args']
            sp_args.extend( sliding_window_args.split( ', ' ) )
            sp_args.append( '--classifier_args' )
            sp_args.extend( classifier_args.split( ', ' ) )

            #print args
            subprocess.call( sp_args )


            #from numpy.testing import assert_allclose
            #self.assertTrue( assert_allclose( ref_fs.data_matrix, target_fs.data_matrix ) )
            #from wndcharm.utils import compare
            #for row_num, (ref_row, test_row) in enumerate( zip( ref_fs.data_matrix, target_fs.data_matrix )):
            #    retval = compare( ref_row, test_row )
            #    if retval == False:
            #        print "error in sample row", row_num
            #        print "FIT: ", ref_fs._contiguous_sample_names[row_num], "FOF", target_fs._contiguous_sample_names[row_num]
            #    self.assertTrue( retval )
        finally:
            rmtree( tempdir )


if __name__ == '__main__':
    unittest.main()
