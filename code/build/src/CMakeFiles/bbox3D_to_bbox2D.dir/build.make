# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/david/TFM/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/david/TFM/code/build

# Include any dependencies generated for this target.
include src/CMakeFiles/bbox3D_to_bbox2D.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/bbox3D_to_bbox2D.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/bbox3D_to_bbox2D.dir/flags.make

src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o: src/CMakeFiles/bbox3D_to_bbox2D.dir/flags.make
src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o: ../src/bbox3D_to_bbox2D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/david/TFM/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o"
	cd /home/david/TFM/code/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o -c /home/david/TFM/code/src/bbox3D_to_bbox2D.cpp

src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.i"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/david/TFM/code/src/bbox3D_to_bbox2D.cpp > CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.i

src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.s"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/david/TFM/code/src/bbox3D_to_bbox2D.cpp -o CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.s

# Object files for target bbox3D_to_bbox2D
bbox3D_to_bbox2D_OBJECTS = \
"CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o"

# External object files for target bbox3D_to_bbox2D
bbox3D_to_bbox2D_EXTERNAL_OBJECTS =

../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/bbox3D_to_bbox2D.dir/bbox3D_to_bbox2D.cpp.o
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/bbox3D_to_bbox2D.dir/build.make
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libqhull.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/libOpenNI.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/libOpenNI2.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libfreetype.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libz.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpng.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libtiff.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libexpat.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_gapi.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_stitching.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_aruco.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_bgsegm.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_bioinspired.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ccalib.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn_superres.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dpm.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_face.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_freetype.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_fuzzy.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_hfs.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_img_hash.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_intensity_transform.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_line_descriptor.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_mcc.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_quality.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_rapid.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_reg.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_rgbd.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_saliency.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_stereo.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_structured_light.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_superres.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_surface_matching.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_tracking.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_videostab.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xfeatures2d.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xobjdetect.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xphoto.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libfreetype.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libz.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libSM.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libICE.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libX11.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libXext.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libXt.so
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_shape.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_highgui.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_datasets.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_plot.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_text.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ml.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_optflow.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ximgproc.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_video.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_videoio.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_objdetect.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_calib3d.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_features2d.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_flann.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_photo.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgproc.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: /usr/local/lib/libopencv_core.so.4.5.1
../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/bbox3D_to_bbox2D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/david/TFM/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so"
	cd /home/david/TFM/code/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bbox3D_to_bbox2D.dir/link.txt --verbose=$(VERBOSE)
	cd /home/david/TFM/code/build/src && /usr/bin/strip /home/david/TFM/code/src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
src/CMakeFiles/bbox3D_to_bbox2D.dir/build: ../src/bbox3D_to_bbox2D.cpython-38-x86_64-linux-gnu.so

.PHONY : src/CMakeFiles/bbox3D_to_bbox2D.dir/build

src/CMakeFiles/bbox3D_to_bbox2D.dir/clean:
	cd /home/david/TFM/code/build/src && $(CMAKE_COMMAND) -P CMakeFiles/bbox3D_to_bbox2D.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/bbox3D_to_bbox2D.dir/clean

src/CMakeFiles/bbox3D_to_bbox2D.dir/depend:
	cd /home/david/TFM/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/david/TFM/code /home/david/TFM/code/src /home/david/TFM/code/build /home/david/TFM/code/build/src /home/david/TFM/code/build/src/CMakeFiles/bbox3D_to_bbox2D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/bbox3D_to_bbox2D.dir/depend
