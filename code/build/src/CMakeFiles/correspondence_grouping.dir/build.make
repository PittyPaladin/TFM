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
include src/CMakeFiles/correspondence_grouping.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/correspondence_grouping.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/correspondence_grouping.dir/flags.make

src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o: src/CMakeFiles/correspondence_grouping.dir/flags.make
src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o: ../src/correspondence_grouping.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/david/TFM/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o"
	cd /home/david/TFM/code/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o -c /home/david/TFM/code/src/correspondence_grouping.cpp

src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.i"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/david/TFM/code/src/correspondence_grouping.cpp > CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.i

src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.s"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/david/TFM/code/src/correspondence_grouping.cpp -o CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.s

src/CMakeFiles/correspondence_grouping.dir/utils.cpp.o: src/CMakeFiles/correspondence_grouping.dir/flags.make
src/CMakeFiles/correspondence_grouping.dir/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/david/TFM/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/correspondence_grouping.dir/utils.cpp.o"
	cd /home/david/TFM/code/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/correspondence_grouping.dir/utils.cpp.o -c /home/david/TFM/code/src/utils.cpp

src/CMakeFiles/correspondence_grouping.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/correspondence_grouping.dir/utils.cpp.i"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/david/TFM/code/src/utils.cpp > CMakeFiles/correspondence_grouping.dir/utils.cpp.i

src/CMakeFiles/correspondence_grouping.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/correspondence_grouping.dir/utils.cpp.s"
	cd /home/david/TFM/code/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/david/TFM/code/src/utils.cpp -o CMakeFiles/correspondence_grouping.dir/utils.cpp.s

# Object files for target correspondence_grouping
correspondence_grouping_OBJECTS = \
"CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o" \
"CMakeFiles/correspondence_grouping.dir/utils.cpp.o"

# External object files for target correspondence_grouping
correspondence_grouping_EXTERNAL_OBJECTS =

../RunDir/correspondence_grouping: src/CMakeFiles/correspondence_grouping.dir/correspondence_grouping.cpp.o
../RunDir/correspondence_grouping: src/CMakeFiles/correspondence_grouping.dir/utils.cpp.o
../RunDir/correspondence_grouping: src/CMakeFiles/correspondence_grouping.dir/build.make
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libboost_system.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libqhull.so
../RunDir/correspondence_grouping: /usr/lib/libOpenNI.so
../RunDir/correspondence_grouping: /usr/lib/libOpenNI2.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libfreetype.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libz.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libjpeg.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpng.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libtiff.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libexpat.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libfreetype.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libz.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libGLEW.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libSM.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libICE.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libX11.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libXext.so
../RunDir/correspondence_grouping: /usr/lib/x86_64-linux-gnu/libXt.so
../RunDir/correspondence_grouping: src/CMakeFiles/correspondence_grouping.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/david/TFM/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../RunDir/correspondence_grouping"
	cd /home/david/TFM/code/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/correspondence_grouping.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/correspondence_grouping.dir/build: ../RunDir/correspondence_grouping

.PHONY : src/CMakeFiles/correspondence_grouping.dir/build

src/CMakeFiles/correspondence_grouping.dir/clean:
	cd /home/david/TFM/code/build/src && $(CMAKE_COMMAND) -P CMakeFiles/correspondence_grouping.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/correspondence_grouping.dir/clean

src/CMakeFiles/correspondence_grouping.dir/depend:
	cd /home/david/TFM/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/david/TFM/code /home/david/TFM/code/src /home/david/TFM/code/build /home/david/TFM/code/build/src /home/david/TFM/code/build/src/CMakeFiles/correspondence_grouping.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/correspondence_grouping.dir/depend

