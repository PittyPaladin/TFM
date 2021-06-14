cmake_minimum_required(VERSION 3.1.0)

project(pcd_streamer)

cmake_policy(SET CMP0072 NEW)
find_package(realsense2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED )
find_package(glfw3 3.3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
LIST(REMOVE_ITEM ${PCL_LIBRARIES} "vtkproj4")

set(DEPENDENCIES realsense2 glfw ${OPENGL_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add the application sources to the target
add_executable(${PROJECT_NAME} src/pcd_streamer.cpp src/example.hpp)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/RunDir)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})
