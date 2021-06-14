cmake_minimum_required(VERSION 3.4.0)

FIND_PACKAGE(PCL 1.8 REQUIRED)
FIND_PACKAGE(nlohmann_json 3.2.0 REQUIRED)
FIND_PACKAGE(realsense2 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})

set(DEPENDENCIES ${PCL_LIBRARIES} nlohmann_json::nlohmann_json)

############
# PROGRAMS #
############
ADD_EXECUTABLE(display_pcd display_pcd.cpp utils.cpp)
TARGET_LINK_LIBRARIES(display_pcd  ${DEPENDENCIES})

ADD_EXECUTABLE(icp_registration2 icp_registration2.cpp utils.cpp)
TARGET_LINK_LIBRARIES(icp_registration2  ${DEPENDENCIES})

ADD_EXECUTABLE(ndt_registration ndt_registration.cpp utils.cpp)
TARGET_LINK_LIBRARIES(ndt_registration  ${DEPENDENCIES})

ADD_EXECUTABLE(model_segmentation model_segmentation.cpp utils.cpp)
TARGET_LINK_LIBRARIES(model_segmentation  ${DEPENDENCIES})

ADD_EXECUTABLE(regiongrowing_segmentation regiongrowing_segmentation.cpp utils.cpp)
TARGET_LINK_LIBRARIES(regiongrowing_segmentation  ${DEPENDENCIES})

ADD_EXECUTABLE(regiongrowingrgb_segmentation regiongrowingrgb_segmentation.cpp utils.cpp)
TARGET_LINK_LIBRARIES(regiongrowingrgb_segmentation  ${DEPENDENCIES})

ADD_EXECUTABLE(correspondence_grouping correspondence_grouping.cpp utils.cpp)
TARGET_LINK_LIBRARIES(correspondence_grouping  ${DEPENDENCIES})

ADD_EXECUTABLE(euclidean_cluster_segmentation euclidean_cluster_segmentation.cpp utils.cpp)
TARGET_LINK_LIBRARIES(euclidean_cluster_segmentation  ${DEPENDENCIES})

ADD_EXECUTABLE(annotator annotator.cpp utils.cpp)
TARGET_LINK_LIBRARIES(annotator  ${DEPENDENCIES})

pybind11_add_module(bbox3D_to_bbox2D bbox3D_to_bbox2D.cpp)
set_target_properties(bbox3D_to_bbox2D PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
TARGET_LINK_LIBRARIES(bbox3D_to_bbox2D PRIVATE ${DEPENDENCIES} realsense2 ${OpenCV_LIBS})
