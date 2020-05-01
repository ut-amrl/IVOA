cmake_minimum_required(VERSION 2.8)
project(IVOA)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()
rosbuild_add_boost_directories()
rosbuild_gensrv()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

set(PROJ_INC_DIR ./include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()

find_package(OpenCV 3.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(gflags REQUIRED)
find_package(PCL REQUIRED)


include_directories( 
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${gflags_INCLUDE_DIR}
  ${PROJ_INC_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Make sure that libcnpy.so is on LD_LIBRARY_PATH
# set(LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib/)


include_directories( BEFORE include )


# Testing **********
# add_library(SHARED_LIB
#   "src/dataset.cpp"
#   "src/depth2pointcloud.cpp"
#   "src/io_access.cpp"
# )
# 
# target_link_libraries(SHARED_LIB
# ${OpenCV_LIBS}
#  boost_system -lboost_filesystem
# -ljsoncpp yaml-cpp glog
#   opencv_imgproc opencv_highgui
# )

# rosbuild_add_executable(airsim_data_processing 
#                                 "src/main_airsim_processing.cpp")
# target_link_libraries(airsim_data_processing  glog 
#                                 ${OpenCV_LIBS}
#                                  SHARED_LIB
#                                  boost_system -lboost_filesystem
#                                 -ljsoncpp yaml-cpp -lcnpy
#                                  opencv_imgproc opencv_highgui)          
# rosbuild_link_boost(airsim_data_processing thread signals system)

rosbuild_add_executable(airsim_data_processing 
                                "src/main_airsim_processing.cpp" 
                                "src/dataset.cpp"
                                "src/depth2pointcloud.cpp"
                                "src/io_access.cpp")
target_link_libraries(airsim_data_processing  glog gflags ${OpenCV_LIBS}
                                "-lpopt" boost_system -lboost_filesystem
                                -ljsoncpp yaml-cpp -lcnpy
                                 opencv_imgproc opencv_highgui)          
rosbuild_link_boost(airsim_data_processing thread signals system)


rosbuild_add_executable(mltool_evaluation
                                "src/main_mltool_eval.cpp"
                                "src/depth2pointcloud.cpp"
                                "src/evaluator.cpp"
                                "src/io_access.cpp")
target_link_libraries(mltool_evaluation  glog gflags ${OpenCV_LIBS} ${PCL_LIBRARIES}
                                "-lpopt" boost_system -lboost_filesystem
                                -ljsoncpp yaml-cpp -lcnpy
                                 opencv_imgproc opencv_highgui)          
rosbuild_link_boost(mltool_evaluation thread signals system)


