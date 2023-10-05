set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/intel/oneapi/mpi/latest/include")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

find_package(MPI REQUIRED)
add_link_options(-L/opt/intel/oneapi/mpi/latest/lib/release -lmpi)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

cmake_minimum_required (VERSION 3.4)
project(mpi_a2av_test LANGUAGES CXX)

add_executable(${PROJECT_NAME} test.cpp)
add_custom_target(run ./${PROJECT_NAME})