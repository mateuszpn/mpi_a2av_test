cmake_minimum_required (VERSION 3.20)
project(mpi_a2av_test LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/intel/oneapi/mpi/latest/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

include(FetchContent)

FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
  GIT_TAG 5300fe3)
FetchContent_MakeAvailable(range-v3)


find_package(MPI REQUIRED)
add_link_options(-L/opt/intel/oneapi/mpi/latest/lib/release -lmpi)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(${PROJECT_NAME} test.cpp)
add_custom_target(run ./${PROJECT_NAME})