cmake_minimum_required(VERSION 3.0.0)
project(NN VERSION 0.1.0)
aux_source_directory(src SOURCES)
set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

add_executable(NN ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
