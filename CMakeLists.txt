# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 4.0
cmake_minimum_required(VERSION 3.15...4.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fpsample MODULE src/lib.cpp)
target_link_libraries(_fpsample PRIVATE pybind11::headers)

target_compile_definitions(_fpsample PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _fpsample DESTINATION fpsample)
