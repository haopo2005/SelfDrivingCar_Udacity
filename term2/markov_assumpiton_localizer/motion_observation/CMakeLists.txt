cmake_minimum_required(VERSION 2.8)
project(BayesFilter)
SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++11)

set(INC_LOCAL_DIR  ${BayesFilter_SOURCE_DIR})  
AUX_SOURCE_DIRECTORY(${BayesFilter_SOURCE_DIR} SOURCE_FILES)
include_directories(${INC_LOCAL_DIR}) 
add_executable( BayesFilter ${SOURCE_FILES})
target_link_libraries(BayesFilter)
