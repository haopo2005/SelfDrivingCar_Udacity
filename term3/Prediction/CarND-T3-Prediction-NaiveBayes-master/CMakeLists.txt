cmake_minimum_required(VERSION 2.8)
project(CarND-T3-Prediction-NaiveBayes)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARAD_REQUIRED ON)

add_executable(${PROJECT_NAME} ${SOURCES})


