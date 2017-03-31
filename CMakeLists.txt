cmake_minimum_required(VERSION 3.4)

project(calcCoulomb)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3 -Wall -Wno-unused-variable -fopenmp")

add_subdirectory(src)

