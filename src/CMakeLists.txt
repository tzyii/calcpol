include_directories(${PROJECT_SOURCE_DIR}/include)

set(LIB_SRC calc.c cluster.c efp.c elec.c utils.c)
set(POL_SRC calcpol.c)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(libpol ${LIB_SRC})
add_executable(calcpol ${POL_SRC})

set_target_properties(libpol PROPERTIES OUTPUT_NAME "pol")

target_link_libraries(calcpol libpol m)
