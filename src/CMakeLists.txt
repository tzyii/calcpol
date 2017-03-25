include_directories(${PROJECT_SOURCE_DIR}/include)

set(LIB_SRC calc.c cluster.c efp.c elec.c utils.c)
set(APP_SRC main.c)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(libcalcpol ${LIB_SRC})
add_executable(calcpol ${APP_SRC})

set_target_properties(libcalcpol PROPERTIES OUTPUT_NAME "calcpol")

target_link_libraries(calcpol libcalcpol m)
