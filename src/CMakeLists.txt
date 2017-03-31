include_directories(${PROJECT_SOURCE_DIR}/include)

set(LIB_SRC calc.c cluster.c efp.c elec.c utils.c)
set(IE_SRC calcie.c)
set(RE_SRC calcre.c)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(libcoulomb ${LIB_SRC})
add_executable(calcie ${IE_SRC})
add_executable(calcre ${RE_SRC})

set_target_properties(libcoulomb PROPERTIES OUTPUT_NAME "coulomb")

target_link_libraries(calcie libcoulomb m)
target_link_libraries(calcre libcoulomb m)
