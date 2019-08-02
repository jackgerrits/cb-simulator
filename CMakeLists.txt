cmake_minimum_required(VERSION 3.5)

project(cb_sim C CXX)
set(CMAKE_CXX_STANDARD 11)

set(VW_INSTALL OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_JAVA OFF CACHE BOOL "")
set(BUILD_PYTHON OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(WARNINGS OFF CACHE BOOL "")

add_executable(cb_sim main.cpp)

# Use the following two lines when using commits where CMake is present
add_subdirectory(vowpal_wabbit EXCLUDE_FROM_ALL)
target_link_libraries(cb_sim PRIVATE vw)

# Use the following lines when CMake is not present. You need to cd into VW and build it first (make vw)
# find_package(Boost REQUIRED COMPONENTS program_options system thread unit_test_framework)
# find_package(ZLIB REQUIRED)
# target_include_directories(cb_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vowpal_wabbit/vowpalwabbit ${CMAKE_CURRENT_SOURCE_DIR}/vowpal_wabbit/explore ${CMAKE_CURRENT_SOURCE_DIR}/vowpal_wabbit/rapidjson/include)
# target_link_libraries(cb_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vowpal_wabbit/vowpalwabbit/libvw.a ${CMAKE_CURRENT_SOURCE_DIR}/vowpal_wabbit/vowpalwabbit/liballreduce.a Boost::program_options Threads::Threads ZLIB::ZLIB  ${CMAKE_DL_LIBS})
