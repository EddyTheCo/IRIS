cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(iris_dataset VERSION 0.1 DESCRIPTION "iris torch dataset" LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(iris iris.cpp include/iris.hpp)

target_link_libraries(iris PUBLIC  "${TORCH_LIBRARIES}")

target_include_directories(iris INTERFACE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS iris EXPORT irisTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT irisTargets
	FILE irisTargets.cmake
	DESTINATION lib/cmake/iris
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/irisConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/iris"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/irisConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/irisConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/irisConfigVersion.cmake
	DESTINATION lib/cmake/iris
	)
export(EXPORT irisTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/irisTargets.cmake"
	)
