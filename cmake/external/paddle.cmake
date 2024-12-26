include(ExternalProject)

ExternalProject_Add(
  paddle_project
  PREFIX ${CMAKE_BINARY_DIR}/third_party/paddle
  URL https://paddle2onnx.bj.bcebos.com/paddle_windows/paddle-win_amd64.zip
  DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/third_party/paddle/downloads
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third_party/paddle/libpaddle
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  PATCH_COMMAND
    ${CMAKE_COMMAND} -E echo "Patching files" && ${CMAKE_COMMAND} -E
    copy_if_different ${CMAKE_SOURCE_DIR}/patch/paddle/interface_support.h
    ${CMAKE_BINARY_DIR}/third_party/paddle/libpaddle/include/paddle/pir/include/core/interface_support.h
)
set(LIBPADDLE_INCLUDE_DIR
    "${CMAKE_BINARY_DIR}/third_party/paddle/libpaddle/include")
set(LIBPADDLE_BASE_LIBRARY
    "${CMAKE_BINARY_DIR}/third_party/paddle/libpaddle/libpaddle.lib")
set(LIBPADDLE_COMMON_LIBRARY
    "${CMAKE_BINARY_DIR}/third_party/paddle/libpaddle/common.lib")

add_library(libpaddle_base SHARED IMPORTED)
set_target_properties(libpaddle_base PROPERTIES IMPORTED_LOCATION
                                                ${LIBPADDLE_BASE_LIBRARY})
add_library(libpaddle_common SHARED IMPORTED)
set_target_properties(libpaddle_common PROPERTIES IMPORTED_LOCATION
                                                  ${LIBPADDLE_COMMON_LIBRARY})
# Ensure the download is complete before building the project
add_dependencies(paddle2onnx paddle_project)
# if(BUILD_PADDLE2ONNX_PYTHON) add_dependencies(paddle2onnx_cpp2py_export
# paddle_project) endif()
