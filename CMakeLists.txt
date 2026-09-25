cmake_minimum_required(VERSION 3.12)
project(TESLA_Quoridor_AI)

set(Python_EXECUTABLE
    "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe"
)
set(pybind11_DIR
    "${CMAKE_SOURCE_DIR}/.venv/Lib/site-packages/pybind11/share/cmake/pybind11"
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(Quoridor AlphaZero_Quoridor_Engine.cpp)