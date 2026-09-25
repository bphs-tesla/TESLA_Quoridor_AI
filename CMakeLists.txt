cmake_minimum_required(VERSION 3.18)

project(TESLA_Quoridor_AI LANGUAGES CXX)

# ============================================================
# C++ 설정
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ============================================================
# Python / venv 경로 설정
#
# Windows:
#   프로젝트/.venv/Scripts/python.exe
#
# WSL/Linux:
#   /home/yujoon/TESLA_Quoridor_AI/.venv/bin/python
# ============================================================

if(WIN32)

    set(VENV_PYTHON
        "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe"
    )

elseif(UNIX)

    set(VENV_PYTHON
        "/home/yujoon/TESLA_Quoridor_AI/.venv/bin/python"
    )

endif()


# ============================================================
# Python 존재 확인
# ============================================================

if(NOT EXISTS "${VENV_PYTHON}")

    message(FATAL_ERROR
        "Python virtual environment를 찾을 수 없습니다:\n"
        "  ${VENV_PYTHON}\n"
    )

endif()


message(STATUS "==========================================")
message(STATUS "Python executable:")
message(STATUS "  ${VENV_PYTHON}")
message(STATUS "==========================================")


# ============================================================
# CMake Python 변수 지정
# ============================================================

set(Python_EXECUTABLE
    "${VENV_PYTHON}"
    CACHE FILEPATH
    "Python executable"
    FORCE
)

set(Python3_EXECUTABLE
    "${VENV_PYTHON}"
    CACHE FILEPATH
    "Python 3 executable"
    FORCE
)


# ============================================================
# Python
# ============================================================

find_package(
    Python
    REQUIRED
    COMPONENTS
        Interpreter
        Development
)


message(STATUS "Python version:")
message(STATUS "  ${Python_VERSION}")

message(STATUS "Python include:")
message(STATUS "  ${Python_INCLUDE_DIRS}")

message(STATUS "Python libraries:")
message(STATUS "  ${Python_LIBRARIES}")


# ============================================================
# pybind11
#
# 현재 사용하는 Python 환경에서 직접 cmake 경로를 가져온다.
#
# Windows:
#   .venv/Lib/site-packages/pybind11/...
#
# WSL:
#   /home/yujoon/.../.venv/lib/python3.12/site-packages/pybind11/...
#
# 따라서 OS별 경로를 직접 하드코딩하지 않는다.
# ============================================================

execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -m pybind11
        --cmakedir

    OUTPUT_VARIABLE
        PYBIND11_CMAKE_DIR

    OUTPUT_STRIP_TRAILING_WHITESPACE

    RESULT_VARIABLE
        PYBIND11_RESULT
)


if(NOT PYBIND11_RESULT EQUAL 0)

    message(FATAL_ERROR
        "현재 Python 환경에 pybind11이 설치되어 있지 않습니다.\n"
        "Python:\n"
        "  ${Python_EXECUTABLE}\n\n"
        "설치:\n"
        "  python -m pip install pybind11\n"
    )

endif()


message(STATUS "pybind11 CMake directory:")
message(STATUS "  ${PYBIND11_CMAKE_DIR}")


set(
    pybind11_DIR
    "${PYBIND11_CMAKE_DIR}"
    CACHE PATH
    "pybind11 CMake directory"
    FORCE
)


# ============================================================
# pybind11
# ============================================================

find_package(
    pybind11
    CONFIG
    REQUIRED
)


# ============================================================
# OpenMP
# ============================================================

find_package(OpenMP REQUIRED)


# ============================================================
# Quoridor Python module
# ============================================================

if(MSVC)
    add_compile_options(/utf-8)
endif()

pybind11_add_module(
    Quoridor
    AlphaZero_Quoridor_Engine.cpp
    Quoridor_MCTS_Engine.cpp
)
#    TESLA_Quoridor_AI.cpp

# ============================================================
# OpenMP 연결
# ============================================================

target_link_libraries(
    Quoridor
    PRIVATE
        OpenMP::OpenMP_CXX
)


# ============================================================
# 출력 정보
# ============================================================

message(STATUS "==========================================")
message(STATUS "TESLA_Quoridor_AI configuration complete")
message(STATUS "Target:")
message(STATUS "  Quoridor")
message(STATUS "C++:")
message(STATUS "  C++${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP:")
message(STATUS "  ${OpenMP_CXX_VERSION}")
message(STATUS "==========================================")