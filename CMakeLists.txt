cmake_minimum_required(VERSION 3.12)
project(TESLA_Quoridor_AI)

# pybind11 패키지 찾기
set(pybind11_DIR "C:/Users/yujoo/Desktop/TESLA/TESLA_Quoridor_AI/.venv/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 파이썬 모듈 생성
pybind11_add_module(TESTfunc TESLA_Quoridor_AI.cpp)
pybind11_add_module(example TESLA_Quoridor_AI.cpp)
pybind11_add_module(QuoridorGame TESLA_Quoridor_AI.cpp)