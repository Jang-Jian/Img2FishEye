CFLAGS = -fPIC -O3 -funroll-loops -march=native 
OMP4CXX = -fopenmp -Wall
CPPSTL = -std=c++11
SHARED = -shared

# get python version.
PY3_VERSION = $(shell python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
ifeq ($(PY3_VERSION),$(filter $(PY3_VERSION), 3.5 3.6 3.7))
PY3_VERSION := $(PY3_VERSION)m
endif
PY_INCLUDE = -I /usr/include/python$(PY3_VERSION)
PY_LIB = -lpython$(PY3_VERSION)
PY_BOOST = -lboost_python3
PYTHON = $(PY_INCLUDE) $(PY_LIB) $(PY_BOOST)

CXX = g++ $(CPPSTL) $(CFLAGS)


OBJ_SRC = ./obj/
PY_LIB_SRC = ./python/


PY_SHARED = $(PY_LIB_SRC)integration.so
CPU_OBJs += $(OBJ_SRC)ndproc.o $(OBJ_SRC)tensor.o $(OBJ_SRC)integration.o \
			$(OBJ_SRC)lenproc.o


all: obj $(PY_SHARED) 

obj:
	mkdir -p $(OBJ_SRC)

clean:
	rm -rf $(OBJ_SRC) $(PY_SHARED)


$(PY_SHARED): $(CPU_OBJs)
	$(CXX) $(OMP4CXX) -shared $(CPU_OBJs) $(PYTHON) -o $(PY_SHARED)


$(OBJ_SRC)%.o: cpp/python/%.cpp
	$(CXX) $(OMP4CXX) $(PYTHON) -c $< -o $@
$(OBJ_SRC)%.o: cpp/imgproc/%.cpp
	$(CXX) $(OMP4CXX)  -c $< -o $@
$(OBJ_SRC)%.o: cpp/container/%.cpp
	$(CXX) $(OMP4CXX) $(PYTHON)  -c $< -o $@