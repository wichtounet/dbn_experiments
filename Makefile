default: release

.PHONY: default release debug all clean

CPP_FILES=$(wildcard src/*.cpp)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

DEBUG_O_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.o)
RELEASE_O_FILES=$(CPP_FILES:%.cpp=release/%.cpp.o)

NONEXEC_CPP_FILES := $(filter-out src/rbm_mnist.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/rbm_mnist_view.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/crbm_mnist_view.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/dbn_mnist_view.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/dbn_mnist.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/dbn_mnist_gray.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/conv_dbn_mnist.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/crbm_mnist.cpp,$(NONEXEC_CPP_FILES))

NON_EXEC_DEBUG_O_FILES=$(NONEXEC_CPP_FILES:%.cpp=debug/%.cpp.o)
NON_EXEC_RELEASE_O_FILES=$(NONEXEC_CPP_FILES:%.cpp=release/%.cpp.o)

CXX=clang++
LD=clang++

WARNING_FLAGS=-Wextra -Wall -Qunused-arguments -Wuninitialized -Wsometimes-uninitialized -Wno-long-long -Winit-self -Wdocumentation
CXX_FLAGS=-Idll/include -Idll/etl/include -Imnist/include -Iinclude -std=c++1y -stdlib=libc++ $(WARNING_FLAGS)
LD_FLAGS=$(CXX_FLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

DEBUG_FLAGS=-g
RELEASE_FLAGS=-g -DNDEBUG -Ofast -march=native -fvectorize -fslp-vectorize-aggressive -fomit-frame-pointer

debug/src/%.cpp.o: src/%.cpp
	@ mkdir -p debug/src/
	$(CXX) $(CXX_FLAGS) $(DEBUG_FLAGS) -o $@ -c $<

release/src/%.cpp.o: src/%.cpp
	@ mkdir -p release/src/
	$(CXX) $(CXX_FLAGS) $(RELEASE_FLAGS) -o $@ -c $<

debug/bin/%: debug/src/%.cpp.o $(NON_EXEC_DEBUG_O_FILES)
	@ mkdir -p debug/bin/
	$(LD) $(LD_FLAGS) $(DEBUG_FLAGS) -o $@ $+

release/bin/%: release/src/%.cpp.o $(NON_EXEC_RELEASE_O_FILES)
	@ mkdir -p release/bin/
	$(LD) $(LD_FLAGS) $(RELEASE_FLAGS) -o $@ $+

debug/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p debug/src/
	@ $(CXX) $(CXX_FLAGS) $(DEBUG_FLAGS) -MM -MT debug/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p release/src/
	@ $(CXX) $(CXX_FLAGS) $(RELEASE_FLAGS) -MM -MT release/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release: release/bin/rbm_mnist release/bin/rbm_mnist_view release/bin/crbm_mnist_view release/bin/dbn_mnist release/bin/dbn_mnist_gray release/bin/crbm_mnist release/bin/conv_dbn_mnist release/bin/dbn_mnist_view
debug: debug/bin/rbm_mnist debug/bin/rbm_mnist_view debug/bin/crbm_mnist_view debug/bin/dbn_mnist debug/bin/dbn_mnist_gray debug/bin/crbm_mnist debug/bin/conv_dbn_mnist debug/bin/dbn_mnist_view

all: release debug

sonar: release
	cppcheck --xml-version=2 --enable=all --std=c++11 src 2> cppcheck_report.xml
	/opt/sonar-runner/bin/sonar-runner

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)