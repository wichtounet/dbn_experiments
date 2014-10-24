default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idll/include -Idll/nice_svm/include -Idll/etl/include -Imnist/include -std=c++1y -stdlib=libc++
LD_FLAGS  += -lsvm -lopencv_core -lopencv_imgproc -lopencv_highgui

$(eval $(call auto_folder_compile,src))

$(eval $(call add_src_executable,rbm_mnist,rbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist,crbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist_view,crbm_mnist_view.cpp))
$(eval $(call add_src_executable,dbn_mnist,dbn_mnist.cpp))
$(eval $(call add_src_executable,dbn_mnist_view,dbn_mnist_view.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist,conv_dbn_mnist.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist_view,conv_dbn_mnist_view.cpp))

release: release/bin/rbm_mnist release/bin/crbm_mnist_view release/bin/dbn_mnist release/bin/crbm_mnist release/bin/conv_dbn_mnist release/bin/conv_dbn_mnist_view
debug: debug/bin/rbm_mnist debug/bin/crbm_mnist_view debug/bin/dbn_mnist debug/bin/crbm_mnist debug/bin/conv_dbn_mnist debug/bin/dbn_mnist_view

all: release debug

sonar: release
	cppcheck --xml-version=2 --enable=all --std=c++11 src 2> cppcheck_report.xml
	/opt/sonar-runner/bin/sonar-runner

clean: base_clean

include make-utils/cpp-utils-finalize.mk
