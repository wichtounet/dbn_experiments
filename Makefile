default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

$(eval $(call use_libcxx))

CXX_FLAGS += -Idll/include -Idll/nice_svm/include -Idll/etl/include -Iicdar/include -Imnist/include
LD_FLAGS  += -lsvm -lopencv_core -lopencv_imgproc -lopencv_highgui -ljpeg -lpthread

$(eval $(call auto_folder_compile,src))

$(eval $(call add_src_executable,rbm_mnist,rbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist,crbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist_view,crbm_mnist_view.cpp))
$(eval $(call add_src_executable,dbn_mnist,dbn_mnist.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist,conv_dbn_mnist.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist_view,conv_dbn_mnist_view.cpp))
$(eval $(call add_src_executable,cdbn_icdar,cdbn_icdar.cpp))
$(eval $(call add_src_executable,cdbn_icdar_2,cdbn_icdar_2.cpp))

release: release/bin/rbm_mnist release/bin/crbm_mnist_view release/bin/dbn_mnist release/bin/crbm_mnist release/bin/conv_dbn_mnist release/bin/cdbn_icdar release/bin/cdbn_icdar_2
debug: debug/bin/rbm_mnist debug/bin/crbm_mnist_view debug/bin/dbn_mnist debug/bin/crbm_mnist debug/bin/conv_dbn_mnist debug/bin/cdbn_icdar debug/bin/cdbn_icdar_2

all: release debug

clean: base_clean

include make-utils/cpp-utils-finalize.mk
