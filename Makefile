default: release_debug

.PHONY: default release release_debug debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idll/etl/lib/include -Idll/etl/include -Idll/include -Idll/nice_svm/include -Imnist/include #-Iicdar/include
LD_FLAGS  += -lsvm -lopencv_core -lopencv_imgproc -lopencv_highgui -ljpeg -lpthread

$(eval $(call auto_folder_compile,src))

$(eval $(call add_src_executable,rbm_mnist,rbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist,crbm_mnist.cpp))
$(eval $(call add_src_executable,crbm_mnist_view,crbm_mnist_view.cpp))
$(eval $(call add_src_executable,dbn_mnist,dbn_mnist.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist,conv_dbn_mnist.cpp))
$(eval $(call add_src_executable,conv_dbn_mnist_view,conv_dbn_mnist_view.cpp))
#$(eval $(call add_src_executable,cdbn_icdar,cdbn_icdar.cpp))
#$(eval $(call add_src_executable,cdbn_icdar_2,cdbn_icdar_2.cpp))

release_debug: release_debug/bin/rbm_mnist release_debug/bin/crbm_mnist_view release_debug/bin/dbn_mnist release_debug/bin/crbm_mnist release_debug/bin/conv_dbn_mnist #release_debug/bin/cdbn_icdar release_debug/bin/cdbn_icdar_2
release: release/bin/rbm_mnist release/bin/crbm_mnist_view release/bin/dbn_mnist release/bin/crbm_mnist release/bin/conv_dbn_mnist #release/bin/cdbn_icdar release/bin/cdbn_icdar_2
debug: debug/bin/rbm_mnist debug/bin/crbm_mnist_view debug/bin/dbn_mnist debug/bin/crbm_mnist debug/bin/conv_dbn_mnist #debug/bin/cdbn_icdar debug/bin/cdbn_icdar_2

all: release release_debug debug

clean: base_clean

include make-utils/cpp-utils-finalize.mk
