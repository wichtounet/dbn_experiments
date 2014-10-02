//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/conv_rbm_mp.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/[]){
    dll::conv_rbm_mp_desc<
            28, 12, 40, 2,
            dll::momentum,
            //dll::weight_decay<dll::decay_type::L2>,
            //dll::sparsity<dll::sparsity_method::LOCAL_TARGET>,
            //dll::trainer<dll::pcd1_trainer_t>,
            dll::batch_size<25>,
            //dll::visible<dll::unit_type::GAUSSIAN>,
            dll::watcher<dll::opencv_rbm_visualizer>>::rbm_t rbm;

    //rbm.momentum = 0.9;
    //rbm.sparsity_target = 0.08;
    //rbm.sparsity_cost = 0.9;
    //rbm.learning_rate = 0.05;
    //rbm.learning_rate /= 10.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    //mnist::normalize_dataset(dataset);
    mnist::binarize_dataset(dataset);

    rbm.train(dataset.training_images, 500, dll::init_watcher);

    return 0;
}
