//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/[]){
    dll::rbm_desc<
            28 * 28, 10*10,
            dll::momentum,
            //dll::weight_decay<dll::decay_type::L1>,
            //dll::sparsity,
            dll::trainer<dll::pcd1_trainer_t>,
            //dll::init_weights,
            dll::batch_size<50>,
            dll::visible<dll::unit_type::GAUSSIAN>,
            dll::watcher<dll::opencv_rbm_visualizer>>::rbm_t rbm;

    //rbm.momentum = 0.9;
    //rbm.sparsity_target = 0.1;
    //rbm.sparsity_cost = 0.9;
    rbm.learning_rate /= 10.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    dataset.training_images.resize(500);
    dataset.training_labels.resize(500);

    mnist::normalize_dataset(dataset);

    rbm.train(dataset.training_images, 500);

    return 0;
}