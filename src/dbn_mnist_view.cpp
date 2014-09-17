//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dll/dbn.hpp"
#include "dll/dbn_desc.hpp"
#include "dll/dbn_layers.hpp"
#include "dll/test.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/[]){
    auto dataset = mnist::read_dataset<std::vector, etl::dyn_vector, double>(5000);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    mnist::binarize_dataset(dataset);

    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<50>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::EXP>>::rbm_t
    >, dll::watcher<dll::silent_dbn_watcher>>::dbn_t dbn_t;

    auto dbn = make_unique<dbn_t>();

    dbn->display();

    std::cout << "Start pretraining" << std::endl;
    dbn->pretrain(dataset.training_images, 10);

    std::cout << "Start fine-tuning" << std::endl;
    dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 1000);

    std::ofstream os("dbn.dat", std::ofstream::binary);
    dbn->store(os);

    return 0;
}
