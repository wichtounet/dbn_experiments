//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dll/conv_rbm.hpp"
#include "dll/conv_dbn.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int argc, char* argv[]){
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
            load = true;
        }
    }

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(5000);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    mnist::binarize_dataset(dataset);

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<28, 1, 17, 40, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            dll::conv_rbm_desc<17, 40, 12, 40, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>>::rbm_t
        >, dll::watcher<dll::opencv_dbn_visualizer>>::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    if(load){
        std::cout << "Load from file" << std::endl;

        std::ifstream is("dbn.dat", std::ifstream::binary);
        dbn->load(is);
    } else {
        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(dataset.training_images, 5);

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);
    }

    return 0;
}
