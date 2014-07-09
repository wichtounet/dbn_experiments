//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "etl/print.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/conv_layer.hpp"
//#include "dll/conv_rbm_mp.hpp"
//#include "dll/conv_mp_layer.hpp"

#include "dll/vector.hpp"
#include "dll/generic_trainer.hpp"

#include "mnist/mnist_reader.hpp"

#include "utils.hpp"

int main(int argc, char* argv[]){
    auto reconstruction = false;
    auto load = false;
    auto train = true;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "sample"){
            reconstruction = true;
        }

        if(command == "init"){
            train = false;
        }

        if(command == "load"){
            load = true;
            train = false;
        }
    }

    dll::conv_rbm<dll::conv_layer<
            28, 12, 8,
            dll::batch_size<25>,
            //dll::weight_decay<dll::decay_type::L1>,
            dll::visible<dll::unit_type::BINARY>
            >> rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    binarize_each(dataset.training_images);
    binarize_each(dataset.test_images);
    //normalize(dataset.training_images);
    //normalize(dataset.test_images);

    if(load){
        std::ifstream is("crbm-1.dat", std::ofstream::binary);
        rbm.load(is);
    } else if(train) {
        rbm.learning_rate = 0.1;
        rbm.train(dataset.training_images, 10);

        std::ofstream os("crbm-1.dat", std::ofstream::binary);
        rbm.store(os);
    }

    if(reconstruction){
        //std::cout << "W:" << sum(rbm.w) << std::endl;
        std::cout << "b:" << rbm.b << std::endl;
        std::cout << "c:" << rbm.c << std::endl;

        std::cout << "Start reconstructions" << std::endl;

        for(size_t t = 0; t < 10; ++t){
            auto& image = dataset.training_images[666 + t];

            //std::cout << "Source image" << std::endl;
            //for(size_t i = 0; i < 28; ++i){
                //for(size_t j = 0; j < 28; ++j){
                    //std::cout << static_cast<size_t>(image[i * 28 + j]) << " ";
                //}
                //std::cout << std::endl;
            //}

            //std::cout << "before:" << sum(rbm.v2_a) << std::endl;
            //std::cout << "before:" << sum(sum(rbm.h2_a)) << std::endl;

            rbm.reconstruct(image);

            //std::cout << "after:" << sum(rbm.v2_a) << std::endl;
            //std::cout << "after:" << sum(sum(rbm.h2_a)) << std::endl;
            //std::cout << "h1_s after:" << sum(rbm.h1_s) << std::endl;
            //std::cout << "h2_s after:" << sum(rbm.h2_s) << std::endl;

            //std::cout << "Reconstructed image" << std::endl;
            rbm.display_visible_unit_samples();
        }
    }

    return 0;
}
