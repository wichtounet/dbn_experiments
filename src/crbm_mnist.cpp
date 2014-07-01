//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/conv_rbm.hpp"
#include "dll/conv_layer.hpp"
#include "dll/vector.hpp"
#include "dll/generic_trainer.hpp"

#include "mnist/mnist_reader.hpp"

#include "utils.hpp"

int main(int argc, char* argv[]){
    auto reconstruction = false;
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "sample"){
            reconstruction = true;
        }

        if(command == "load"){
            load = true;
        }
    }

    dll::conv_rbm<dll::conv_layer<
            28, 12, 40
            >> rbm;

    auto training_images = mnist::read_training_images<std::vector, vector, double>();

    binarize_each(training_images);

    if(load){
        std::ifstream is("rbm-1.dat", std::ofstream::binary);
        //TODO rbm.load(is);
    } else {
        rbm.train(training_images, 1);

        std::ofstream os("rbm-1.dat", std::ofstream::binary);
        //TODO rbm.store(os);
    }

    if(reconstruction){
        auto test_images = mnist::read_test_images<std::vector, vector, double>();
        binarize_each(test_images);

        for(size_t t = 0; t < 10; ++t){
            auto& image = test_images[666 + t];

            std::cout << "Source image" << std::endl;
            for(size_t i = 0; i < 28; ++i){
                for(size_t j = 0; j < 28; ++j){
                    std::cout << static_cast<size_t>(image[i * 28 + j]) << " ";
                }
                std::cout << std::endl;
            }

            //rbm.reconstruct(image);

            std::cout << "Reconstructed image" << std::endl;
            //TODO rbm.display_visible_units(28);
        }
    }

    return 0;
}
