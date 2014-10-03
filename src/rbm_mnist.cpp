//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

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

    dll::rbm_desc<
            28 * 28, 200,
            dll::momentum,
            dll::batch_size<25>,
            dll::hidden<dll::unit_type::RELU>>::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    mnist::binarize_dataset(dataset);

    if(load){
        std::ifstream is("rbm-1.dat", std::ofstream::binary);
        rbm.load(is);
    } else {
        rbm.train(dataset.training_images, 10);

        std::ofstream os("rbm-1.dat", std::ofstream::binary);
        rbm.store(os);
    }

    if(reconstruction){
        for(size_t t = 0; t < 10; ++t){
            auto& image = dataset.test_images[6 + t];

            std::cout << "Source image" << std::endl;
            for(size_t i = 0; i < 28; ++i){
                for(size_t j = 0; j < 28; ++j){
                    std::cout << static_cast<size_t>(image[i * 28 + j]) << " ";
                }
                std::cout << std::endl;
            }

            rbm.reconstruct(image);

            std::cout << "Reconstructed image" << std::endl;
            rbm.display_visible_units(28);
        }
    }

    return 0;
}
