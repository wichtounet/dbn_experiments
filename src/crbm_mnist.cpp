//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "etl/print.hpp"

#include "dll/conv_rbm_mp.hpp"
#include "dll/conv_mp_layer.hpp"

#include "dll/vector.hpp"

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

    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::visible<dll::unit_type::BINARY>
        >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    dataset.training_images.resize(100);

    binarize_each(dataset.training_images);
    binarize_each(dataset.test_images);

    if(load){
        std::ifstream is("crbm-1.dat", std::ofstream::binary);
        rbm.load(is);
    } else if(train) {
        rbm.learning_rate = 0.001;
        rbm.train(dataset.training_images, 100);

        std::ofstream os("crbm-1.dat", std::ofstream::binary);
        rbm.store(os);
    }

    if(reconstruction){
        std::cout << "Start reconstructions of training images" << std::endl;

        for(size_t t = 0; t < 5; ++t){
            auto& image = dataset.training_images[6 + t];

            std::cout << "Source image" << std::endl;
            for(size_t i = 0; i < 28; ++i){
                for(size_t j = 0; j < 28; ++j){
                    std::cout << static_cast<size_t>(image[i * 28 + j]) << " ";
                }
                std::cout << std::endl;
            }

            rbm.reconstruct(image);
            rbm.display_visible_unit_samples();
        }

        std::cout << "Start reconstructions of test images" << std::endl;

        for(size_t t = 0; t < 5; ++t){
            auto& image = dataset.test_images[6 + t];

            std::cout << "Source image" << std::endl;
            for(size_t i = 0; i < 28; ++i){
                for(size_t j = 0; j < 28; ++j){
                    std::cout << static_cast<size_t>(image[i * 28 + j]) << " ";
                }
                std::cout << std::endl;
            }

            rbm.reconstruct(image);
            rbm.display_visible_unit_samples();
        }
    }

    return 0;
}
