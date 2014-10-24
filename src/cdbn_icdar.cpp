//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "icdar/icdar_reader.hpp"

constexpr const std::size_t context = 5;
constexpr const std::size_t window = context * 2 + 1;

int main(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 1, 1);

    if(dataset.training_labels.empty() || dataset.training_images.empty()){
        std::cout << "Problem while reading the dataset" << std::endl;

        return -1;
    }

    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << dataset.test_images.size() << " test images" << std::endl;

    for(auto& image : dataset.training_images){
        std::vector<std::vector<uint8_t>> windows;

        std::cout << image.width << "x" << image.height << std::endl;

        for(std::size_t i = context; i < image.width - context; ++i){
            for(std::size_t j = context; j < image.height - context; ++j){

                windows.emplace_back(window * window);

                //std::cout << i << ":" << j << std::endl;

                for(std::size_t a = i - context; a < i - context + window; ++a){
                    for(std::size_t b = j - context; b <= j - context + window; ++b){
                        //std::cout << a << "<->" << b << std::endl;
                        //std::cout << ((a * image.height) + b) << std::endl;
                        //std::cout << ((i -context + a) * window + (j - context + b)) << std::endl;
                        windows.back().at((i - context + a) * window + (j - context + b)) = 1;//image.pixels.at(a * image.height + b).r;
                    }
                }

            }
        }

    }

    return 0;
}
