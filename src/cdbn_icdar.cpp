//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#define DLL_PARALLEL
#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/conv_dbn.hpp"
#include "dll/test.hpp"
#include "dll/cpp_utils/algorithm.hpp"
#include "dll/cpp_utils/data.hpp"

#include "icdar/icdar_reader.hpp"

constexpr const std::size_t context = 5;
constexpr const std::size_t window = context * 2 + 1;

template<typename Label>
bool is_text(const Label& label, std::size_t x, std::size_t y){
    for(auto& rectangle : label.rectangles){
        if(x >= rectangle.left && x <= rectangle.right && y >= rectangle.top && y <= rectangle.bottom){
            return true;
        }
    }

    return false;
}

int main(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 5, 1);

    if(dataset.training_labels.empty() || dataset.training_images.empty()){
        std::cout << "Problem while reading the dataset" << std::endl;

        return -1;
    }

    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << dataset.test_images.size() << " test images" << std::endl;

    //TODO What about randomization ?

    std::vector<std::vector<float>> windows;
    std::vector<std::size_t> labels;

    for(std::size_t image_id = 0; image_id < dataset.training_images.size(); ++image_id){
        auto& image = dataset.training_images[image_id];

        windows.reserve(windows.size() + image.width * image.height);

        for(std::size_t i = context; i < image.width - context; ++i){
            for(std::size_t j = context; j < image.height - context; ++j){

                windows.emplace_back(window * window);

                labels.push_back(is_text(dataset.training_labels[image_id], i, j) ? 1 : 0);

                for(std::size_t a = i - context; a < i - context + window; ++a){
                    for(std::size_t b = j - context; b < j - context + window; ++b){
                        auto w_i = (a - (i - context));
                        auto w_j = (b - (j - context));
                        windows.back().at(w_i * window + w_j) = image.pixels.at(a * image.height + b).r;
                    }
                }
            }
        }
    }

    cpp::normalize_each(windows);

    std::cout << windows.size() << " windows and labels extracted" << std::endl;

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<window, 1, 5, 40,
                dll::momentum, dll::batch_size<50>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::visible<dll::unit_type::GAUSSIAN>,
                dll::sparsity<dll::sparsity_method::LEE>>::rbm_t
            >>::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->layer<0>().pbias = 0.05;
    dbn->layer<0>().pbias_lambda = 50;

    dbn->pretrain(windows, 50);
    dbn->svm_train(windows, labels);

    std::cout << "Pixel Accuracy:" << dll::test_set(dbn, windows, labels, dll::svm_predictor()) << std::endl;

    return 0;
}
