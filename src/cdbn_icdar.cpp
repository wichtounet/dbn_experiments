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

template<typename Images, typename Labels>
void extract(std::vector<std::vector<float>>& windows, std::vector<std::size_t>& labels, Images& d_images, Labels& d_labels){
    for(std::size_t image_id = 0; image_id < d_images.size(); ++image_id){
        auto& image = d_images[image_id];

        windows.reserve(windows.size() + image.width * image.height);

        for(std::size_t i = context; i < image.width - context; ++i){
            for(std::size_t j = context; j < image.height - context; ++j){

                windows.emplace_back(window * window);

                labels.push_back(is_text(d_labels[image_id], i, j) ? 1 : 0);

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
}

template<typename Container>
std::size_t count_one(const Container& labels){
    return std::count(labels.begin(), labels.end(), 1);
}

int main(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 5, 1);

    if(dataset.training_labels.empty() || dataset.training_images.empty()){
        std::cout << "Problem while reading the dataset" << std::endl;

        return -1;
    }

    std::random_device rd;
    std::mt19937_64 g(28);

    std::cout << window << "x" << window << " window dimension" << std::endl;

    std::cout << "Dataset" << std::endl;
    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << dataset.test_images.size() << " test images" << std::endl;

    std::vector<std::vector<float>> training_windows;
    std::vector<std::size_t> training_labels;

    std::vector<std::vector<float>> test_windows;
    std::vector<std::size_t> test_labels;

    extract(training_windows, training_labels, dataset.training_images, dataset.training_labels);

    cpp::parallel_shuffle(training_windows.begin(), training_windows.end(), training_labels.begin(), training_labels.end(), g);

    auto total_training = training_windows.size();
    training_windows.resize(10000);
    training_labels.resize(10000);

    extract(test_windows, test_labels, dataset.test_images, dataset.test_labels);

    cpp::parallel_shuffle(test_windows.begin(), test_windows.end(), test_labels.begin(), test_labels.end(), g);

    auto total_test = test_windows.size();
    test_windows.resize(5000);
    test_labels.resize(5000);

    //Normalize everything for Gaussian visible units
    cpp::normalize_each(training_windows);
    cpp::normalize_each(test_windows);

    std::cout << "Extraction" << std::endl;
    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << training_windows.size() << "(" << total_training << ") training windows and labels extracted" << std::endl;
    std::cout << count_one(training_labels) << " text window pixels" << std::endl;

    std::cout << dataset.test_images.size() << " test images" << std::endl;
    std::cout << test_windows.size() << "(" << total_test << ") test windows and labels extracted" << std::endl;
    std::cout << count_one(test_labels) << " text window pixels" << std::endl;

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<window, 1, 5, 40
                , dll::momentum
                , dll::batch_size<50>
                , dll::weight_decay<dll::decay_type::L2>
                , dll::visible<dll::unit_type::GAUSSIAN>
                , dll::sparsity<dll::sparsity_method::LEE>
            >::rbm_t
            ,
            dll::conv_rbm_desc<5, 40, 3, 40
                , dll::momentum
                , dll::batch_size<50>
                , dll::weight_decay<dll::decay_type::L2>
                , dll::visible<dll::unit_type::GAUSSIAN>
                , dll::sparsity<dll::sparsity_method::LEE>
            >::rbm_t
            >>::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    std::cout << "DBN is " << sizeof(dbn_t) << " bytes long" << std::endl;
    std::cout << "DBN input is " << dbn->input_size() << std::endl;
    std::cout << "DBN output is " << dbn->output_size() << std::endl;

    dbn->layer<0>().pbias = 0.07;
    dbn->layer<0>().pbias_lambda = 100;

    dbn->layer<1>().pbias = 0.07;
    dbn->layer<1>().pbias_lambda = 100;

    //TODO What about randomization ?

    dbn->pretrain(training_windows, 50);
    dbn->svm_train(training_windows, training_labels);

    double training_error = dll::test_set(dbn, training_windows, training_labels, dll::svm_predictor());
    std::cout << "Pixel error (training):" << training_error << std::endl;

    double test_error = dll::test_set(dbn, test_windows, test_labels, dll::svm_predictor());
    std::cout << "Pixel error (test):" << test_error << std::endl;

    return 0;
}
