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

#include <opencv2/opencv.hpp>

constexpr const std::size_t deep_context = 5;
constexpr const std::size_t deep_window = deep_context * 2 + 1;

constexpr const std::size_t large_window = 135;
constexpr const std::size_t large_first_border = 8;

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
void deep_extract(std::vector<std::vector<float>>& windows, std::vector<std::size_t>& labels, Images& d_images, Labels& d_labels){
    for(std::size_t image_id = 0; image_id < d_images.size(); ++image_id){
        auto& image = d_images[image_id];

        windows.reserve(windows.size() + image.width * image.height);

        for(std::size_t i = deep_context; i < image.width - deep_context; ++i){
            for(std::size_t j = deep_context; j < image.height - deep_context; ++j){

                windows.emplace_back(deep_window * deep_window);

                labels.push_back(is_text(d_labels[image_id], i, j) ? 1 : 0);

                for(std::size_t a = i - deep_context; a < i - deep_context + deep_window; ++a){
                    for(std::size_t b = j - deep_context; b < j - deep_context + deep_window; ++b){
                        auto w_i = (a - (i - deep_context));
                        auto w_j = (b - (j - deep_context));
                        windows.back().at(w_i * deep_window + w_j) = image.pixels.at(a * image.height + b).r;
                    }
                }
            }
        }
    }
}

template<typename Images>
Images large_pad(Images& d_images){
    Images padded_images;

    for(auto& image : d_images){
        auto width = image.width + 2 * large_first_border;
        auto height = image.height + 2 * large_first_border;

        width = width % large_window > 0 ? (width / large_window + 1) * large_window : width;
        height = height % large_window > 0 ? (height / large_window + 1) * large_window : height;

        padded_images.emplace_back(width, height);

        for(auto& pixel : padded_images.back().pixels){
            pixel.r = pixel.g = pixel.b = 0;
        }

        for(std::size_t row = 0; row < image.height; ++row){
            for(std::size_t col = 0; col < image.width; ++col){
                auto padded_row = row + large_first_border;
                auto padded_col = col + large_first_border;

                padded_images.back().pixels[padded_row * width + padded_col].r = image.pixels[row * image.width + col].r;
                padded_images.back().pixels[padded_row * width + padded_col].g = image.pixels[row * image.width + col].g;
                padded_images.back().pixels[padded_row * width + padded_col].b = image.pixels[row * image.width + col].b;
            }
        }
    }

    return padded_images;
}

template<typename Images>
void large_extract(std::vector<std::vector<float>>& patches, const Images& d_images){
    for(auto& image : d_images){
        patches.reserve(patches.size() + (image.width / large_window) * (image.height / large_window));

        for(std::size_t i = 0; i < image.height; i += large_window){
            for(std::size_t j = 0; j < image.width; j += large_window){
                patches.emplace_back(large_window * large_window);

                for(std::size_t a = i; a < i + large_window; ++a){
                    for(std::size_t b = j; b < j + large_window; ++b){
                        auto w_i = a - i;
                        auto w_j = b - j;
                        patches.back().at(w_i * large_window + w_j) = image.pixels.at(a * image.width + b).r;
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

int deep_wise(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 5, 1);

    if(dataset.training_labels.empty() || dataset.training_images.empty()){
        std::cout << "Problem while reading the dataset" << std::endl;

        return -1;
    }

    std::random_device rd;
    std::mt19937_64 g(28);

    std::cout << "Dataset" << std::endl;
    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << dataset.test_images.size() << " test images" << std::endl;

    std::cout << deep_window << "x" << deep_window << " window dimension" << std::endl;

    std::vector<std::vector<float>> training_windows;
    std::vector<std::size_t> training_labels;

    std::vector<std::vector<float>> test_windows;
    std::vector<std::size_t> test_labels;

    deep_extract(training_windows, training_labels, dataset.training_images, dataset.training_labels);

    cpp::parallel_shuffle(training_windows.begin(), training_windows.end(), training_labels.begin(), training_labels.end(), g);

    auto total_training = training_windows.size();
    training_windows.resize(10000);
    training_labels.resize(10000);

    deep_extract(test_windows, test_labels, dataset.test_images, dataset.test_labels);

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
            dll::conv_rbm_desc<deep_window, 1, 5, 40
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

template<typename Images>
void debug_padded(const Images& images){
    cv::namedWindow("Padded", cv::WINDOW_NORMAL);

    for(auto& image : images){
        cv::Mat buffer_image(cv::Size(image.width, image.height), CV_8UC1);
        //buffer_image = cv::Scalar(255);

        for(std::size_t row = 0; row < image.height; ++row){
            for(std::size_t col = 0; col < image.width; ++col){
                buffer_image.at<uint8_t>(row, col) = 255 * image.pixels[row * image.width + col].r;
            }
        }

        cv::imshow("Padded", buffer_image);
        cv::waitKey(0);
    }
}

template<typename Images>
void debug_patches(const Images& patches){
    cv::namedWindow("Patches", cv::WINDOW_NORMAL);

    for(auto& patch : patches){
        cv::Mat buffer_image(cv::Size(large_window, large_window), CV_8UC1);

        for(std::size_t row = 0; row < large_window; ++row){
            for(std::size_t col = 0; col < large_window; ++col){
                buffer_image.at<uint8_t>(row, col) = 255.0 * patch[row * large_window + col];
            }
        }

        cv::imshow("Patches", buffer_image);
        cv::waitKey(0);
    }
}

int large_wise(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 10, 10);

    if(dataset.training_labels.empty() || dataset.training_images.empty()){
        std::cout << "Problem while reading the dataset" << std::endl;

        return -1;
    }

    std::random_device rd;
    std::mt19937_64 g(28);

    std::cout << "Dataset" << std::endl;
    std::cout << dataset.training_images.size() << " training images" << std::endl;
    std::cout << dataset.test_images.size() << " test images" << std::endl;

    std::cout << large_window << "x" << large_window << " window dimension\n\n";

    auto training_images_padded = large_pad(dataset.training_images);
    auto test_images_padded = large_pad(dataset.test_images);

    std::vector<std::vector<float>> training_patches;
    std::vector<std::vector<float>> test_patches;

    large_extract(training_patches, training_images_padded);
    large_extract(test_patches, test_images_padded);

    //Normalize everything for Gaussian visible units
    cpp::normalize_each(training_patches);
    cpp::normalize_each(test_patches);

    std::cout << "Extraction" << std::endl;
    std::cout << training_images_padded.size() << " training images padded" << std::endl;
    std::cout << training_patches.size() << " training patches" << std::endl;
    std::cout << test_images_padded.size() << " test images padded" << std::endl;
    std::cout << test_patches.size() << " test patches\n\n";

    //debug_padded();
    //debug_patches();

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<large_window, 1, 128, 25
                , dll::momentum
                , dll::batch_size<8>
                , dll::weight_decay<dll::decay_type::L2>
                , dll::visible<dll::unit_type::GAUSSIAN>
                //, dll::sparsity<dll::sparsity_method::LEE>
            >::rbm_t
            >>::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    std::cout << "DBN is " << sizeof(dbn_t) << " bytes long" << std::endl;
    std::cout << "DBN input is " << dbn->input_size() << std::endl;
    std::cout << "DBN output is " << dbn->output_size() << std::endl;

    //dbn->layer<0>().pbias = 0.07;
    //dbn->layer<0>().pbias_lambda = 100;

    dbn->pretrain(training_patches, 5);

    //TODO Classify

    return 0;
}

int main(){
    return large_wise();
}
