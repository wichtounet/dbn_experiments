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
#include "dll/ocv_visualizer.hpp"

#include "dll/cpp_utils/algorithm.hpp"
#include "dll/cpp_utils/data.hpp"

#include "icdar/icdar_reader.hpp"

#include <opencv2/opencv.hpp>

constexpr const std::size_t deep_context = 5;
constexpr const std::size_t deep_window = deep_context * 2 + 1;

constexpr const std::size_t large_window = 40;
constexpr const std::size_t large_filter = 8;
constexpr const std::size_t large_features = 40;

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
        auto width = image.width + 2 * large_filter;
        auto height = image.height + 2 * large_filter;

        width = width % large_window > 0 ? (width / large_window + 1) * large_window : width;
        height = height % large_window > 0 ? (height / large_window + 1) * large_window : height;

        padded_images.emplace_back(width, height);

        for(auto& pixel : padded_images.back().pixels){
            pixel.r = pixel.g = pixel.b = 0;
        }

        for(std::size_t row = 0; row < image.height; ++row){
            for(std::size_t col = 0; col < image.width; ++col){
                auto padded_row = row + large_filter;
                auto padded_col = col + large_filter;

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
                patches.emplace_back(large_window * large_window * 3);

                for(std::size_t a = i; a < i + large_window; ++a){
                    for(std::size_t b = j; b < j + large_window; ++b){
                        auto w_i = a - i;
                        auto w_j = b - j;
                        patches.back().at(w_i * large_window + w_j) = image.pixels.at(a * image.width + b).r;
                        patches.back().at(w_i * large_window + w_j + 1) = image.pixels.at(a * image.width + b).g;
                        patches.back().at(w_i * large_window + w_j + 2) = image.pixels.at(a * image.width + b).b;
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

template<typename DBN, typename Labels, typename Images, typename Patches, typename SFeatures, typename SLabels, typename RNG>
void large_svm_extract(DBN& dbn, const Labels& labels, const Images& images, const Images& padded_images, const Patches& patches, SFeatures& svm_features, SLabels& svm_labels, std::size_t limit, RNG&& g){
    std::cout << "Extraction for SVM..." << std::endl;

    //1. Get features from DBN

    std::vector<std::vector<float>> rbm_features;
    rbm_features.reserve(patches.size());

    for(auto& patch : patches){
        rbm_features.emplace_back(DBN::output_size());
        dbn.activation_probabilities(patch, rbm_features.back());
    }

    std::cout << "Features extracted for " << patches.size() << " patches" << std::endl;

    //2. Extract all locations

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> locations;
    locations.reserve(images.size() * 1000000);

    std::size_t prev_patches = 0;
    for(std::size_t i_i = 0; i_i < images.size(); ++i_i){
        auto& image = images[i_i];
        auto& padded_image = padded_images[i_i];

        for(std::size_t y = 0; y < image.height; ++y){
            for(std::size_t x = 0; x < image.width; ++x){
                locations.emplace_back(i_i, x, y, prev_patches);
            }
        }

        prev_patches += padded_image.height / large_window * padded_image.width / large_window;
    }

    std::shuffle(locations.begin(), locations.end(), g);

    std::cout << locations.size() << " locations extracted" << std::endl;

    //3. Get features for SVM

    svm_features.reserve(limit);
    svm_labels.reserve(limit);

    std::size_t count_0 = 0;
    std::size_t count_1 = 0;

    for(auto& location : locations){
        if(svm_features.size() == limit){
            break;
        }

        auto i_i = std::get<0>(location);
        auto x = std::get<1>(location);
        auto y = std::get<2>(location);

        auto& padded_image = padded_images[i_i];
        auto label = is_text(labels[i_i], x, y) ? 1 : 0;

        if(label == 1 && (count_1 < 1.1 * count_0 || count_1 < 100)){
            ++count_1;
        } else if(label == 0 && (count_0 < 1.1 * count_1 || count_0 < 100)){
            ++count_0;
        } else {
            continue;
        }

        //Indexes in padded images
        auto global_y = y + large_filter;
        auto global_x = x + large_filter;

        //Index of the patch
        auto patch_y = global_y / large_window;
        auto patch_x = global_x / large_window;

        //Index inside the patch
        auto local_y = y % large_window;
        auto local_x = x % large_window;

        auto& patch = rbm_features[std::get<3>(location) + (patch_y * (padded_image.width / large_window) + patch_x)];

        svm_features.emplace_back(large_features);

        for(std::size_t i = 0; i < large_features; ++i){
            svm_features.back()[i] = patch[local_y * large_window + local_x + i];
        }

        svm_labels.push_back(label);
    }

    svm_features.shrink_to_fit();
    svm_labels.shrink_to_fit();

    std::cout << "... done" << std::endl;
}

void svm_scale(std::vector<std::vector<float>>& features){
    std::cout << "Scale features" << std::endl;

    //Scale each column

    if(true){
        //Version 1: Scale each feature in [0,1]

        float a = 0.0;
        float b = 1.0;

        for(std::size_t i = 0; i < large_features; ++i){
            float min = 0.0;
            float max = 0.0;
            for(auto& feature : features){
                min = std::min(min, feature[i]);
                max = std::max(max, feature[i]);
            }

            for(auto& feature : features){
                feature[i] = a + ((b - a) * (feature[i] - min)) / (max - min);
            }
        }
    } else {
        //Version 1: Normalize each feature with zero-mean and unit variance

        for(std::size_t i = 0; i < large_features; ++i){
            float mean = 0.0;
            for(auto& feature : features){
                mean += feature[i];
            }
            mean /= features.size();
            for(auto& feature : features){
                feature[i] -= mean;
            }
            double stddev = 0.0;
            for(auto& feature : features){
                stddev += feature[i] * feature[i];
            }
            stddev = std::sqrt(stddev / features.size());
            for(auto& feature : features){
                feature[i] /= stddev;
            }
        }
    }
}

int large_wise(){
    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test", 4, 2);

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
    std::cout << test_images_padded.size() << " test images padded" << std::endl; std::cout << test_patches.size() << " test patches\n\n";

    //debug_padded();
    //debug_patches();

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<large_window, 3, large_filter, large_features
                , dll::momentum
                , dll::batch_size<8>
                , dll::weight_decay<dll::decay_type::L2>
                , dll::visible<dll::unit_type::GAUSSIAN>
                //, dll::sparsity<dll::sparsity_method::LEE>
            >::rbm_t
            >
            //, dll::watcher<dll::opencv_dbn_visualizer>
            >::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    std::cout << "DBN is " << sizeof(dbn_t) << " bytes long" << std::endl;
    std::cout << "DBN input is " << dbn->input_size() << std::endl;
    std::cout << "DBN output is " << dbn->output_size() << std::endl;

    dbn->layer<0>().learning_rate /= 100;
    //dbn->layer<0>().pbias = 0.01;
    //dbn->layer<0>().pbias_lambda = 0.001;

    //dbn->load("icdar_3d.dbn");

    dbn->pretrain(training_patches, 20);
    dbn->store("icdar_3d.dbn");

    svm::model model;

    //TODO Maybe think of scaling features

    //Make it quiet
    //svm::make_quiet();

    svm::problem training_problem;

    //Train and test on training set
    {

        {
            std::vector<std::vector<float>> features;
            std::vector<uint8_t> labels;

            large_svm_extract(*dbn, dataset.training_labels, dataset.training_images,
                training_images_padded, training_patches, features, labels, 50000, g);

            std::cout << features.size() << " training feature vectors extracted" << std::endl;
            std::cout << count_one(labels) / static_cast<double>(labels.size()) << "% text pixel" << std::endl;

            svm_scale(features);

            std::cout << "Make SVM Problem" << std::endl;

            training_problem = svm::make_problem(labels, features);
        }

        auto mnist_parameters = svm::default_parameters();

        mnist_parameters.svm_type = C_SVC;
        mnist_parameters.kernel_type = RBF;
        mnist_parameters.probability = 0;
        mnist_parameters.shrinking = 0;
        mnist_parameters.C = 1;
        mnist_parameters.gamma = 1.0 / 2.1;

        //Make sure parameters are not too messed up
        if(!svm::check(training_problem, mnist_parameters)){
            return 1;
        }

        model = svm::train(training_problem, mnist_parameters);

        std::cout << model.classes() << " classes found" << std::endl;

        std::cout << "Test on training set" << std::endl;
        svm::test_model(training_problem, model);
    }

    //Test on test set
    {
        svm::problem test_problem;

        {
            std::vector<std::vector<float>> features;
            std::vector<uint8_t> labels;

            large_svm_extract(*dbn, dataset.test_labels, dataset.test_images, test_images_padded, test_patches, features, labels, 50000, g);

            std::cout << features.size() << " test feature vectors extracted" << std::endl;
            std::cout << count_one(labels) / static_cast<double>(labels.size()) << "% text pixel" << std::endl;

            svm_scale(features);

            test_problem = svm::make_problem(labels, features);
        }

        std::cout << "Test on test set" << std::endl;
        svm::test_model(test_problem, model);
    }

    return 0;
}

int main(){
    return large_wise();
}
