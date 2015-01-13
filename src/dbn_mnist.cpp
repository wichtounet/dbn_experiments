//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <iomanip>
#include <memory>

#define DLL_SVM_SUPPORT

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

template<typename DBN, typename Dataset, typename P>
void test_all(DBN& dbn, Dataset& dataset, P&& predictor){
    std::cout << "Start testing" << std::endl;

    std::cout << "Training Set" << std::endl;
    auto error_rate = dll::test_set(dbn, dataset.training_images, dataset.training_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;

    std::cout << "Test Set" << std::endl;
    error_rate =  dll::test_set(dbn, dataset.test_images, dataset.test_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
}

template<typename DBN, typename Image>
void display(const DBN& dbn, const Image& image){
    auto weights = dbn->activation_probabilities(image);

    for(std::size_t i = 0; i < 28; ++i){
        for(std::size_t j = 0; j < 28; ++j){
            std::cout << (image[i * 28 + j] * 255.0 > 10.0 ? 1.0 : 0.0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Activation probabilities" << std::endl;
    for(std::size_t i = 0; i < 10; ++i){
        auto w = weights[i];
        std::cout << "\t" << i << ":" << w << "(" << w * 100.0 << "%)" << std::endl;
    }
    std::cout  << "Answer" << dbn->predict(image) << std::endl;
}

template<typename DBN, typename Images, typename Labels>
void errors(const DBN& dbn, Images& images, Labels& labels){
    std::size_t types[100];
    std::size_t sources[10];
    std::size_t errors = 0;
    std::size_t second_errors = 0;

    std::fill(&types[0], &types[100], 0);
    std::fill(&sources[0], &sources[10], 0);

    for(std::size_t i = 0; i < images.size(); ++i){
        auto predicted = dbn->predict(images[i]);

        if(predicted != labels[i]){
            auto weights = dbn->activation_probabilities(images[i]);
            auto max = 0.0;
            auto second = 0;

            for(std::size_t j = 0; j < weights.size(); ++j){
                if(j != predicted && weights[j] > max){
                    second = j;
                    max = weights[j];
                }
            }

            if(second == labels[i]){
                ++second_errors;
            }

            ++errors;
            ++types[labels[i] * 10 + predicted];
            ++sources[labels[i]];
        }
    }

    std::cout << "Error rate " << 100.0 * (static_cast<double>(errors) / images.size()) << std::endl;
    std::cout << errors << " errors " << " / " << images.size() << std::endl;
    std::cout << "Second guess error rate " << 100.0 * ((static_cast<double>(errors) - second_errors) / images.size()) << std::endl;
    std::cout << "Second guess was right " << second_errors << " / " << images.size() << std::endl;

    std::cout << "Errors sources: ";
    for(std::size_t i = 0; i < 10; ++i){
        std::cout << i << ":" << sources[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Error matrix" << std::endl;
    std::cout << "   ";
    for(std::size_t i = 0; i < 10; ++i){
        std::cout << std::setw(3) << i << " ";
    }
    std::cout << std::endl;
    for(std::size_t i = 0; i < 10; ++i){
        std::cout << i << ": ";
        for(std::size_t j = 0; j < 10; ++j){
            std::cout << std::setw(3) << types[i * 10 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    auto simple = false;
    auto load = false;
    auto svm = false;
    auto grid = false;
    auto gray = false;
    auto prob = false;
    auto view = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        } else if(command == "load"){
            load = true;
        } else if(command == "svm"){
            svm = true;
        } else if(command == "grid"){
            grid = true;
        } else if(command == "gray"){
            gray = true;
        } else if(command == "prob"){
            prob = true;
        } else if(command == "view"){
            view = true;
        }
    }

    auto dataset = mnist::read_dataset<std::vector, etl::dyn_vector, double>(100);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    //Gray input
    if(gray){
        mnist::normalize_dataset(dataset);

        if(simple){
            typedef dll::dbn_desc<
                dll::dbn_label_layers<
                dll::rbm_desc<28 * 28, 100, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<100, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<110, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
                    >>::dbn_t dbn_simple_t;

            auto dbn = std::make_shared<dbn_simple_t>();

            dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

            test_all(dbn, dataset, dll::label_predictor());
        } else {
            typedef dll::dbn_desc<
                dll::dbn_layers<
                dll::rbm_desc<28 * 28, 300, dll::momentum, dll::batch_size<100>, dll::init_weights, dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t,
                dll::rbm_desc<300, 500, dll::momentum, dll::batch_size<100>>::rbm_t,
                dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<100>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
                    >>::dbn_t dbn_t;

            auto dbn = std::make_unique<dbn_t>();

            dbn->display();

            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn_gray.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 20);

                std::cout << "Start fine-tuning" << std::endl;
                dbn->fine_tune(dataset.training_images, dataset.training_labels, 2, 1000);

                std::ofstream os("dbn_gray.dat", std::ofstream::binary);
                dbn->store(os);
            }

            if(prob){
                display(dbn, dataset.training_images[256]);
                display(dbn, dataset.training_images[512]);
                display(dbn, dataset.training_images[666]);
                display(dbn, dataset.training_images[1024]);
                display(dbn, dataset.training_images[2048]);

                std::cout << std::endl << "Results on training dataset" << std::endl;
                errors(dbn, dataset.training_images, dataset.training_labels);

                std::cout << std::endl << "Results on test dataset" << std::endl;
                errors(dbn, dataset.test_images, dataset.test_labels);
            } else {
                test_all(dbn, dataset, dll::predictor());
            }
        }
    } else if(view){
        mnist::binarize_dataset(dataset);

        typedef dll::dbn_desc<
            dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<50>>::rbm_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
                >, dll::watcher<dll::opencv_dbn_visualizer>>::dbn_t dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(dataset.training_images, 10);

        std::cout << "Start fine-tuning" << std::endl;
        dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 1000);

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);
    } else {
        mnist::binarize_dataset(dataset);

        if(simple){
            typedef dll::dbn_desc<
                dll::dbn_label_layers<
                dll::rbm_desc<28 * 28, 100, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<100, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<110, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
                    >>::dbn_t dbn_simple_t;

            auto dbn = std::make_shared<dbn_simple_t>();

            dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

            test_all(dbn, dataset, dll::label_predictor());
        } else if(svm){ typedef dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<28 * 28, 400, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
                dll::rbm_desc<400, 600, dll::momentum, dll::batch_size<50>>::rbm_t
                    >>::dbn_t dbn_t;

            auto dbn = std::make_unique<dbn_t>();

            dbn->display();

            if(load){
                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                dbn->pretrain(dataset.training_images, 20);

                if(grid){
                    dbn->svm_grid_search(dataset.training_images, dataset.training_labels);
                } else {
                    if(!dbn->svm_train(dataset.training_images, dataset.training_labels)){
                        std::cout << "SVM training failed" << std::endl;
                    }
                }

                std::ofstream os("dbn.dat", std::ofstream::binary);
                dbn->store(os);
            }

            if(!grid){
                test_all(dbn, dataset, dll::svm_predictor());
            }
        } else {
            typedef dll::dbn_desc<
                dll::dbn_layers<
                dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
                dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<50>>::rbm_t,
                dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
                    >, dll::watcher<dll::default_dbn_watcher>>::dbn_t dbn_t;

            auto dbn = std::make_unique<dbn_t>();

            dbn->display();

            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 10);

                std::cout << "Start fine-tuning" << std::endl;
                dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 1000);

                std::ofstream os("dbn.dat", std::ofstream::binary);
                dbn->store(os);
            }

            test_all(dbn, dataset, dll::predictor());
        }
    }

    return 0;
}
