//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <iomanip>
#include <memory>

#include "dbn/dbn.hpp"
#include "dbn/layer.hpp"
#include "dbn/conf.hpp"
#include "dbn/labels.hpp"
#include "dbn/test.hpp"

#include "mnist/mnist_reader.hpp"

#include "utils.hpp"

namespace {

template<typename DBN, typename Dataset, typename P>
void test_all(DBN& dbn, Dataset& dataset, P&& predictor){
    std::cout << "Start testing" << std::endl;

    std::cout << "Training Set (" << dataset.training_images.size() << ")" << std::endl;
    auto error_rate = dbn::test_set(dbn, dataset.training_images, dataset.training_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;

    std::cout << "Test Set (" << dataset.test_images.size() << ")" << std::endl;
    error_rate =  dbn::test_set(dbn, dataset.test_images, dataset.test_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
}

template<typename DBN, typename Image>
void display(const DBN& dbn, const Image& image){
    auto weights = dbn->predict_weights(image);

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
            auto weights = dbn->predict_weights(images[i]);
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
    auto prob = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        }

        if(command == "load"){
            load = true;
        }

        if(command == "prob"){
            load = true;
            prob = true;
        }
    }

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    normalize(dataset.training_images);
    normalize(dataset.test_images);

    if(simple){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 50, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 60, 100>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

        test_all(dbn, dataset, dbn::label_predictor());
    } else {
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 100, true, true, true, dbn::Type::GAUSSIAN, dbn::Type::NRLU>, 28 * 28, 500>,
            //dbn::layer<dbn::conf<true, 100, false, true, true, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>,
            //dbn::layer<dbn::conf<true, 100, false, true>, 400, 1100>,
            dbn::layer<dbn::conf<true, 100, false, true, true, dbn::Type::SIGMOID, dbn::Type::SOFTMAX>, 500, 10>> dbn_t;

        auto labels = dbn::make_fake(dataset.training_labels);

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn_gray.dat", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(dataset.training_images, 10);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(dataset.training_images, labels, 2, 1000);

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
            test_all(dbn, dataset, dbn::predictor());
        }
    }

    return 0;
}
