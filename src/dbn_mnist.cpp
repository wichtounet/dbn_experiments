//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dll/dbn.hpp"
#include "dll/layer.hpp"
#include "dll/labels.hpp"
#include "dll/test.hpp"

#include "mnist/mnist_reader.hpp"

namespace {

template<typename Container>
void binarize_each(Container& values){
    for(auto& vec : values){
        for(auto& v : vec){
            v = v > 10.0 ? 1.0 : 0.0;
        }
    }
}

template<typename DBN, typename P>
void test_all(DBN& dbn, std::vector<vector<double>>& training_images, const std::vector<uint8_t>& training_labels, P&& predictor){
    auto test_images = mnist::read_test_images<std::vector, vector, double>();
    auto test_labels = mnist::read_test_labels<std::vector>();

    if(test_images.empty() || test_labels.empty()){
        std::cout << "Impossible to read test set" << std::endl;
        return;
    }

    std::cout << "Start testing" << std::endl;

    std::cout << "Training Set" << std::endl;
    auto error_rate = dll::test_set(dbn, training_images, training_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;

    std::cout << "Test Set" << std::endl;
    error_rate =  dll::test_set(dbn, test_images, test_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    auto simple = false;
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        }

        if(command == "load"){
            load = true;
        }
    }

    auto training_images = mnist::read_training_images<std::vector, vector, double>();
    auto training_labels = mnist::read_training_labels<std::vector>();

    if(training_images.empty() || training_labels.empty()){
        return 1;
    }

    binarize_each(training_images);

    if(simple){
        typedef dll::dbn<
            dll::layer<28 * 28, 100, dll::in_dbn, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::DecayType::L2>>,
            dll::layer<100, 100, dll::in_dbn, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::DecayType::L2>>,
            dll::layer<110, 200, dll::in_dbn, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::DecayType::L2>>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(training_images, training_labels, 10, 5);

        test_all(dbn, training_images, training_labels, dll::label_predictor());
    } else {
        typedef dll::dbn<
            dll::layer<28 * 28, 30, dll::in_dbn, dll::momentum, dll::batch_size<100>, dll::init_weights>,
            dll::layer<30, 30, dll::in_dbn, dll::momentum, dll::batch_size<100>>,
            dll::layer<30, 10, dll::in_dbn, dll::momentum, dll::batch_size<100>, dll::hidden_unit<dll::Type::EXP>>> dbn_t;

        auto labels = dll::make_fake(training_labels);

        auto dbn = make_unique<dbn_t>();

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn.dat", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(training_images, 5);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(training_images, labels, 5, 1000);

            std::ofstream os("dbn.dat", std::ofstream::binary);
            dbn->store(os);
        }

        test_all(dbn, training_images, training_labels, dll::predictor());
    }

    return 0;
}
