//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

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

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    mnist::binarize_dataset(dataset);

    if(simple){
        typedef dll::dbn<
            dll::layer<28 * 28, 100, dll::in_dbn, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>,
            dll::layer<100, 100, dll::in_dbn, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>,
            dll::layer<110, 200, dll::in_dbn, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

        test_all(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    } else {
        typedef dll::dbn<
            dll::layer<28 * 28, 100, dll::in_dbn, dll::momentum, dll::batch_size<50>, dll::init_weights>,
            dll::layer<100, 200, dll::in_dbn, dll::momentum, dll::batch_size<50>>,
            dll::layer<200, 10, dll::in_dbn, dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::EXP>>> dbn_t;

        auto dbn = make_unique<dbn_t>();

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn.dat", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(dataset.training_images, 5);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 1000);

            std::ofstream os("dbn.dat", std::ofstream::binary);
            dbn->store(os);
        }

        test_all(dbn, dataset.training_images, dataset.training_labels, dll::predictor());
    }

    return 0;
}
