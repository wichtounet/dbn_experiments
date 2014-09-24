//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#define DLL_SVM_SUPPORT

#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

template<typename DBN, typename P>
void test_all(DBN& dbn, std::vector<etl::dyn_vector<double>>& training_images, const std::vector<uint8_t>& training_labels, P&& predictor){
    auto test_images = mnist::read_test_images<std::vector, etl::dyn_vector, double>();
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
    auto svm = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        }

        if(command == "load"){
            load = true;
        }

        if(command == "svm"){
            svm = true;
        }
    }

    auto dataset = mnist::read_dataset<std::vector, etl::dyn_vector, double>(100);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    mnist::binarize_dataset(dataset);

    if(simple){
        typedef dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<28 * 28, 100, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<100, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
                dll::rbm_desc<110, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
            >>::dbn_t dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

        test_all(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    } else if(svm){
        typedef dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<28 * 28, 400, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
                dll::rbm_desc<400, 600, dll::momentum, dll::batch_size<50>>::rbm_t
        >>::dbn_t dbn_t;

        auto dbn = make_unique<dbn_t>();

        dbn->display();

        dbn->pretrain(dataset.training_images, 20);

        if(!dbn->svm_train(dataset.training_images, dataset.training_labels)){
            std::cout << "SVM training failed" << std::endl;
        }

        test_all(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    } else {
        typedef dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
                dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<50>>::rbm_t,
                dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
        >, dll::watcher<dll::default_dbn_watcher>>::dbn_t dbn_t;

        auto dbn = make_unique<dbn_t>();

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

        test_all(dbn, dataset.training_images, dataset.training_labels, dll::predictor());
    }

    return 0;
}
