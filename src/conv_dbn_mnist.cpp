//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>
#include <random>

#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

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

int main(int argc, char* argv[]){
    auto load = false;
    auto svm = false;
    auto mp = false;
    auto shuffle = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
            load = true;
        } else if(command == "svm"){
            svm = true;
        } else if(command == "mp"){
            mp = true;
        } else if(command == "shuffle"){
            shuffle = true;
        }
    }

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(1000);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    if(shuffle){
        std::default_random_engine rng;

        cpp::parallel_shuffle(
            dataset.training_images.begin(), dataset.training_images.end(),
            dataset.training_labels.begin(), dataset.training_labels.end(),
            rng);
    }

    //dataset.training_images.resize(10000);
    //dataset.training_labels.resize(10000);

    mnist::binarize_dataset(dataset);

    if(mp){
        typedef dll::dbn_desc<
            dll::dbn_layers<
            dll::conv_rbm_mp_desc_square<1, 28, 40, 18, 2, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::layer_t,
            dll::conv_rbm_mp_desc_square<40, 9, 40, 6, 2, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::layer_t
                >, dll::svm_concatenate>::dbn_t dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->layer_get<0>().pbias = 0.05;
        dbn->layer_get<0>().pbias_lambda = 50;

        dbn->layer_get<1>().pbias = 0.05;
        dbn->layer_get<1>().pbias_lambda = 100;

        dbn->display();

        std::cout << "RBM1: Input: " << dbn->layer_get<0>().input_size() << std::endl;
        std::cout << "RBM1: Output: " << dbn->layer_get<0>().output_size() << std::endl;

        std::cout << "RBM2: Input: " << dbn->layer_get<1>().input_size() << std::endl;
        std::cout << "RBM2: Output: " << dbn->layer_get<1>().output_size() << std::endl;

        if(svm){
            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 50);
            }

            auto parameters = dll::default_svm_parameters();
            //parameters.C = 2.09091;
            //parameters.gamma = 0.272727;

            if(!dbn->svm_train(dataset.training_images, dataset.training_labels, parameters)){
                std::cout << "SVM training failed" << std::endl;
            }

            std::ofstream os("dbn.dat", std::ofstream::binary);
            dbn->store(os);

            test_all(dbn, dataset, dll::svm_predictor());
        } else {
            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 5);

                std::ofstream os("dbn.dat", std::ofstream::binary);
                dbn->store(os);
            }
        }
    } else {
        typedef dll::dbn_desc<
            dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 40, 17, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::layer_t,
            dll::conv_rbm_desc_square<40, 17, 40, 12, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::layer_t
                >>::dbn_t dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->layer_get<0>().pbias = 0.05;
        dbn->layer_get<0>().pbias_lambda = 50;

        dbn->layer_get<1>().pbias = 0.05;
        dbn->layer_get<1>().pbias_lambda = 100;

        dbn->display();

        std::cout << "RBM1: Input: " << dbn->layer_get<0>().input_size() << std::endl;
        std::cout << "RBM1: Output: " << dbn->layer_get<0>().output_size() << std::endl;

        std::cout << "RBM2: Input: " << dbn->layer_get<1>().input_size() << std::endl;
        std::cout << "RBM2: Output: " << dbn->layer_get<1>().output_size() << std::endl;

        if(svm){
            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 50);
            }

            auto parameters = dll::default_svm_parameters();
            //parameters.C = 2.09091;
            //parameters.gamma = 0.272727;

            if(!dbn->svm_train(dataset.training_images, dataset.training_labels, parameters)){
                std::cout << "SVM training failed" << std::endl;
            }

            std::ofstream os("dbn.dat", std::ofstream::binary);
            dbn->store(os);

            test_all(dbn, dataset, dll::svm_predictor());
        } else {
            if(load){
                std::cout << "Load from file" << std::endl;

                std::ifstream is("dbn.dat", std::ifstream::binary);
                dbn->load(is);
            } else {
                std::cout << "Start pretraining" << std::endl;
                dbn->pretrain(dataset.training_images, 5);

                std::ofstream os("dbn.dat", std::ofstream::binary);
                dbn->store(os);
            }
        }
    }

    return 0;
}
