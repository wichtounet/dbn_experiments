//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#define DLL_SVM_SUPPORT

#include "dll/conv_dbn.hpp"

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
    auto grid = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
            load = true;
        }

        if(command == "svm"){
            svm = true;
        }

        if(command == "grid"){
            grid = true;
        }
    }

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(5000);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        return 1;
    }

    mnist::binarize_dataset(dataset);

    if(svm){
        typedef dll::conv_dbn_desc<
            dll::dbn_layers<
            dll::conv_rbm_desc<28, 17, 40, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::rbm_t//,
            //dll::conv_rbm_desc<12, 6, 40, dll::momentum, dll::batch_size<50>, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>>::rbm_t
                >>::dbn_t dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->layer<0>().pbias = 0.05;
        dbn->layer<0>().pbias_lambda = 100;

        //dbn->layer<1>().pbias = 0.05;
        //dbn->layer<1>().pbias_lambda = 100;

        std::cout << dbn->rbm_output<0>() << std::endl;
        //std::cout << dbn->rbm_input<1>() << std::endl;

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn.dat", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(dataset.training_images, 100);

            if(grid){
                svm::rbf_grid grid;
                grid.type = svm::grid_search_type::LINEAR;
                grid.c_first = 0.0;
                grid.c_last = 20;
                grid.c_steps = 25;
                grid.gamma_first = 0;
                grid.gamma_last = 200;
                grid.gamma_steps = 12;

                dbn->svm_grid_search(dataset.training_images, dataset.training_labels, 5, grid);
            } else {
                auto parameters = dll::default_svm_parameters();
                //parameters.C = 6.22222;
                //parameters.gamma = 88;

                if(!dbn->svm_train(dataset.training_images, dataset.training_labels, parameters)){
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
        typedef dll::conv_dbn_desc<
            dll::dbn_layers<
            dll::conv_rbm_desc<28, 12, 40, dll::momentum, dll::batch_size<50>>::rbm_t,
            dll::conv_rbm_desc<12, 6, 40, dll::momentum, dll::batch_size<50>>::rbm_t
                >>::dbn_t dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

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

    return 0;
}
