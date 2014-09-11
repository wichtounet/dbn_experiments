//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/[]){
    dll::layer<
            28 * 28, 200,
            dll::momentum,
            dll::batch_size<25>,
            dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read dataset" << std::endl;
        return 1;
    }

    dataset.training_images.resize(1000);
    dataset.training_labels.resize(1000);

    mnist::normalize_dataset(dataset);

    rbm.train(dataset.training_images, 100);

    return 0;
}
