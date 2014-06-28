//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "etl/fast_vector.hpp"

#include <iostream>

int main(){
    etl::fast_vector<double, 3> a(1.0);
    auto e = a + etl::fast_vector<double, 3>(2.0) + etl::fast_vector<double, 3>(4.0) + etl::fast_vector<double, 3>(3.0);

    std::cout << e[2] << std::endl;

    auto f = (etl::fast_vector<double, 3>(1.33) * 2.0) * 9.0 * etl::fast_vector<double, 3>(9.23);

    std::cout << f[2] << std::endl;

    return 0;
}
