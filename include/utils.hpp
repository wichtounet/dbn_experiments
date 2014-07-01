//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>

template<typename Container>
void scale_each(Container& values){
    for(auto& vec : values){
        for(auto& v : vec){
            v /= 255.0;
        }
    }
}

template<typename Container>
void binarize_each(Container& values, double threshold = 30.0){
    for(auto& vec : values){
        for(auto& v : vec){
            v = v > threshold ? 1.0 : 0.0;
        }
    }
}

template<typename Container>
double mean(const Container& container){
    double mean = 0.0;
    for(auto& value : container){
        mean += value;
    }
    return mean / container.size();
}

template<typename Container>
double stddev(const Container& container, double mean){
    double std = 0.0;
    for(auto& value : container){
        std += (value - mean) * (value - mean);
    }
    return std::sqrt(std / container.size());
}

template<typename Container>
void normalize(Container& values){
    for(auto& vec : values){
        //zero-mean
        auto m = mean(vec);
        for(auto& v : vec){
            v -= m;
        }
        //unit variance
        auto s = stddev(vec, 0.0);
        for(auto& v : vec){
            v /= s;
        }
    }
}

#endif
