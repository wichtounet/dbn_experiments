#include "icdar/icdar_reader.hpp"

int main(){

    auto dataset = icdar::read_2013_dataset(
        "/home/wichtounet/datasets/icdar_2013_natural/train",
        "/home/wichtounet/datasets/icdar_2013_natural/test"/*,
        100, 100*/);

    return 0;
}
